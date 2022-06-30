from cmath import inf
import os
import queue
from struct import unpack
import time
import smbus
import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal
import scipy.integrate as intg
import ahrs
from matplotlib import pyplot as plt
import logging
from scipy.fftpack import fft, ifft, fftfreq, fftshift

import ghostipy as gp
class daq:
    def __init__(self):
        self.__name__ = "daq"
        try:
            self.bus = smbus.SMBus(1)
            logging.info("I2C bus successfully initialized")
        except Exception as e:
            logging.warning("I2C connection Error: ", exc_info=e)
            logging.info("I2C bus unable to initialize")

        # list of variables
        self.root: str = os.getcwd()
        self.sessionname: str = None
        self.devices: dict = {}
        self.settings: dict = {}
        self.fs: float = 1666  # sampling frequency
        self.dt: float = 1/self.fs  # sampling period
        self.running: bool = False
        self.raw: bool = False
        self.data_rate: int = 10  # 8=1666Hz 9=3330Hz 10=6660Hz
        self.data_range: list[int] = [1, 3]  # [16G, 2000DPS]
        
        self.init_devices()

    def init_devices(self):
        for address in range(128):
            try:
                self.bus.read_byte(address)
                logging.info("Found device at address: 0x%02X", address)
                if address == 0x6a or address == 0x6b:
                    num = str(107-address)
                    self.devices[address] = {'reg': 0x22, 
                                             'len': 12, 
                                             'fmt': '<hhhhhh', 
                                             'lbl': ['Gx_'+num, 'Gy_'+num, 'Gz_'+num, 'Ax_'+num, 'Ay_'+num, 'Az_'+num],
                                             'cal': None}
                    self.settings[address] = {0x10:(self.data_rate << 4 | self.data_range[0] << 2 | 1 << 1),
                                              0x11: (self.data_rate << 4 | self.data_range[1] << 2),
                                              0x12: 0x44,
                                              0x13: 1 << 1,
                                              0x15: 0b011,
                                              0X17: 0x44}  # [0x44 is hardcoded acording to LSM6DSO datasheet](0b000 << 5
                    self.set_device(address)
                        
                elif address == 0x48:
                    self.devices[address] = {'reg': 0x00, 
                                             'len': 2, 
                                             'fmt': '>h', 
                                             'lbl': ['cur'], 
                                             'cal': None}
                    config = (3 << 9 | 0 << 8 | 4 << 5 | 3)
                    self.settings[address] = {0x01: [(config >> 8 & 0xFF), (config & 0xFF)]}
                    self.set_device(address)
                elif address == 0x36:
                    self.devices[address] = {'reg': 0x0C, 
                                             'len': 2, 
                                             'fmt': '>H', 
                                             'lbl': ['rot'], 
                                             'cal': None}
                    self.settings[address] = {}

            except:
                logging.debug("can`t connect address: : 0x%02X", address)
                pass

    def set_device(self, address: int) -> bool: 
        try:
            for reg, value in self.settings[address].items():
                self.bus.write_byte_data(address, reg, value)    
            logging.info("Set device address: : 0x%02X", address)
        except Exception as e:
            logging.debug("Could not set device address: : 0x%02X", address, exc_info=e)
            return False
        return True

    def calibrate_imu(self, acc: np.array=None, gyr: np.array=None, Ts: float=None, Td: float=None, fs: float=None, name: str=None) -> tuple or bool: 
        Ns: int = Ts*fs
        Nd: int = Td*fs
        
        if acc is not None:
            acc_mean = np.array(
                [np.mean(acc[Ns*n:Ns*(n+1)-1], axis=0) for n in range(6)]).T
            #acc_std = np.array([np.std(acc[Ns*n:Ns*(n+1)-1], axis=0) for n in range(6)])
            acc_bias = np.array([(acc_mean[n, :].max()+acc_mean[n, :].min())/2 for n in range(3)], ndmin=2).T
            acc_ub = acc_mean-acc_bias
            acc_grv = (acc_ub > 1000)*9.81 + (acc_ub < -1000)*-9.81
            acc_KS = acc_grv@np.linalg.pinv(acc_ub)

        if gyr is not None:
            gyr_mean = np.array([np.mean(gyr[Ns*n:Ns*(n+1)-1], axis=0) for n in range(6)]).T
            # gyr_std = np.std(gyr[:6*Ns, :]).reshape((3, 1))
            gyr_bias = np.mean(gyr[:6*Ns, :], axis=0).reshape((3, 1))
            gyr_rot = np.zeros_like(gyr_mean)
            # gyr_ub = gyr_mean-gyr_bias
            # gyr_KS = (np.ones_like(gyr_ub)*gyr_std)@np.linalg.pinv(gyr_ub)

            gyr_d = gyr[6*Ns:, :]-gyr_bias.T
            # gyr_rot = np.array([np.sum(gyr_d[Nd*n:Nd*(n+1)-1], axis=0) for n in range(3)]).T*dt
            gyr_rot = np.array([np.mean(gyr_d[Nd*n:Nd*(n+1)-1], axis=0) for n in range(3)]).T
            # gyr_ref = (gyr_rot>1000)*np.pi + (gyr_rot<-1000)*-np.pi
            gyr_ref = ((gyr_rot > 100)*np.pi + (gyr_rot < -100)*-np.pi)/(Nd/fs)
            gyr_KS = gyr_ref@np.linalg.inv(gyr_rot)
            if name is not None:
                try:
                    pd.DataFrame([acc_KS, acc_bias, gyr_KS, gyr_bias]).to_csv('./sensors/'+name+'.csv')
                    return True
                except:
                    logging.warning("ERROR: unable to save calibration data")
                    return False
            else:
                return (acc_KS, acc_bias) (gyr_KS, gyr_bias)   

    def translate_imu(self, acc: np.array=None, gyr=None, fs=None, acc_param=None, gyr_param=None):
        acc_t = acc_param[0]@(acc.T-acc_param[1])
        gyr_t = gyr_param[0]@(gyr.T-gyr_param[1])
        return np.hstack((gyr_t, acc_t))

    def pull_data(self, durr: float=0, devices=None, raw=False, rtrn_array=False) -> dict[pd.DataFrame]:
        q = queue.Queue()
        logging.info('Start pulling')
        if durr == 0:
            N = inf
        else:
            N = durr*self.fs
        if devices is None:
            devices = self.devices
               
        logging.info('activate dq.running')
        self.running = True

        value = []
        for addr, val in devices.items():
            value.append([addr, val['reg'], val['len']])

        ii=0
        t0 = ti = tf = time.perf_counter()
        while self.running and ii<N:
            tf = time.perf_counter()
            if tf-ti>=self.dt:
                ti = time.perf_counter()
                ii+=1    
                for val in value:
                    try:
                        q.put(self.bus.read_i2c_block_data(val[0], val[1], val[2]))                                     
                    except:
                        q.put((0,))
                        pass
        t1 = time.perf_counter()
        logging.info("Pulled data in %.6f s" % (t1-t0))

        return self.dequeue_data(q, raw, rtrn_array)

    def dequeue_data(self,q: queue.Queue, raw: bool=False, rtrn_array=False) -> dict[pd.DataFrame]:
        logging.info('start dequeueing...')
        data = {}
        for addr, val in self.devices.items():
            data[addr] = []
        logging.info('start looping through queue')    
        while not q.empty():       #block dequeueing data
            for addr, val in self.devices.items():
                qq = q.get()
                if any(qq):
                    data[addr].append(unpack(val['fmt'], bytearray(qq)))
                else:
                    logging.info('dequeue data error, atribute NaN')
                    data[addr].append((np.NaN,)*(val['len']//2))
                    pass
                
        logging.info('Dequeue successfull')
        DFs = {}
        for addr, val in self.devices.items():      #block translate from raw to meaningful data
            array = np.array(data[addr], ndmin=2)
            name = 'Sensor_{}'.format(hex(addr))
            if rtrn_array:
                return array
            logging.info('inserting values of {} into dict'.format(name))
            if val['cal'] is not None and not raw:
                if addr == 0x6a or addr == 0x6b:
                    
                    params = pd.read_csv(self.root+'/sensors/'+val['cal']+'.csv')
                    array = self.translate_imu(acc=array[:,3:],
                                                    gyr=array[:,:3],
                                                    fs=self.fs,
                                                    acc_param=(params['acc_p']),
                                                    gyr_param=(params['gyr_p']))
                elif addr == 0x36 or addr==0x48:
                   
                    scale = pd.read_csv('./sensors/'+val['cal']+'.csv')
                    array = array*scale 
            DFs[name] = pd.DataFrame(array,
                       index={'t': np.arange(len(data[addr]))*self.dt},
                       columns=val['lbl'])
            logging.info('Inserted {} data into dict DFs'.format(name)) 
        
        return DFs
        
    def save_data(self, DFs: dict[pd.DataFrame], session: str=None):
        logging.info('start seving data')
        
        if session is None:
            session = self.sessionname
        
        try:
            os.chdir(self.root+'/data/{}'.format(session))
        except Exception as e:
            logging.info("path", exc_info=e)
            os.mkdir(self.root+'/data/{}'.format(session))
            pass
        num: int = os.listdir(path).__len__()
        path = self.root+'/data/{}/data_{:03d}/'.format(session, num)
        
        
        try:
            for name, df in DFs:
                filename = path+'{}.csv'.format(name)
                df.to_csv(filename)
                logging.info("saved data to {}".format(filename))
            return True
        except Exception as e:
            logging.error('Could not save data_{:03d}'.format(num), exc_info=e)
            return False


class dsp:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__name__ = "dsp"
        self.fs = 1666
        self.dt = 1/self.fs

    def PSD(self, df, fs, units='unid.', fig=None, line='-', linewidth=1, S_ref=1, return_fig=True):
        f, Pxx = signal.welch(df, fs, nperseg=fs//4, noverlap=fs//8, window='hann', average='mean', scaling='density', detrend=False, axis=0)
        if return_fig is False:
            return f, 20*np.log10(abs(Pxx))
        else:
            if fig==None:
                fig, ax = plt.subplots()
                fig=plt.figure()
            plt.subplot(211)
            # plt.title('Sinal')
            plt.xlabel('Tempo [s]')
            plt.ylabel('Amplitude [{}]'.format(units))
            plt.plot(df, line, linewidth=linewidth)
            plt.legend(df.columns)
            plt.grid(True, which='both')
            plt.subplot(212)
            # plt.title('Densidade do Espectro de Potência')
            plt.plot(f, 20*np.log10(abs(Pxx/S_ref)))
            plt.xlim((1,800))
            plt.xlabel('Frequência [Hz]')
            plt.ylabel('PSD [dB/Hz] ref= {} {}'.format(S_ref,units))
            plt.grid(True, which='both')  
            plt.tight_layout() 
            return fig


    def WOLA(self, data, factor=1, NFFT=None):
        if NFFT is None:
            n = self.fs//4
        else:
            n = NFFT
        try:
            width = data.shape[1]
        except:
            width = 0
        _data = np.vstack((np.zeros((2*n,width)), data, np.zeros((2*n,width))))
        N = len(_data)
        w = scipy.windows.hann(n).reshape((n,1))
        Data = np.zeros_like(_data, dtype=complex)
        for ii in range(0, N-n, n//2):
            Y = _data[ii:ii+n,:]*w
            k =  (1j*2*np.pi*fftfreq(len(Y), self.dt).reshape((n,1)))
            y = (ifft(np.vstack((np.zeros((factor,width)),fft(Y, axis=0)[factor:]/(k[factor:]))), axis=0))
            Data[ii:ii+n,:] += y
        return np.real(Data[2*n:-2*n,:])

    def FDI(self, data):
        Y = fft(data, axis=0)
        f = fftfreq(data, self.dt)
        Y[0,:] = 0
        y=ifft(Y/(1j*2*np.pi*f), axis=0)
        return y[:len(data),:]


    def spect(self, df: pd.DataFrame=None, dbmin=80, print: bool=True, freqlims: tuple=(1,800)):
        for frame in df:
            f, t, Sxx = scipy.signal.spectrogram(df[frame], fs=self.fs, axis=0, scaling='spectrum', nperseg=self.fs//2, noverlap=self.fs//4, detrend=False, mode='psd', window='hann')
            Sxx[Sxx==0] = 10**(-20)
            if print==True:
                plt.figure()
                plt.pcolormesh(t, f, 20*np.log10(abs(Sxx)), shading='gouraud',  cmap='turbo',vmax=20*np.log10(abs(Sxx)).max(), vmin=20*np.log10(abs(Sxx)).max()-dbmin)
                plt.ylim(freqlims)
                plt.colorbar()
                plt.title(frame)
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.tight_layout()
                plt.show()
            else:
                return t, f, np.flip(20*np.log10(abs(Sxx)), axis=0)

    def FDD(self, _data, factor=1, NFFT=None):
        if NFFT is None:
            n = self.fs
        else:
            n = NFFT
        N = len(_data)
        try:
            width = _data.shape[1]
        except:
            _data = _data.reshape((N,1))
            width = 1
        
        w = signal.windows.hann(n).reshape((n,1))
        Data = np.zeros_like(_data, dtype=complex)
        for ii in range(0, N-n, n//2):
            Y = _data[ii:ii+n,:]*w
            k =  (1j*2*np.pi*fftfreq(len(Y), self.dt).reshape((n,1)))
            y = (ifft(np.vstack((np.zeros((factor,width)),fft(Y, axis=0)[factor:]*(k[factor:]))), axis=0))
            Data[ii:ii+n,:] += y
        return np.real(Data)        

    def TDI(self, data): 
        N = len(data)
        if len(data.shape)==1:
            data = data.reshape((N,1))
        data = self.zmean(data)
        dataout = np.zeros_like(data)
        dataout[0,:] = data[0,:]*self.dt/2
        for ii in range(1,N):
            dataout[ii,:] = intg.simpson(data[0:ii,:], dx=self.dt, axis=0)
            #dataout[ii,:] = intg.trapz(data[0:ii,:], dx=self.dt, axis=0)
        return dataout

    def zmean(self, _data):
        return np.real(ifft(np.vstack((np.zeros((2,_data.shape[1])),fft(_data, axis=0)[2:])), axis=0))
        
    def imu2body(self, df: pd.DataFrame, t, fs, pos=[0, 0, 0], method='complementary'):
        gyr = df[:,0:3]
        acc = df[:,3:]
        grv = np.array([[0],[0],[-9.81]])
        alpha = self.FDD(gyr)
        accc = acc - np.cross(gyr,np.cross(gyr,pos)) - np.cross(gyr,pos)
        q0=ahrs.Quaternion(ahrs.common.orientation.acc2q(accc[0]))
        if method == 'complementary':
                imu = ahrs.filters.Complementary(acc=accc, gyr=gyr, frequency=fs, q0=q0, gain=0.001)
        elif method == 'madgwick':
            imu = ahrs.filters.Madgwick(acc=accc, gyr=gyr, frequency=fs, q0=q0, gain=0.001)
        elif method == 'kalman':
            imu = ahrs.filters.EKF(acc=accc, gyr=gyr, frequency=fs, q0=q0)
        elif method == 'aqua':
            imu = ahrs.filters.AQUA(acc=accc, gyr=gyr, frequency=fs, q0=q0, adaptative=True, threshold=0.95)
        elif method == 'mahony':
            imu = ahrs.filters.Mahony(acc=accc, gyr=gyr, frequency=fs, q0=q0, k_P=1.0, k_I=0.3)
        else:
            print('method not found')
            return False

        theta = ahrs.QuaternionArray(imu.Q).to_angles()
        
        acccc = np.zeros_like(accc)
        for ii in range(len(acc)):
            acccc[ii,:] = accc[ii,:] + ahrs.Quaternion(imu.Q[ii]).rotate(grv).T
        
        
        v = self.FDI(acccc)
        d = self.FDI(v)
        ah = {}
        ah['Dx'] = d[:,0]
        ah['Dy'] = d[:,1]
        ah['Dz'] = d[:,2]
        ah['Vx'] = v[:,0]
        ah['Vy'] = v[:,1]
        ah['Vz'] = v[:,2]
        ah['Ax'] = acccc[:,0]
        ah['Ay'] = acccc[:,1]
        ah['Az'] = acccc[:,2]
        ah['thx'] = theta[:,0]
        ah['thy'] = theta[:,1]
        ah['thz'] = theta[:,2]
        ah['omx'] = gyr[:,0]
        ah['omy'] = gyr[:,1]
        ah['omz'] = gyr[:,2]
        ah['alx'] = alpha[:,0]
        ah['aly'] = alpha[:,1]
        ah['alz'] = alpha[:,2]
        
        
        

        dataFrame = pd.DataFrame(ah, t)
        return dataFrame

    def vizspect(self, tt, ff, Sxx, Title, xlims=None, ylims=None, fscale='linear', fig=None, return_fig=False):
        if fig is None:
            fig = plt.figure() 
        ax = fig.add_subplot(111)
        plt.yscale(fscale)
        spec = ax.imshow(Sxx, aspect='auto', cmap='turbo', extent=[tt[0], tt[-1], ff[0], ff[-1]])
        plt.colorbar(spec)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(Title)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        return fig

    def WSST(self, df: pd.DataFrame, fs): 
        coefs_wsst, _, f_wsst, t_wsst, _ = gp.wsst(df.to_numpy(), fs=fs,   
                                                   voices_per_octave=32,
                                                   freq_limits=[1, 800],
                                                   boundary='zeros',
                                                   method='ola')
        
        psd_wsst = coefs_wsst.real**2 + coefs_wsst.imag**2
        psd_wsst /= np.max(psd_wsst)
        psd_wsst[psd_wsst==0] = 1e-8
        return t_wsst, f_wsst[::-1], 20*np.log10(psd_wsst)
        

        
        