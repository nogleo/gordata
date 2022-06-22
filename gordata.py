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
from scipy.fftpack import fft, ifft, fftfreq
class daq:
    def __init__(self):
        self.__name__ = "daq"
        logging.basicConfig(level=logging.DEBUG)
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
        self.settings: dict(dict) = {}
        self.fs: float = 1666  # sampling frequency
        self.dt: float = 1/self.fs  # sampling period
        self.running: bool = False
        self.raw: bool = False
        self.data_rate: int = 8  # 8=1666Hz 9=3330Hz 10=6660Hz
        self.data_range: list[int] = [1, 3]  # [16G, 2000DPS]
        self.q = queue.Queue()
        self.init_devices()

    def init_devices(self):
        for address in range(128):
            try:
                self.bus.read_byte(address)
                logging.info("Found device at address: 0x%02X", address)
                if address == 0x6a or address == 0x6b:
                    num = str(107-address)
                    self.devices[address] = [0x22, 12, '<hhhhhh', 
                                             ['Gx_'+num, 'Gy_'+num, 'Gz_'+num, 'Ax_'+num, 'Ay_'+num, 'Az_'+num],
                                              None]
                    self.settings[address] = {0x10:(self.data_rate << 4 | self.data_range[0] << 2 | 1 << 1),
                                     0x11: (self.data_rate << 4 | self.data_range[1] << 2),
                                     0x12: 0x44,
                                     0x13: 1 << 1,
                                     0x15: 0b011,
                                     0X17: 0x44}  # [0x44 is hardcoded acording to LSM6DSO datasheet](0b000 << 5
                    self.set_device(address)
                        
                elif address == 0x48:
                    self.devices[address] = [0x00, 2, '>h', ['cur'], None]
                    config = (3 << 9 | 0 << 8 | 4 << 5 | 3)
                    self.settings[address] = {0x01: [(config >> 8 & 0xFF), (config & 0xFF)]}
                    self.set_device(address)
                elif address == 0x36:
                    self.devices[address] = [0x0C, 2, '>H', ['rot'], None]
                    self.settings[address] = []

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
                return (acc_KS, acc_bias), (gyr_KS, gyr_bias)   

    def translate_imu(self, acc: np.array=None, gyr=None, fs=None, acc_param=None, gyr_param=None):
        acc_t = acc_param[0]@(acc.T-acc_param[1])
        gyr_t = gyr_param[0]@(gyr.T-gyr_param[1])
        return acc_t.T, gyr_t.T

    def pull_data(self, durr: int=None, devices=None, raw=True):
        if durr == 0:
            durr = float('inf')
        if devices is None:
            devices = self.devices
        if self.q.not_empty:
            self.q.queue.clear()        
        t0 = ti = tf = time.perf_counter()
        ii=0
        self.running = True
        while self.running and ii<durr*self.fs:
            tf = time.perf_counter()
            if tf-ti>=self.dt:
                ti = time.perf_counter()
                for addr, val in devices.items():
                    try:
                        self.q.put(self.bus.read_i2c_block_data(addr, val[0], val[1]))                                       
                    except Exception as e:
                        self.q.put((0,)*val[1])
                        #q.put((np.NaN,)*val[1])
                        logging.warning("Could not pull data. Error: ", exc_info=e)
            ii=+1    
        t1 = time.perf_counter()
        logging.debug("Pulled data in %.6f s" % (t1-t0))
        if raw is False:
            return 
        return self.dequeue_data(raw=raw)

    def dequeue_data(self, raw: bool=True) -> pd.DataFrame():
        data = {}
        if self.q is not None:
            for addr, val in self.devices.items():
                columns = []
                columns.append(val[-2])
                data[addr] = []
                logging.debug("Dequeuing data from {}".format(addr))
            while self.q.qsize() > 0:       #block dequeueing data
                for addr, val in self.devices.items():
                    qq = self.q.get()
                    if any(qq):
                        #logging.debug("Dequeuing data to {} and {}".format(addr, val))
                        data[addr].append(unpack(val[2], bytearray(qq)))
                    else:
                        logging.info('dequeue data error, atribute NaN')
                        data[addr].append((np.NaN,)*(val[1]//2))
            for addr, val in self.devices.items():  #block translate from raw to meaningful data
                if val[-1] is not None and not raw:
                    if addr == 0x6a or addr == 0x6b:
                        params = pd.read_csv('./sensors/'+val[-1]+'.csv')
                        array = np.array(data[addr])
                        data[addr] = self.translate_imu(acc=array[:,3:],
                                                        gyr=array[:,:3],
                                                        fs=self.fs,
                                                        acc_param=(params['acc_p']),
                                                        gyr_param=(params['gyr_p']))
                    elif addr == 0x36 or addr==0x48:
                        scale = pd.read_csv('./sensors/'+val[-1]+'.csv')
                        data[addr] = np.array(data[addr]*scale[0])               
        return pd.DataFrame(data, index=np.arange(len(data[addr]))/self.fs, columns=columns)

    def save_data(self, df: pd.DataFrame):
        path = self.root+'/data/'
        try:
            num: int = os.listdir(path).__len__()
        except Exception as e:
            logging.debug("can`t list path", exc_info=e)
            os.mkdir(path)
            pass
        try:
            filename = path+'data_{:03d}.csv'.format(num)
            df.to_csv(filename)
            logging.info("saved data to {}".format(filename))
            return True
        except Exception as e:
            logging.error('Could not savedata_{:03d}.csv'.format(num), exc_info=e)
            return False


class dsp:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__name__ = "dsp"
        self.fs = 1666
        self.dt = 1/self.fs

    def PSD(self, df, fs, units='unid.', fig=None, line='-', linewidth=1, S_ref=1):
        f, Pxx = signal.welch(df, fs, nperseg=fs//4, noverlap=fs//8, window='hann', average='mean', scaling='density', detrend=False, axis=0)
        if fig==None:
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


    def FDI(self, data, factor=1, NFFT=None):
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
            y = (ifft(np.vstack((np.zeros((factor,width)),scipy.fft.fft(Y, axis=0)[factor:]/(k[factor:]))), axis=0))
            Data[ii:ii+n,:] += y
        return np.real(Data[2*n:-2*n,:])
        
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
                return t, f, 20*np.log10(abs(Sxx))

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