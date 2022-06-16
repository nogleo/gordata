import os
import queue
from struct import unpack
import time
from smbus import SMBus
import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal
import scipy.integrate as intg
import ahrs
from matplotlib import pyplot as plt
import logging
import emd 
from scipy.fftpack import fft, ifft, fftfreq


class Daq:
    def __init__(self) -> None:
        self.__name__ = "daq"
        try:
            self.bus = SMBus(1)
            logging.info("I2C bus successfully initialized")
        except Exception as e:
            logging.warning("I2C connection Error: ", e)
            logging.info("I2C bus unable to initialize")

        # list of variables
        self.root: str = os.getcwd()
        self.sessionname: str = None
        self.devices: dict = {}
        self.fs = 1666  # sampling frequency
        self.dt = 1/self.fs  # sampling period
        self.daq_running = False
        self.daq_raw = False
        self.data_rate = 9  # 8=1666Hz 9=3330Hz 10=6660Hz
        self.data_range = [1, 3]  # [16G, 2000DPS]
        self.queue = queue.Queue()

        for address in range(128):
            try:
                self.bus.read_byte(address)
                self.devices_list.append(address)
                if address == 0x6a or address == 0x6b:
                    num = str(107-address)
                    self.devices_config[address] = [0x22, 12, '<hhhhhh', ['Gx_'+num, 'Gy_'+num, 'Gz_'+num, 'Ax_'+num, 'Ay_'+num, 'Az_'+num], None]
                    _settings = [[0x10, (self.odr << 4 | self.range[0] << 2 | 1 << 1)],
                                 [0x11, (self.odr << 4 | self.range[1] << 2)],
                                 [0x12, 0x44],
                                 [0x13, 1 << 1],
                                 [0x15, 0b011],
                                 [0X17, (0b000 << 5)]]  # [0x44 is hardcoded acording to LSM6DSO datasheet]
                    for _set in _settings:
                        try:
                            self.bus.write_byte_data(address, _set[0], _set[1])

                        except Exception as e:
                            logging.warning("ERROR: ", e)
                elif address == 0x48:
                    self.devices_config[address] = [
                        0x00, 2, '>h', ['cur'], None]
                    _config = (3 << 9 | 0 << 8 | 4 << 5 | 3)
                    _settings = [0x01, [_config >> 8 & 0xFF, _config & 0xFF]]
                    try:
                        self.bus.write_i2c_block_data(
                            address, _settings[0], _settings[1])
                    except Exception as e:
                        logging.warning("ERROR: ", e)
                elif address == 0x36:
                    self.devices_config[address] = [
                        0x0C, 2, '>H', ['rot'], None]

            except Exception as e:
                logging.warning("ERROR ", e)

    def calibrate_imu(acc: np.array=None, gyr: np.array=None, Ts: float=None, Td: float=None, fs: float=None, name: str=None):
        Ns: int = Ts*fs
        Nd: int = Td*fs
        
        if acc is not None:
            acc_mean = np.array(
                [np.mean(acc[Ns*n:Ns*(n+1)-1], axis=0) for n in range(6)]).T
            acc_std = np.array(
                [np.std(acc[Ns*n:Ns*(n+1)-1], axis=0) for n in range(6)])
            acc_bias = np.array(
                [(acc_mean[n, :].max()+acc_mean[n, :].min())/2 for n in range(3)], ndmin=2).T
            acc_ub = acc_mean-acc_bias
            acc_grv = (acc_ub > 1000)*9.81 + (acc_ub < -1000)*-9.81
            acc_KS = acc_grv@np.linalg.pinv(acc_ub)

        if gyr is not None:
            gyr_mean = np.array(
                [np.mean(gyr[Ns*n:Ns*(n+1)-1], axis=0) for n in range(6)]).T
            # gyr_std = np.std(gyr[:6*Ns, :]).reshape((3, 1))
            gyr_bias = np.mean(gyr[:6*Ns, :], axis=0).reshape((3, 1))
            gyr_rot = np.zeros_like(gyr_mean)
            # gyr_ub = gyr_mean-gyr_bias
            # gyr_KS = (np.ones_like(gyr_ub)*gyr_std)@np.linalg.pinv(gyr_ub)

            gyr_d = gyr[6*Ns:, :]-gyr_bias.T
            # gyr_rot = np.array([np.sum(gyr_d[Nd*n:Nd*(n+1)-1], axis=0) for n in range(3)]).T*dt
            gyr_rot = np.array(
                [np.mean(gyr_d[Nd*n:Nd*(n+1)-1], axis=0) for n in range(3)]).T
            # gyr_ref = (gyr_rot>1000)*np.pi + (gyr_rot<-1000)*-np.pi
            gyr_ref = ((gyr_rot > 100)*np.pi + (gyr_rot < -100)*-np.pi)/(Nd/fs)
            gyr_KS = gyr_ref@np.linalg.inv(gyr_rot)
            try:
                pd.DataFrame([acc_KS, acc_bias, gyr_KS, gyr_bias]).to_csv('./sensors/'+name+'.csv')
                return True
            except:
                logging.warning("ERROR: unable to save calibration data")
                return False
    
    def translate_imu(acc = None, gyr=None, fs=None, acc_param=None, gyr_param=None):
        acc_t = acc_param[0]@(acc.T-acc_param[1])
        gyr_t = gyr_param[0]@(gyr.T-gyr_param[1])

        return acc_t.T, gyr_t.T

    def pull_data(self, durr: float=None) -> queue.Queue:
        q = queue.Queue()
        self.daq_running = True
        t0=ti = time.perf_counter()
        while self.daq_running and tf-t0<= durr:
            tf = time.perf_counter()
            if tf-ti>=self.dt:
                ti = time.perf_counter()
                for addr, val in self.devices:
                    try:
                        self.queue.put(self.bus.read_i2c_block_data(addr, val[0], val[1], val[2]))               
                        
                    except Exception as e:
                        self.queue.put((np.NaN,)*val[2])
                        logging.warning("Could not pull data. Error: ", e)
                
        t1 = time.perf_counter()
        logging.debug("Pulled data in %.6f s" % (t1-t0))
        return self.dequeue_data(q)

    def dequeue_data(self, q: queue=None) -> pd.DataFrame:
        data = {}
        if q is None:
            while q.size() > 0:
                columns = []
                for addr, val in self.devices:
                    data[addr].append(unpack(val[3], q.get()))
                    columns.append(val[-2])
        return self.save_data(pd.DataFrame(data, index=np.arange(len(data)/self.fs), columns=columns))

    def save_data(self, df: pd.DataFrame):
        if self.sessionname is not None:
            path = self.root+'/data/'+self.sessionname+'/'
        else:
            path = self.root+'/data/'
        num: int = os.listdir(path).__len__()
        try:
            df.to_csv(path+'data_%2i.csv' % num)
            return True
        except:
            logging.warning("Could not save data")
            return False


        

class Dsp:
    def __init__(self) -> None:
        self.__name__ = "dsp"
        self.fs = 1666
        self.dt = 1/self.fs

        def PSD(df, fs, units='unid.', fig=None, line='-', linewidth=1, S_ref=1):
            f, Pxx = scipy.signal.welch(df, fs, nperseg=fs//4, noverlap=fs//8, window='hann', average='mean', scaling='density', detrend=False, axis=0)
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


        def FDI(data, factor=1, NFFT=self.fs//4):
            n = NFFT
            try:
                width = data.shape[1]
            except:
                width = 0
            _data = np.vstack((np.zeros((2*n,width)), data, np.zeros((2*n,width))))
            N = len(_data)
            w = scipy.signal.windows.hann(n).reshape((n,1))
            Data = np.zeros_like(_data, dtype=complex)
            for ii in range(0, N-n, n//2):
                Y = _data[ii:ii+n,:]*w
                k =  (1j*2*np.pi*scipy.fft.fftfreq(len(Y), self.dt).reshape((n,1)))
                y = (scipy.fft.ifft(np.vstack((np.zeros((factor,width)),scipy.fft.fft(Y, axis=0)[factor:]/(k[factor:]))), axis=0))
                Data[ii:ii+n,:] += y
            return np.real(Data[2*n:-2*n,:])
            
        # def spect(df,fs, dbmin=80):
                
        #     plt.figure()
        #     if len(_data.shape)<2:
        #         _data = _data.reshape((len(_data),1))
        #     kk = _data.shape[1]           
        #     for ii in range(kk):
        #         plt.subplot(kk*100+10+ii+1)
        #         f, t, Sxx = scipy.signal.spectrogram(_data[:,ii], fs=fs, axis=0, scaling='spectrum', nperseg=fs//4, noverlap=fs//8, detrend='linear', mode='psd', window='hann')
        #         Sxx[Sxx==0] = 10**(-20)
        #         plt.pcolormesh(t, f, 20*np.log10(abs(Sxx)), shading='auto', cmap=plt.inferno(),vmax=20*np.log10(abs(Sxx)).max(), vmin=20*np.log10(abs(Sxx)).max()-dbmin)
        #         plt.ylim((0, 300))
        #         plt.colorbar()
        #         plt.ylabel('Frequency [Hz]')
        #         plt.xlabel('Time [sec]')
        #         plt.tight_layout()
        #         plt.show()

        def spect(df,fs, dbmin=80, print=True, freqlims=(1,480)):
            for frame in df:
                f, t, Sxx = scipy.signal.spectrogram(df[frame], fs=fs, axis=0, scaling='spectrum', nperseg=fs//2, noverlap=fs//4, detrend=False, mode='psd', window='hann')
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


            
            


                
        def FDD(_data, factor=1, NFFT=self.fs):
            N = len(_data)
            try:
                width = _data.shape[1]
            except:
                _data = _data.reshape((N,1))
                width = 1
            n = NFFT
            w = signal.windows.hann(n).reshape((n,1))
            Data = np.zeros_like(_data, dtype=complex)
            for ii in range(0, N-n, n//2):
                Y = _data[ii:ii+n,:]*w
                k =  (1j*2*np.pi*fftfreq(len(Y), self.dt).reshape((n,1)))
                y = (ifft(np.vstack((np.zeros((factor,width)),fft(Y, axis=0)[factor:]*(k[factor:]))), axis=0))
                Data[ii:ii+n,:] += y
            return np.real(Data)        

        def TDI(data): 
            N = len(data)
            if len(data.shape)==1:
                data = data.reshape((N,1))
            data = zmean(data)
            dataout = np.zeros_like(data)
            dataout[0,:] = data[0,:]*self.dt/2
            for ii in range(1,N):
                #dataout[ii,:] = intg.simpson(data[0:ii,:], dx=self.dt, axis=0)
                dataout[ii,:] = intg.trapz(data[0:ii,:], dx=self.dt, axis=0)
            return dataout

        def zmean(_data):
            return np.real(ifft(np.vstack((np.zeros((2,_data.shape[1])),fft(_data, axis=0)[2:])), axis=0))
            
        def imu2body(df: pd.DataFrame, t, fs, pos=[0, 0, 0], method='complementary'):
            gyr = df[:,0:3]
            acc = df[:,3:]
            grv = np.array([[0],[0],[-9.81]])
            alpha = FDD(gyr)
            accc = acc - np.cross(gyr,np.cross(gyr,pos)) - np.cross(gyr,pos)
            q0=ahrs.Quaternion(ahrs.common.orientation.acc2q(accc[0]))
            if method is 'complementary':
                    imu = ahrs.filters.Complementary(acc=accc, gyr=gyr, frequency=fs, q0=q0, gain=0.001)
            elif method is 'madgwick':
                imu = ahrs.filters.Madgwick(acc=accc, gyr=gyr, frequency=fs, q0=q0, gain=0.001)
            elif method is 'kalman':
                imu = ahrs.filters.EKF(acc=accc, gyr=gyr, frequency=fs, q0=q0)
            elif method is 'aqua':
                imu = ahrs.filters.AQUA(acc=accc, gyr=gyr, frequency=fs, q0=q0, adaptative=True, threshold=0.95)
            elif method is 'mahony':
                imu = ahrs.filters.Mahony(acc=accc, gyr=gyr, frequency=fs, q0=q0, k_P=1.0, k_I=0.3)
            else:
                print('method not found')
                return False

            theta = ahrs.QuaternionArray(imu.Q).to_angles()
            
            acccc = np.zeros_like(accc)
            for ii in range(len(acc)):
                acccc[ii,:] = accc[ii,:] + ahrs.Quaternion(imu.Q[ii]).rotate(grv).T
            
            
            v = FDI(acccc)
            d = FDI(v)
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



        def vizspect(tt, ff, Sxx, Title, xlims=None, ylims=None, fscale='linear'):
            
            
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
            
            fig.show()
            
            
            
        def apply_emd(df, fs):
            t = df.index.to_numpy()
            mfreqs = np.array([360,300,240,180,120,90,60,30,15,7.5])
            for frame in df.columns:
                S = df[frame].to_numpy()
                
            imf, _ = emd.sift.mask_sift(S, mask_freqs=mfreqs/fs,  mask_amp_mode='ratio_sig', ret_mask_freq=True, nphases=8, mask_amp=S.max())
            Ip, If, Ia = emd.spectra.frequency_transform(imf, fs, 'nht')
            emd.plotting.plot_imfs(imf,t, scale_y=True, cmap=True)
            plt.suptitle('IMFs - {}'.format(frame))
            emd.plotting.plot_imfs(Ia,t, scale_y=True, cmap=True)
            plt.suptitle(' Envelope - {}'.format(frame))
            # emd.plotting.plot_imfs(Ip,t, scale_y=True, cmap=True)
            # emd.plotting.plot_imfs(If,t, scale_y=True, cmap=True)    
            
        #def WSST(df, fs, ridge_ext = False):
        #    t = df.index.to_numpy()
        #    for frame in df.columns:
        #        S = df[frame].to_numpy()
        #        Tw, _, nf, na, *_ = sq.ssq_cwt(S, fs=fs, nv=64, ssq_freqs='linear', maprange='energy')
        #        vizspect(t, nf, np.abs(Tw), 'WSST - '+frame, ylims=[1, 480])
        #        if ridge_ext:
        #           ridge = sq.ridge_extraction.extract_ridges(Tw, bw=4, scales=nf, n_ridges=3)
