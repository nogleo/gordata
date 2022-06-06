import scipy
import matplotlib.pyplot as plt
import numpy as np
import ahrs
import pandas as pd
import scipy.signal as signal
import scipy.integrate as intg
from numpy import pi
from scipy.fftpack import fft, ifft, dct, idct, dst, idst, fftshift, fftfreq
from numpy import linspace, zeros, array, pi, sin, cos, exp, arange
import emd 
#import ssqueezepy as sq

fs = 1666
dt = 1/fs
def prep_data(df, fs,  factor=1, rotroll=None, fc=None, senslist=None):
    t = df.index.to_numpy()
    if 'rot' in df.columns:        
        if df.rot.max()>=300.0:
            df['rot'] = np.deg2rad(df['rot'])
        df.rot = np.unwrap(df.rot)
        if fc!= None:
            b,a = scipy.signal.cheby1(23, 0.175, fc, fs=fs)
            S = scipy.signal.filtfilt(b, a, df, axis=0)
        else:
            S = df.to_numpy()
    
    ss, tt = scipy.signal.resample(S,factor*len(S), t=t, axis=0, window='hann')
    df = pd.DataFrame(ss, tt, columns=df.columns)   
    
    # S[:,2:] = freqfilt(S[:,2:], fs, fc)
    # ss[:,0] = ss[:,0]%(2*np.pi)
    # ss = ss[100:-100,:]
    # tt = tt[100:-100]
    FS = factor*fs
    n_start = 0
    Nq = 0
    Q = [None,]*len(senslist)
    if senslist!=None and 'C'in senslist:
        n_start=+2
        Q[Nq] = df[['cur','rot']]
        Nq=+1
        
    if senslist!=None and 'A'in senslist:
        cma = np.array([-4.4019e-004	, 1.2908e-003,	-1.9633e-002])
        La = np.array([-8.3023e-019, 	-8.1e-002,	-8.835e-002])
        posa = La-cma
        Q[Nq] = imu2body(df[df.columns[n_start: n_start+6]].to_numpy(),tt, FS, posa)
        Nq=+1
        
    if senslist!=None and 'B'in senslist:
        cmb = np.array([8.0563e-005,	5.983e-004,	-6.8188e-003])
        Lb = np.array([5.3302e-018, -7.233e-002, 3.12e-002+2.0e-003])
        posb = Lb-cmb
        Q[Nq] = imu2body(df[df.columns[-6:]].to_numpy(), tt, FS, posb)
        
    
    
    
    
    for ii in range(len(Q)):
        try:
            _q, _t = scipy.signal.resample(Q[ii], len(Q[ii])//factor, t=Q[ii].index, axis=0, window='hann')
            Q[ii] = pd.DataFrame(_q,_t, columns=Q[ii].columns)
        except:
            print(Exception)
        if 'rot'in Q[ii].columns:
            Q[ii]['rot'] = Q[ii]['rot']%(2*np.pi)
        
   
    
    return Q
   

def freqfilt(data, fs, fc):
    data[:,0] = np.unwrap(np.deg2rad(data[:,0]))
    N = len(data)
    ff = fftfreq(N,1/fs)
    k = (abs(ff)<=fc).reshape((N,1))
    Data = fft(data, axis=0)
    Data.real = Data.real*k
    # Data.real = Data.real*k
    data_out = np.real(ifft(Data, axis=0))
    data_out[:,0] = data_out[:,0]%(2*np.pi)
    return data_out
    
    

def fix_outlier(_data):
    _m = _data.mean()
    peaks,_ = scipy.signal.find_peaks(abs(_data.flatten()),width=1, prominence=2, height=3*_m, distance=5)
    for peak in peaks:
        _f = scipy. interpolate.interp1d(np.array([0,9]), np.array([_data[peak-5],_data[peak+5]]).T, kind='linear')
        _data[peak-5:peak+5] = _f(np.arange(0,10)).T
    return _data

# def PSD(_data, fs):
#     f, Pxx = scipy.signal.welch(_data, fs, nperseg=fs//4, noverlap=fs//8, window='hann', average='median', scaling='spectrum', detrend='linear', axis=0)
#     plt.figure()
#     plt.subplot(211)
#     _t = np.linspace(0, len(_data)*dt, len(_data))
#     plt.plot(_t, _data)
#     plt.subplot(212)
#     plt.semilogx(f, 20*np.log10(abs(Pxx)))
#     plt.xlim((1,415))
#     plt.grid()

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


def FDI(data, factor=1, NFFT=fs//4):
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
        k =  (1j*2*np.pi*scipy.fft.fftfreq(len(Y), dt).reshape((n,1)))
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


    
    


        
def FDD(_data, factor=1, NFFT=fs):
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
        k =  (1j*2*pi*fftfreq(len(Y), dt).reshape((n,1)))
        y = (ifft(np.vstack((np.zeros((factor,width)),fft(Y, axis=0)[factor:]*(k[factor:]))), axis=0))
        Data[ii:ii+n,:] += y
    return np.real(Data)        

def TDI(_data): 
    N = len(_data)
    if len(_data.shape)==1:
        _data = _data.reshape((N,1))
    _data = zmean(_data)
    _dataout = np.zeros_like(_data)
    _dataout[0,:] = _data[0,:]*dt/2
    for ii in range(1,N):
        _dataout[ii,:] = intg.simpson(_data[0:ii,:], dx=dt, axis=0)
    return _dataout

def zmean(_data):
    return np.real(ifft(np.vstack((np.zeros((2,_data.shape[1])),fft(_data, axis=0)[2:])), axis=0))
       
def imu2body(df, t, fs, pos=[0, 0, 0], method='complementary'):
    gyr = df[:,0:3]
    acc = df[:,3:]
    grv = np.array([[0],[0],[-9.81]])
    alpha = FDD(gyr)
    accc = acc - np.cross(gyr,np.cross(gyr,pos)) - np.cross(gyr,pos)
    q0=ahrs.Quaternion(ahrs.common.orientation.acc2q(accc[0]))
    match method:
        case 'complementary':
            imu = ahrs.filters.Complementary(acc=accc, gyr=gyr, frequency=fs, q0=q0, gain=0.001)
        case 'madgwick':
            imu = ahrs.filters.Madgwick(acc=accc, gyr=gyr, frequency=fs, q0=q0, gain=0.001)
        case 'kalman':
            imu = ahrs.filters.EKF(acc=accc, gyr=gyr, frequency=fs, q0=q0)
        case 'aqua':
            imu = ahrs.filters.AQUA(acc=accc, gyr=gyr, frequency=fs, q0=q0, adaptative=True, threshold=0.95)
        case 'mahony':
            imu = ahrs.filters.Mahony(acc=accc, gyr=gyr, frequency=fs, q0=q0, k_P=1.0, k_I=0.3)


            

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

