import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy import signal
import time
import os
import ahrs
from numpy.linalg import norm
from ahrs.common.orientation import acc2q
import scipy.integrate as intg
import scipy.stats as stats
from numpy import linspace, zeros, array, pi, sin, cos, exp, arange
from scipy.fftpack import fft, ifft, dct, idct, dst, idst, fftshift, fftfreq
from PyEMD import EEMD, CEEMDAN, EMD
import PyEMD
from scipy.interpolate import interp1d
import pywt

import emd




# %%


fs = 1666
dt = 1/fs


def spect(df, dbmin=80):
    for frame in df:
        plt.figure()           
        f, t, Sxx = scipy.signal.spectrogram(df[frame], fs=fs, scaling='spectrum',nfft=fs, nperseg=fs//2, noverlap=fs//4, detrend='linear', mode='complex', window='hann')
        Sxx[Sxx==0] = 10**(-20)
        plt.pcolormesh(t, f, 20*np.log10(np.abs(Sxx)), shading='gouraud', cmap=plt.cm.Spectral_r,vmax=20*np.log10(np.abs(Sxx)).max(), vmin=20*np.log10(np.abs(Sxx)).max()-dbmin)
        plt.ylim((0, fs//4))
        plt.colorbar()
        plt.title('Spectrogram of {}'.format(frame))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        plt.tight_layout()
    return f, t, Sxx

def PSD(df):
    plt.figure()
    for frame in df:
        f, Pxx = signal.welch(df[frame], fs, nperseg=fs//2, noverlap=fs//4, window='hann', average='mean', scaling='spectrum', detrend='linear')
        plt.semilogx(f, np.abs(Pxy))
        plt.title('Power Spectrum of {}'.format(frame))
        plt.ylabel('Power Spectrum [Unit**2]')
        plt.xlabel('Frequency [Hz]')
        plt.grid()
    return f, Pxx
        

def CSD(df, _df):
    plt.figure()    
    for _output in df:
        for _input in _df:            
            f, Pxy = signal.csd(df[_output], _df[_input], fs, nfft=fs nperseg=fs//2, noverlap=fs//4 window='hann', average='mean', scaling='spectrum', detrend='linear')  
            plt.semilogx(f, np.abs(Pxy))
            plt.title('Cross Spectrum between {} and {}'.format(_input, _output))
            plt.ylabel('Power Spectrum [Unit**2]')
            plt.xlabel('Frequency [Hz]')
    return f, Pxy
    # plt.show()
def FFT(df):
    plt.figure()
    for frame in df:
        F = np.fft.fft(df[frame],n=fs)[:fs//2]
        f = np.fft.fftfreq(fs, dt)[:fs//2]
        plt.semilogx(f, F.real)
        plt.semilogx(f, F.imag)
        plt.semilogx(f, np.abs(F))
        plt.semilogx(f, np.sqrt(np.abs(F)))
        plt.semilogx(f, 10*np.log10(np.sqrt(np.abs(F))))
        
    return f, F
              
def FRF(_in, _out):
    
from pandas.core.base import PandasObject


df.spect()
df.PSD()





def imu2body(dF, pos=[0, 0, 0]):
    acc = dF.to_numpy()[:,3:]
    gyr = dF.to_numpy()[:,:3]
    acc_c = rate2acc(gyr, pos)
    grv = np.array([[0],[0],[-9.81]])
    q0=ahrs.Quaternion(acc2q(acc[0]))
    imu = ahrs.filters.Complementary(acc=acc, gyr=gyr, frequency=fs, q0=q0, gain=0.001)
    theta = ahrs.QuaternionArray(imu.Q)
    th = ahrs.QuaternionArray(imu.Q).to_angles()
    acc_cc = np.zeros_like(acc)
    for ii in range(len(acc)):
        acc_cc[ii,:] = acc[ii,:] + ahrs.Quaternion(imu.Q[ii]).rotate(grv).T 
    a = acc_cc
    v = FDI(a)
    d = FDI(v)
    om = gyr
    
    
    
            
def cm_acc(acc,gyr,alp pos):
        al = TDD(gyr)
        aux = np.zeros_like(gyr)
        for ii in range(len(gyr)):
            aux[ii,:] = np.cross(al[ii,:].T,pos) + np.cross(gyr[ii,:],np.cross(gyr[ii,:],pos))
        
        return np.array(aux)
        
        
def FEI(_data, NFFT = 1024):    
    N = len(_data)
    width = _data.shape[1]
    n = NFFT
    _data = (np.vstack((np.zeros((n,width)), zmean(_data),np.zeros((n-N%n,width)))))
    # w = signal.windows.chebwin(n, 100).reshape((n,1))
    # y = np.zeros_like(_data, dtype=complex)
    # for ii in range(0, len(_data)-n, n//2):
    #     Y = _data[ii:ii+n,:]*w
    #     k =  (1j*2*pi*fftfreq(len(Y), dt).reshape((n,1)))
    #     y[ii:ii+n,:] = (ifft(np.vstack((np.zeros((1,width)),fft(Y, axis=0)[1:]/(k[1:]))), axis=0))
    # _datai = zmean(np.real(y))
    _datai = zmean(integ(_data))
    # return _datai[n:N+n]
    _dataI = np.zeros_like(_datai)
    for ii in range(0, len(_data)-n, n):
        for jj in range(width):
            F0 = featExt(_datai[ii:ii+n,jj])
            eemd = EEMD(noise_width=0.15)
            eIMFs = eemd.eemd(_datai[ii:ii+n,jj]).T
            F = []
            for kk in range(eIMFs.shape[1]):
                F.append(featExt(eIMFs[:,kk]))
            nn = norm(np.array(F)**2-F0, axis=1).argmin()
            _dataI[ii:ii+n,jj] = eIMFs[:,nn]
    return _dataI[n:N+n]  
        
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
        
def featExt(_data):
    feat=[]
    for ii in range(_data.shape[1]):
        feat.append([scipy.stats.kurtosis(_data[:,ii]), ((_data[:,ii]-_data[:,ii].mean())**2).mean() ,scipy.linalg.svdvals(_data[:,ii].reshape((len(_data[:,ii]),1))).max()])  #,np.sum(_data[:,ii]**2)
    return np.array(feat)
        
        
    
    
            
def zmean(_data):
    return np.real(ifft(np.vstack((np.zeros((2,_data.shape[1])),fft(_data, axis=0)[2:])), axis=0))
    
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

def TDD(_data):
    _dout = np.zeros_like(_data)
    _dout[1:,:] = np.diff(_data, axis=0)/dt
    return _dout
    
    
    

    
def TDI(_data):
    width = data.shape[1]
    _data = np.vstack((np.zeros((2,width)),data))
    _dout = np.zeros_like(_data)
    for ii in range(1,len(data)):
        _dout[ii,:] = intg.simps(_data[ii:ii+3,:], dx=dt, axis=0)+_dout[ii-1,:]
    return _dout
    
    
def deriv(_data):
    
    N = len(_data)
    k = 2*pi*fftfreq(len(_data), dt).reshape((N,1))
    # k[N//4:N//4+1] = 0
    return np.real(ifft(1j*k*fft(_data, axis=0), axis=0))

def FDI(data, factor=1, NFFT=fs//4):
    n = NFFT
    try:
        width = data.shape[1]
    except:
        width = 0
    _data = np.vstack((np.zeros((2*n,width)), data, np.zeros((2*n,width))))
    N = len(_data)
    w = signal.windows.hann(n).reshape((n,1))
    Data = np.zeros_like(_data, dtype=complex)
    for ii in range(0, N-n, n//2):
        Y = _data[ii:ii+n,:]*w
        k =  (1j*2*pi*fftfreq(len(Y), dt).reshape((n,1)))
        y = (ifft(np.vstack((np.zeros((factor,width)),fft(Y, axis=0)[factor:]/(k[factor:]))), axis=0))
        Data[ii:ii+n,:] += y
    return np.real(Data[2*n:-2*n,:])


    
    
def spect(_data, dbmin=80):
    plt.figure()
    for ii in range(_data.shape[1]):
        plt.subplot(_data.shape[1]*100+10+ii+1)
        f, t, Sxx = scipy.signal.spectrogram(_data[:,ii], fs=fs, axis=0, scaling='spectrum', nperseg=fs//4, noverlap=fs//8, detrend='linear', mode='psd', window='hann')
        Sxx[Sxx==0] = 10**(-20)
        plt.pcolormesh(t, f, 20*np.log10(abs(Sxx)), shading='gouraud', cmap=plt.inferno(),vmax=20*np.log10(abs(Sxx)).max(), vmin=20*np.log10(abs(Sxx)).max()-dbmin)
        plt.ylim((0, fs//8))
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.tight_layout()
        plt.show()
# spect(cur,False)
# spect(b0.a)
# spect(b1.om)
# def PSD(_data):
#     plt.figure()
#     plt.subplot(211)
#     _t = np.linspace(0, len(_data)*dt, len(_data))
#     plt.plot(_t, _data)
#     for ii in range(_data.shape[1]):
#         plt.subplot(212)
#         plt.psd(_data[:,ii], Fs=fs, NFFT=fs//2, noverlap=fs//4, scale_by_freq=False, detrend='linear')
#         plt.xlim((0,830))
#         # plt.ylim((-50))
#     plt.tight_layout()
#     plt.show()
# # PSD(b1.d)


def CSD(X,Y):
    plt.figure()
    plt.subplot(311)
    _t = np.linspace(0, len(X)*dt, len(X))
    plt.plot(_t, X)
    plt.subplot(312)
    plt.plot(_t, Y)
    for ii in range(X.shape[1]):
        for jj in range(Y.shape[1]):
            plt.subplot(313)
            plt.csd(X[:,ii], Y[:,jj], Fs=fs, NFFT=fs//2, noverlap=fs//4, scale_by_freq=False, detrend='linear')
        # plt.xlim((0,830))
        # plt.ylim((-50,40))
    plt.tight_layout()
    plt.show()
    
import ghostipy as gsp
def CWT(df,fs):
    t=df.index.to_numpy()
    kwargs_dict = {}
    kwargs_dict['cmap'] = plt.cm.Spectral_r
    kwargs_dict['vmin'] = 0
    kwargs_dict['vmax'] = 1
    kwargs_dict['linewidth'] = 0
    kwargs_dict['rasterized'] = True
    kwargs_dict['shading'] = 'auto'
    for frame in df:        
        plt.figure()
        coefs_cwt, _, f_cwt, t_cwt, _ = gsp.cwt(df[frame],fs=fs,timestamps=t, boundary='zeros', freq_limits=[1, 415], voices_per_octave=16, wavelet=gsp.wavelets.AmorWavelet())
        psd_cwt = coefs_cwt.real**2 + coefs_cwt.imag**2
        psd_cwt /= np.max(psd_cwt)
        plt.pcolormesh(t_cwt, f_cwt, psd_cwt, **kwargs_dict)
        plt.colorbar()

import emd   
def HHT(df, fs):
    t = df.index.to_numpy()
    mfreqs = np.array([360,300,240,180,120,90,60,30,15,7.5])
    freq_edges, freq_bins = emd.spectra.define_hist_bins(1, 415, 414, 'log')
    for frame in df:        
        imf, _ = emd.sift.mask_sift(df[frame.to_numpy()], mask_freqs=mfreqs/fs,  mask_amp_mode='ratio_sig', ret_mask_freq=True, nphases=8, mask_amp=5)
        Ip, If, Ia = emd.spectra.frequency_transform(imf, fs, 'nht')
        emd.plotting.plot_imfs(imf,t, scale_y=True, cmap=True)
        emd.plotting.plot_imfs(Ia,t, scale_y=True, cmap=True)
        emd.plotting.plot_imfs(Ip,t, scale_y=True, cmap=True)
        emd.plotting.plot_imfs(If,t, scale_y=True, cmap=True)


def PSD(_data):
    f, Pxx = signal.welch(_data, fs, nperseg=fs//2, noverlap=fs//4, window='hann', average='median', scaling='spectrum', detrend='linear', axis=0)
    plt.figure()
    plt.subplot(211)
    _t = np.linspace(0, len(_data)*dt, len(_data))
    plt.plot(_t, _data)
    plt.subplot(212)
    plt.plot(f, 20*np.log10(Pxx))
    plt.grid()


def fix_outlier(_data):
    _m = _data.mean()
    peaks,_ =signal.find_peaks(abs(_data.flatten()),width=1, prominence=2, height=3*_m, distance=5)
    for peak in peaks:
        _f = interp1d(np.array([0,9]), np.array([_data[peak-5],_data[peak+5]]).T, kind='linear')
        _data[peak-5:peak+5] = _f(np.arange(0,10)).T
    return _data
    

    

# %%

num=7
folder = 'data_{}'.format(num)
data = {}
files = os.listdir(folder)
files.sort()
for _name in files:
    data[_name] = np.load("{}/{}".format(folder, _name))


N = len(data['cur.npy'])
t = np.arange(N).reshape((N,1))*dt
# %%


acc0 = data['accB01.npz.npy']
gyr0 = data['gyrB01.npz.npy']
acc1 = data['acc106.npy']
gyr1 = data['gyr106.npy']
rot = fix_outlier(data['rot.npy'])
cur = data['cur.npy']
rot_rad = [x*pi/180 for x in rot[:,0]]
rot_rad = np.deg2rad(rot)
rr = np.unwrap(rot_rad, axis=0)
cm = np.array([8.0563e-005,	5.983e-004,	-6.8188e-003])
L = np.array([5.3302e-018, -7.233e-002, 3.12e-002+2.0e-003])

pos = L-cm


b0 = imu2body(acc0, gyr0, pos)
b1 = imu2body(acc1, gyr1)

peaks,_ =signal.find_peaks(abs(cur.flatten()),width=2, prominence=5, height=2000, distance=10)
T = [peaks[0], peaks[peaks>4000][0], peaks[-1]]
# %%






spect(inst_freq)

CSD(cur, b0.a)

PSD(cur)



rundown = peaks[-1]
plt.figure()
plt.plot(cur)
plt.plot(T,df[frame][T*dt],'o')

plt.figure()
plt.plot(rot)
plt.plot(T,rot[T],'o')

plt.figure()
plt.plot(t[T[0]:T[1]], b0.a[T[0]:T[1],:])

plt.figure()
plt.plot(t[T[1]:T[2]], b0.a[T[1]:T[2],:])

plt.plot(peaks[peaks>4000],cur[peaks[peaks>4000]],'o')

runup = peaks[0:]

plt.plot(rot)
plt.plot(peaks,rot[peaks],'o')
plt.plot(stactic, rot[stactic], 'o')
plt.plot(rundown,rot[rundown],'o')

plt.figure()
plt.plot(rot)
plt.plot(peaks,rot[peaks],'o')

plt.figure()
plt.plot(acc0[peak[0]-fs:])
plt.plot([fs]*3,acc0[peak[0],:],'o')
spect(acc0[peak[0]-fs:])
spect(gyr0[peak[0]-fs:])

# Data = np.hstack((cur,rot))

Data = b0.a[T[2]:,:]

imf = emd.sift.mask_sift(acc0[T[2]:,0])

imf, _ = emd.sift.complete_ensemble_sift(acc0[T[2]:,0])
emd.plotting.plot_imfs(imf, sample_rate=fs,cmap=True)
Ip, If, Ia= emd.spectra.frequency_transform(D, fs, 'hilbert')
edges, centres = emd.spectra.define_hist_bins(0, fs//2,fs)
spec = emd.spectra.hilberthuang_1d(If, Ia, edges)/N
plt.figure()
plt.plot(centres, spec[:,0])
# %%
# Assign EEMD to `eemd` variable
D = b0.a[T[1]:T[2],2]
imf = emd.sift.sift(D, imf_opts={'sd_thresh': 0.2})
emd.plotting.plot_imfs(imf, cmap=True, scale_y=True)


# %%

def sensor_anal(_data):
    fig,ax = plt.subplots(1,_data.shape[1])
    for ii in range(_data.shape[1]):        
        mu = _data[:,ii].mean()
        sigma = _data[:,ii].std()
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
        ax2=ax[ii].twinx()
        ax[ii].hist(_data[::,ii], bins=111)
        gauss = stats.norm.pdf(x, mu, sigma)
        ax2.plot(x, gauss, color="red")
        plt.ylim([0,gauss.max()])
        plt.show()
