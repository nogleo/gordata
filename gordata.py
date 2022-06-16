from email import header
import os
import gc
import queue
from struct import unpack
import time
import numpy as np
from smbus import SMBus
import sigprocess as sp
import scipy
import pandas as pd
import logging


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


        

