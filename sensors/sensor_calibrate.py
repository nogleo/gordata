#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:56:03 2022

@author: nog
"""

import pandas as pd
import numpy as np
import ahrs
import matplotlib.pyplot as plt


df = pd.read_csv('./rawdata_m28_0x6a.csv')
df.columns = [['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az']]
fs = 1666
dt = 1/fs
df.index = df.index.to_numpy()*dt
Ts = 5
Td = 5

acc = df[['Ax', 'Ay', 'Az']].to_numpy()
gyr = df[['Gx', 'Gy', 'Gz']].to_numpy()


def calibrate_imu(acc=None, gyr=None, Ts=None, Td=None, fs=None):
    """


    Args:
        acc (TYPE, optional): DESCRIPTION. Defaults to None.
        gyr (TYPE, optional): DESCRIPTION. Defaults to None.
        Ts (TYPE, optional): DESCRIPTION. Defaults to None.
        Td (TYPE, optional): DESCRIPTION. Defaults to None.
        fs (TYPE, optional): DESCRIPTION. Defaults to None.

    Returns:
        TYPE: DESCRIPTION.

    """

    Ns = Ts*fs
    Nd = Td*fs

    if acc is not None:
        acc_mean = np.array([np.mean(acc[Ns*n:Ns*(n+1)-1], axis=0)
                            for n in range(6)]).T
        acc_bias = np.array(
            [(acc_mean[n, :].max()+acc_mean[n, :].min())/2 for n in range(3)], ndmin=2).T
        acc_ub = acc_mean-acc_bias
        acc_grv = (acc_ub > 1000)*9.81 + (acc_ub < -1000)*-9.81
        acc_KS = acc_grv@np.linalg.pinv(acc_ub)

    if gyr is not None:
        gyr_mean = np.array([np.mean(gyr[Ns*n:Ns*(n+1)-1], axis=0)
                            for n in range(6)]).T
        gyr_bias = np.mean(gyr[:6*Ns, :], axis=0).reshape((3, 1))
        gyr_rot = np.zeros_like(gyr_mean)
        #gyr_ub = gyr_mean-gyr_bias
        # gyr_KS = (np.ones_like(gyr_ub)*gyr_std)@np.linalg.pinv(gyr_ub)

        gyr_d = gyr[6*Ns:, :]-gyr_bias.T
        # gyr_rot = np.array([np.sum(gyr_d[Nd*n:Nd*(n+1)-1], axis=0) for n in range(3)]).T*dt
        gyr_rot = np.array([np.mean(gyr_d[Nd*n:Nd*(n+1)-1], axis=0)
                           for n in range(3)]).T
        # gyr_ref = (gyr_rot>1000)*np.pi + (gyr_rot<-1000)*-np.pi
        gyr_ref = ((gyr_rot > 100)*np.pi + (gyr_rot < -100)*-np.pi)/(Nd*dt)
        gyr_KS = gyr_ref@np.linalg.inv(gyr_rot)

    # acc_std = np.array([np.std(acc_cal[Ns*n:Ns*(n+1)-1], axis=0) for n in range(6)]).mean()
    # gyr_std = np.std(gyr_cal[:6*Ns,:])

    return (acc_KS, acc_bias), (gyr_KS, gyr_bias)


def translate_imu(acc = None, gyr=None, fs=None, acc_param=None, gyr_param=None):
    """


    Args:
        acc (TYPE, optional): DESCRIPTION. Defaults to None.
        gyr (TYPE, optional): DESCRIPTION. Defaults to None.
        fs (TYPE, optional): DESCRIPTION. Defaults to None.
        acc_param (TYPE, optional): DESCRIPTION. Defaults to None.
        gyr_param (TYPE, optional): DESCRIPTION. Defaults to None.

    Returns:
        TYPE: DESCRIPTION.
        TYPE: DESCRIPTION.

    """

    acc_t = acc_param[0]@(acc.T-acc_param[1])
    gyr_t = gyr_param[0]@(gyr.T-gyr_param[1])

    return acc_t.T, gyr_t.T


acc_param, gyr_param = calibrate_imu(acc=acc, gyr=gyr, Ts=Ts, Td=Td, fs=fs)

# %%
df = pd.read_csv('./data_23.csv')
acc_cal, gyr_cal = translate_imu(acc=df[['B_Ax', 'B_Ay', 'B_Az']].to_numpy(), gyr=df[[
                                 'B_Gx', 'B_Gy', 'B_Gz']].to_numpy(), acc_param=acc_param, gyr_param=gyr_param)


# gyr_cal = (gyr_KS@(gyr.T-gyr_bias)).T
# acc_cal = (acc_KS@(acc.T-acc_bias)).T
q0 = ahrs.common.orientation.acc2q(acc_cal[0, :])


ekf = ahrs.filters.ekf.EKF(acc=acc_cal,
                           gyr=gyr_cal,
                           q0=q0,
                           frequency=1666)#,
                           #noise=[acc_std**2, gyr_std**2, 0.8**2])
theta_ekf = ahrs.QuaternionArray(ekf.Q).to_angles()+np.pi




cf = ahrs.filters.complementary.Complementary(acc=acc_cal,
                                              gyr=gyr_cal,
                                              q0=q0,
                                              frequency=1666,
                                              gain=0.01)
theta_cf = ahrs.QuaternionArray(cf.Q).to_angles()



mck = ahrs.filters.madgwick.Madgwick(acc=acc_cal, 
                                     gyr=gyr_cal,
                                     q0=q0,
                                     frequency=1666,
                                     gain=0.01)
theta_mck = ahrs.QuaternionArray(mck.Q).to_angles()



mhn = ahrs.filters.mahony.Mahony(acc=acc_cal,
                                 gyr=gyr_cal,
                                 q0=q0, 
                                 frequency=1666,
                                 k_P=1.0,
                                 k_I=0.3)
theta_mhn = ahrs.QuaternionArray(mhn.Q).to_angles()



aqua = ahrs.filters.aqua.AQUA(acc=acc_cal,
                              gyr=gyr_cal,
                              q0=q0, 
                              frequency=1666, 
                              adaptative=True, 
                              threshold=0.95)
theta_aqua = ahrs.QuaternionArray(aqua.Q).to_angles()

angular_rate = ahrs.filters.AngularRate(gyr=gyr_cal,
                                        q0=q0)
theta_ar = ahrs.QuaternionArray(angular_rate.Q).to_angles()
