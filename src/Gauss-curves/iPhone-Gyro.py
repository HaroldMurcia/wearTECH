# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 00:40:47 2019

@author: juanita
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import kurtosis
from numpy.fft import fft, fftfreq
from scipy.stats import norm
import pickle
import os 

main_path = os.path.abspath(__file__+"/../../..")+"/"

def loadData(user,source):
    #0=celular en la posicion #1
    #1=celular en la posicion #2
    #2=celular en la posicion #3
    #3= posicion aleatoria, por lo general solo se usara este
    if source == 0:
        ## CAMINANDO
        # iPhone Header
        inames=['loggingTime',	'loggingSample',	'identifierForVendor',	'deviceID',	'locationTimestamp_since1970',	'locationLatitude',	'locationLongitude',	'locationAltitude',	'lSpeed',	'locationCourse',	'locationVerticalAccuracy',	'locationHorizontalAccuracy',	'locationFloor',	'locationHeadingTimestamp_since1970',	'locationHeadingX',	'locationHeadingY',	'locationHeadingZ',	'locationTrueHeading',	'locationMagneticHeading',	'locationHeadingAccuracy',	'aTime',	'AX',	'AY',	'AZ',	'gTime',	'gX',	'gY',	'gZ',	'magnetometerTimestamp_sinceReboot',	'magnetometerX',	'magnetometerY',	'magnetometerZ',	'motionTimestamp_sinceReboot',	'motionYaw',	'motionRoll',	'motionPitch',	'motionRotationRateX',	'motionRotationRateY',	'motionRotationRateZ',	'motionUserAccelerationX',	'motionUserAccelerationY',	'motionUserAccelerationZ',	'motionAttitudeReferenceFrame',	'motionQuaternionX',	'motionQuaternionY',	'motionQuaternionZ',	'motionQuaternionW',	'motionGravityX',	'motionGravityY',	'motionGravityZ',	'motionMagneticFieldX',	'motionMagneticFieldY',	'motionMagneticFieldZ',	'motionMagneticFieldCalibrationAccuracy',	'activityTimestamp_sinceReboot',	'activity',	'activityCon',	'activityActivityStartDate',	'altimeterTimestamp_sinceReboot',	'altimeterReset',	'altimeter',	'altimeterPressure',	'IP_en0',	'IP_pdp_ip0',	'deviceOrientation',	'batteryState',	'batteryLevel',	'avAudioRecorderPeakPower',	'avAudioRecorderAveragePower',	'label']
        # iPhone dataFrame
        file = main_path+'data/iPhone/walking/i_'+user+'_w_0.csv'
        iwalk = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        # myo dataFrame
        file = main_path+'data/Myo/walking/m_'+user+'_w_0.txt'
        mwalk= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## CORRIENDO
        file = main_path+'data/iPhone/running/i_'+user+'_r_0.csv'
        irun = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/running/m_'+user+'_r_0.txt'
        mrun= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## SUBIENDO ESCALERAS
        file = main_path+'data/iPhone/upstairs/i_'+user+'_up_0.csv'
        iup = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/upstairs/m_'+user+'_up_0.txt'
        mup= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## BAJANDO ESCALERAS
        file = main_path+'data/iPhone/downstairs/i_'+user+'_down_0.csv'
        idown = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/downstairs/m_'+user+'_down_0.txt'
        mdown= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## CARRO
        file = main_path+'data/iPhone/car/i_'+user+'_car_0.csv'
        icar = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/car/m_'+user+'_car_0.txt'
        mcar= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## ESTATICO
        file = main_path+'data/iPhone/static/i_'+user+'_su_0.csv'
        isu = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/static/m_'+user+'_su_0.txt'
        msu= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        #CYCLE
        file = main_path+'data/iPhone/cycle/i_'+user+'_cycle_0.csv'
        icy = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/cycle/m_'+user+'_cycle_0.txt'
        mcy= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
    elif source == 1:
        ## CAMINANDO
        # iPhone Header
        inames=['loggingTime',	'loggingSample',	'identifierForVendor',	'deviceID',	'locationTimestamp_since1970',	'locationLatitude',	'locationLongitude',	'locationAltitude',	'lSpeed',	'locationCourse',	'locationVerticalAccuracy',	'locationHorizontalAccuracy',	'locationFloor',	'locationHeadingTimestamp_since1970',	'locationHeadingX',	'locationHeadingY',	'locationHeadingZ',	'locationTrueHeading',	'locationMagneticHeading',	'locationHeadingAccuracy',	'aTime',	'AX',	'AY',	'AZ',	'gTime',	'gX',	'gY',	'gZ',	'magnetometerTimestamp_sinceReboot',	'magnetometerX',	'magnetometerY',	'magnetometerZ',	'motionTimestamp_sinceReboot',	'motionYaw',	'motionRoll',	'motionPitch',	'motionRotationRateX',	'motionRotationRateY',	'motionRotationRateZ',	'motionUserAccelerationX',	'motionUserAccelerationY',	'motionUserAccelerationZ',	'motionAttitudeReferenceFrame',	'motionQuaternionX',	'motionQuaternionY',	'motionQuaternionZ',	'motionQuaternionW',	'motionGravityX',	'motionGravityY',	'motionGravityZ',	'motionMagneticFieldX',	'motionMagneticFieldY',	'motionMagneticFieldZ',	'motionMagneticFieldCalibrationAccuracy',	'activityTimestamp_sinceReboot',	'activity',	'activitCon',	'activityActivityStartDate',	'altimeterTimestamp_sinceReboot',	'altimeterReset',	'altimeter',	'altimeterPressure',	'IP_en0',	'IP_pdp_ip0',	'deviceOrientation',	'batteryState',	'batteryLevel',	'avAudioRecorderPeakPower',	'avAudioRecorderAveragePower',	'label']
        # iPhone dataFrame
        file = main_path+'data/iPhone/walking/i_'+user+'_w_1.csv'
        iwalk = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        # myo dataFrame
        file = main_path+'data/Myo/walking/m_'+user+'_w_1.txt'
        mwalk= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## CORRIENDO
        file = main_path+'data/iPhone/running/i_'+user+'_r_1.csv'
        irun = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/running/m_'+user+'_r_1.txt'
        mrun= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## SUBIENDO ESCALERAS
        file = main_path+'data/iPhone/upstairs/i_'+user+'_up_1.csv'
        iup = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/upstairs/m_'+user+'_up_1.txt'
        mup= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## BAJANDO ESCALERAS
        file = main_path+'data/iPhone/downstairs/i_'+user+'_down_1.csv'
        idown = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/downstairs/m_'+user+'_down_1.txt'
        mdown= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## CARRO
        file = main_path+'data/iPhone/car/i_'+user+'_car_1.csv'
        icar = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/car/m_'+user+'_car_1.txt'
        mcar= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## ESTATICO
        file = main_path+'data/iPhone/static/i_'+user+'_su_1.csv'
        isu = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/static/m_'+user+'_su_1.txt'
        msu= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        #CYCLE
        file = main_path+'data/iPhone/cycle/i_'+user+'_cycle_1.csv'
        icy = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/cycle/m_'+user+'_cycle_1.txt'
        mcy= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
    elif source == 2:
        ## CAMINANDO
        # iPhone Header
        inames=['loggingTime',	'loggingSample',	'identifierForVendor',	'deviceID',	'locationTimestamp_since1970',	'locationLatitude',	'locationLongitude',	'locationAltitude',	'lSpeed',	'locationCourse',	'locationVerticalAccuracy',	'locationHorizontalAccuracy',	'locationFloor',	'locationHeadingTimestamp_since1970',	'locationHeadingX',	'locationHeadingY',	'locationHeadingZ',	'locationTrueHeading',	'locationMagneticHeading',	'locationHeadingAccuracy',	'aTime',	'AX',	'AY',	'AZ',	'gTime',	'gX',	'gY',	'gZ',	'magnetometerTimestamp_sinceReboot',	'magnetometerX',	'magnetometerY',	'magnetometerZ',	'motionTimestamp_sinceReboot',	'motionYaw',	'motionRoll',	'motionPitch',	'motionRotationRateX',	'motionRotationRateY',	'motionRotationRateZ',	'motionUserAccelerationX',	'motionUserAccelerationY',	'motionUserAccelerationZ',	'motionAttitudeReferenceFrame',	'motionQuaternionX',	'motionQuaternionY',	'motionQuaternionZ',	'motionQuaternionW',	'motionGravityX',	'motionGravityY',	'motionGravityZ',	'motionMagneticFieldX',	'motionMagneticFieldY',	'motionMagneticFieldZ',	'motionMagneticFieldCalibrationAccuracy',	'activityTimestamp_sinceReboot',	'activity',	'activityCon',	'activityActivityStartDate',	'altimeterTimestamp_sinceReboot',	'altimeterReset',	'altimeter',	'altimeterPressure',	'IP_en0',	'IP_pdp_ip0',	'deviceOrientation',	'batteryState',	'batteryLevel',	'avAudioRecorderPeakPower',	'avAudioRecorderAveragePower',	'label']
        # iPhone dataFrame
        file = main_path+'data/iPhone/walking/i_'+user+'_w_2.csv'
        iwalk = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        # myo dataFrame
        file = main_path+'data/Myo/walking/m_'+user+'_w_2.txt'
        mwalk= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## CORRIENDO
        file = main_path+'data/iPhone/running/i_'+user+'_r_2.csv'
        irun = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/running/m_'+user+'_r_2.txt'
        mrun= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## SUBIENDO ESCALERAS
        file = main_path+'data/iPhone/upstairs/i_'+user+'_up_2.csv'
        iup = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/upstairs/m_'+user+'_up_2.txt'
        mup= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## BAJANDO ESCALERAS
        file = main_path+'data/iPhone/downstairs/i_'+user+'_down_2.csv'
        idown = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/downstairs/m_'+user+'_down_2.txt'
        mdown= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## CARRO
        file = main_path+'data/iPhone/car/i_'+user+'_car_2.csv'
        icar = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/car/m_'+user+'_car_2.txt'
        mcar= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## ESTATICO
        file = main_path+'data/iPhone/static/i_'+user+'_su_2.csv'
        isu = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/static/m_'+user+'_su_2.txt'
        msu= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        #CYCLE
        file = main_path+'data/iPhone/cycle/i_'+user+'_cycle_2.csv'
        icy = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/cycle/m_'+user+'_cycle_2.txt'
        mcy= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
    elif source == 3:
        ## CAMINANDO
        # iPhone Header
        inames=['loggingTime',	'loggingSample',	'identifierForVendor',	'deviceID',	'locationTimestamp_since1970',	'locationLatitude',	'locationLongitude',	'locationAltitude',	'lSpeed',	'locationCourse',	'locationVerticalAccuracy',	'locationHorizontalAccuracy',	'locationFloor',	'locationHeadingTimestamp_since1970',	'locationHeadingX',	'locationHeadingY',	'locationHeadingZ',	'locationTrueHeading',	'locationMagneticHeading',	'locationHeadingAccuracy',	'aTime',	'AX',	'AY',	'AZ',	'gTime',	'gX',	'gY',	'gZ',	'magnetometerTimestamp_sinceReboot',	'magnetometerX',	'magnetometerY',	'magnetometerZ',	'motionTimestamp_sinceReboot',	'motionYaw',	'motionRoll',	'motionPitch',	'motionRotationRateX',	'motionRotationRateY',	'motionRotationRateZ',	'motionUserAccelerationX',	'motionUserAccelerationY',	'motionUserAccelerationZ',	'motionAttitudeReferenceFrame',	'motionQuaternionX',	'motionQuaternionY',	'motionQuaternionZ',	'motionQuaternionW',	'motionGravityX',	'motionGravityY',	'motionGravityZ',	'motionMagneticFieldX',	'motionMagneticFieldY',	'motionMagneticFieldZ',	'motionMagneticFieldCalibrationAccuracy',	'activityTimestamp_sinceReboot',	'activity',	'activityCon',	'activityActivityStartDate',	'altimeterTimestamp_sinceReboot',	'altimeterReset',	'altimeter',	'altimeterPressure',	'IP_en0',	'IP_pdp_ip0',	'deviceOrientation',	'batteryState',	'batteryLevel',	'avAudioRecorderPeakPower',	'avAudioRecorderAveragePower',	'label']
        # iPhone dataFrame
        file = main_path+'data/iPhone/walking/i_'+user+'_w_3.csv'
        iwalk = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        # myo dataFrame
        file = main_path+'data/Myo/walking/m_'+user+'_w_3.txt'
        mwalk= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## CORRIENDO
        file = main_path+'data/iPhone/running/i_'+user+'_r_3.csv'
        irun = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/running/m_'+user+'_r_3.txt'
        mrun= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## SUBIENDO ESCALERAS
        file = main_path+'data/iPhone/upstairs/i_'+user+'_up_3.csv'
        iup = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/upstairs/m_'+user+'_up_3.txt'
        mup= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## BAJANDO ESCALERAS
        file = main_path+'data/iPhone/downstairs/i_'+user+'_down_3.csv'
        idown = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/downstairs/m_'+user+'_down_3.txt'
        mdown= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## CARRO
        file = main_path+'data/iPhone/car/i_'+user+'_car_3.csv'
        icar = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/car/m_'+user+'_car_3.txt'
        mcar= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        ## ESTATICO
        file = main_path+'data/iPhone/static/i_'+user+'_su_3.csv'
        isu = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/static/m_'+user+'_su_3.txt'
        msu= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
        #CYCLE
        file = main_path+'data/iPhone/cycle/i_'+user+'_cycle_3.csv'
        icy = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/cycle/m_'+user+'_cycle_3.txt'
        mcy= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
    return iwalk, mwalk, irun, mrun, iup, mup, idown, mdown, icar, mcar,isu,msu,icy,mcy

def valSignal(df,source):
    # source = 1 for iPhone
    # source = 0 for Myo
    if source == 1:
        signal = np.sqrt(pow(df.AX,2)+pow(df.AY,2)+pow(df.AZ,2))
        tS = (max(df.aTime)-min(df.aTime))/len(signal)
        #Gyro
        signalg =np.sqrt(pow(df.gX,2)+pow(df.gY,2)+pow(df.gZ,2))

    elif source == 0:
        signal = np.sqrt(pow(df.aX,2)+pow(df.aY,2)+pow(df.aZ,2))
        tS = (max(df.sec) - min(df.sec))/(len(signal)*1.0)
        #Gyro
        signalg =np.sqrt(pow(df.oX,2)+pow(df.oY,2)+pow(df.oZ,2))


	# fragment 3 secs
    N = round(3.0/tS)
    N = int(N)
    if N<len(signal):
        signal = signal[0:N]
    else:
        print ("lenght error, signal < 3 secs")
    #Gyro
    if N<len(signalg):
        signalg = signalg[0:N]
    else:
        print ("lenght error, signal < 3 secs")
    return signal,signalg

def sigFeatures(signal):
    # sum
    f1 = np.sum(signal)
    # mean
    f2 = np.mean(signal)
    # std
    f3 = np.std(signal)
    # kurtosis
    f4 = kurtosis(signal)
    #valor maximo
    f5= max(signal)
    #valor minimo
    f6= min(signal)
    #rango
    f7=f5-f6

    fV = np.array([f1,f2,f3,f4,f5,f6,f7])
    return  fV

def tFourier(df,signal,source):
    # source = 1 for iPhone
    # source = 0 for Myo
    if source == 1:
        signal=signal-np.mean(signal)
        fourier= np.fft.fft(signal)
        mf=np.abs(fourier/2)
        L=(len(mf)/2)
        mf=mf[0:L]
        tS = (max(df.aTime)-min(df.aTime))/len(signal)*1.0
        Fs=1.0/tS
        xf=np.linspace(0, Fs/2.0,  L)
        f8=max(mf) #Magnitud de Fourier
        pos= np.where(mf == f8)
        f9=xf[pos] #Frecuencia donde se encuentra el pico mas grande
        f10= np.mean(mf) #Media Magnitud de Fourier
        f11=np.std(mf) #Desviacion estandar Magnitud de Fourier

        Fv=np.array([f8,f9,f10,f11])

    elif source == 0:
        signal=signal-np.mean(signal)
        fourier= np.fft.fft(signal)
        mf=np.abs(fourier/2)
        L=(len(mf)/2)
        mf=mf[0:L]
        tS = (max(df.sec) - min(df.sec))/(len(signal)*1.0)
        Fs=1.0/tS
        xf=np.linspace(0, Fs/2.0,  L)
        f8=max(mf)
        pos= np.where(mf == f8)
        f9=xf[pos] #Frecuencia donde se encuentra el pico mas grande
        f10= np.mean(mf) #Media Magnitud fourier
        f11=np.std(mf) #Desviacion estandar magnitud fourier

        Fv=np.array([f8,f9,f10,f11])
    return Fv

def gFeatures(signalg):
    # sum
    f_1 = np.sum(signalg)
    # mean
    f_2 = np.mean(signalg)
    # std
    f_3 = np.std(signalg)
    # kurtosis
    f_4 = kurtosis(signalg)
    #valor maximo
    f_5= max(signalg)
    #valor minimo
    f_6= min(signalg)
    #rango
    f_7=f_5-f_6

    f_V = np.array([f_1,f_2,f_3,f_4,f_5,f_6,f_7])
    return  f_V

def feature8(df,signalg):
    signalg=signalg-np.mean(signalg)
    gfourier= np.fft.fft(signalg)
    gmf=np.abs(gfourier/2)
    Lg=(len(gmf)/2)
    gmf=gmf[0:Lg]
    t_S = (max(df.gTime)-min(df.gTime))/len(signalg)*1.0
    F_s=1.0/t_S
    gxf=np.linspace(0, F_s/2.0,  Lg)
    f_8=max(gmf) #Magnitud de Fourier
    gpos= np.where(gmf == f_8)
    f_9=gxf[gpos] #Frecuencia donde se encuentra el pico mas grande
    f_10= np.mean(gmf) #Media Magnitud de Fourier
    f_11=np.std(gmf) #Desviacion estandar Magnitud de Fourier

    return f_8
    
def feature9(df,signalg):
    signalg=signalg-np.mean(signalg)
    gfourier= np.fft.fft(signalg)
    gmf=np.abs(gfourier/2)
    Lg=(len(gmf)/2)
    gmf=gmf[0:Lg]
    t_S = (max(df.gTime)-min(df.gTime))/len(signalg)*1.0
    F_s=1.0/t_S
    gxf=np.linspace(0, F_s/2.0,  Lg)
    f_8=max(gmf) #Magnitud de Fourier
    gpos= np.where(gmf == f_8)
    f_9=gxf[gpos] #Frecuencia donde se encuentra el pico mas grande
    f_10= np.mean(gmf) #Media Magnitud de Fourier
    f_11=np.std(gmf) #Desviacion estandar Magnitud de Fourier

    return f_9
    
def feature10(df,signalg):
    signalg=signalg-np.mean(signalg)
    gfourier= np.fft.fft(signalg)
    gmf=np.abs(gfourier/2)
    Lg=(len(gmf)/2)
    gmf=gmf[0:Lg]
    t_S = (max(df.gTime)-min(df.gTime))/len(signalg)*1.0
    F_s=1.0/t_S
    gxf=np.linspace(0, F_s/2.0,  Lg)
    f_8=max(gmf) #Magnitud de Fourier
    gpos= np.where(gmf == f_8)
    f_9=gxf[gpos] #Frecuencia donde se encuentra el pico mas grande
    f_10= np.mean(gmf) #Media Magnitud de Fourier
    f_11=np.std(gmf) #Desviacion estandar Magnitud de Fourier
    
    return f_10
    
def feature11(df,signalg):
    signalg=signalg-np.mean(signalg)
    gfourier= np.fft.fft(signalg)
    gmf=np.abs(gfourier/2)
    Lg=(len(gmf)/2)
    gmf=gmf[0:Lg]
    t_S = (max(df.gTime)-min(df.gTime))/len(signalg)*1.0
    F_s=1.0/t_S
    gxf=np.linspace(0, F_s/2.0,  Lg)
    f_8=max(gmf) #Magnitud de Fourier
    gpos= np.where(gmf == f_8)
    f_9=gxf[gpos] #Frecuencia donde se encuentra el pico mas grande
    f_10= np.mean(gmf) #Media Magnitud de Fourier
    f_11=np.std(gmf) #Desviacion estandar Magnitud de Fourier

    return f_11

def lSpeed(df):
    sp=df.lSpeed
    sa=np.abs(sp)
    f12=np.mean(sa)
    return f12

def feature13(df):
    alt=df.altimeter
    #Derivada
    delta=alt.diff()
    f13=np.mean(delta)
    f14=np.std(delta)
    return f13
    
def feature14(df):
    alt=df.altimeter
    #Derivada
    delta=alt.diff()
    f13=np.mean(delta)
    f14=np.std(delta)
    return f14

def val_Scores(Y,pred_1,pred_2,pred_3,pred_4,pred_5):
    f1_RF   = f1_score(Y, pred_1, average=None)
    f1_SVM  = f1_score(Y, pred_2, average=None)
    f1_NN   = f1_score(Y, pred_3, average=None)
    f1_bdt  = f1_score(Y, pred_4, average=None)
    f1_GAUSS= f1_score(Y, pred_5, average=None)
    F1_scores=np.matrix([f1_RF, f1_SVM, f1_NN, f1_bdt, f1_GAUSS])
    return F1_scores

if __name__ == '__main__':
    
    C_1=[] #WALK
    C_2=[] #RUN
    C_3=[] #UP 
    C_4=[] #DOWN
    C_5=[] #STATIC
    C_6=[] #CYCLE
    C_7=[] #CAR

 #----USER1------------------------------------------------------------------
    user = 'dylan'
    #CARGAR DATOS
    diwalk, dmwalk, dirun, dmrun, diup, dmup, didown, dmdown, dicar, dmcar,disu,dmsu,dicy,dmcy = loadData(user,2)
    is_walk,ig_walk=valSignal(diwalk,1)
    is_run,ig_run=valSignal(dirun,1)
    is_up,ig_up=valSignal(diup,1)
    is_down,ig_down=valSignal(didown,1)
    is_su,ig_su=valSignal(disu,1)
    is_cy,ig_cy=valSignal(dicy,1)
    is_car,ig_car=valSignal(dicar,1)
    
    user = 'eli'
    #CARGAR DATOS
    eiwalk, emwalk, eirun, emrun, eiup, emup, eidown, emdown, eicar, emcar,eisu,emsu,eicy,emcy = loadData(user,1)
    eis_walk,eig_walk=valSignal(eiwalk,1)
    eis_run,eig_run=valSignal(eirun,1)
    eis_up,eig_up=valSignal(eiup,1)
    eis_down,eig_down=valSignal(eidown,1)
    eis_su,eig_su=valSignal(eisu,1)
    eis_cy,eig_cy=valSignal(eicy,1)
    eis_car,eig_car=valSignal(eicar,1)
#--------------------------------------------------------------------------------------------------------
#-----USER3----------------------------------------------------------------------------------------------
    user = 'guayara'
    #CARGAR DAgOS
    tiwalk, tmwalk, tirun, tmrun, tiup, tmup, tidown, tmdown, ticar, tmcar,tisu,tmsu,ticy,tmcy = loadData(user,0)
    tis_walk,tig_walk=valSignal(tiwalk,1)
    tis_run,tig_run=valSignal(tirun,1)
    tis_up,tig_up=valSignal(tiup,1)
    tis_down,tig_down=valSignal(tidown,1)
    tis_su,tig_su=valSignal(tisu,1)
    tis_cy,tig_cy=valSignal(ticy,1)
    tis_car,tig_car=valSignal(ticar,1)
    
#-----USER4-------------------------------------------------------------------------------------------
    user = 'janeth'
    #CARGAR DATOS
    jiwalk, jmwalk, jirun, jmrun, jiup, jmup, jidown, jmdown, jicar, jmcar,jisu,jmsu,jicy,jmcy = loadData(user,2)
    jis_walk,jig_walk=valSignal(jiwalk,1)
    jis_run,jig_run=valSignal(jirun,1)
    jis_up,jig_up=valSignal(jiup,1)
    jis_down,jig_down=valSignal(jidown,1)
    jis_su,jig_su=valSignal(jisu,1)
    jis_cy,jig_cy=valSignal(jicy,1)
    jis_car,jig_car=valSignal(jicar,1)
    
#-----USER5-------------------------------------------------------------------------------------------
    user = 'juan'
    #CARGAR DATOS
    jgiwalk, jgmwalk, jgirun, jgmrun, jgiup, jgmup, jgidown, jgmdown, jgicar, jgmcar,jgisu,jgmsu,jgicy,jgmcy = loadData(user,0)
    jgis_walk,jgig_walk=valSignal(jgiwalk,1)
    jgis_run,jgig_run=valSignal(jgirun,1)
    jgis_up,jgig_up=valSignal(jgiup,1)
    jgis_down,jgig_down=valSignal(jgidown,1)
    jgis_su,jgig_su=valSignal(jgisu,1)
    jgis_cy,jgig_cy=valSignal(jgicy,1)
    jgis_car,jgig_car=valSignal(jgicar,1)
    
#-----USER6-------------------------------------------------------------------------------------------
    user = 'juana'
    #CARGAR DATOS
    jtiwalk, jtmwalk, jtirun, jtmrun, jtiup, jtmup, jtidown, jtmdown, jticar, jtmcar,jtisu,jtmsu,jticy,jtmcy = loadData(user,1)
    jtis_walk,jtig_walk=valSignal(jtiwalk,1)
    jtis_run,jtig_run=valSignal(jtirun,1)
    jtis_up,jtig_up=valSignal(jtiup,1)
    jtis_down,jtig_down=valSignal(jtidown,1)
    jtis_su,jtig_su=valSignal(jtisu,1)
    jtis_cy,jtig_cy=valSignal(jticy,1)
    jtis_car,jtig_car=valSignal(jticar,1)
    
#-----USER7-------------------------------------------------------------------------------------------
    user = 'nicolas'
    #CARGAR DATOS
    niwalk, nmwalk, nirun, nmrun, niup, nmup, nidown, nmdown, nicar, nmcar,nisu,nmsu,nicy,nmcy = loadData(user,2)
    nis_walk,nig_walk=valSignal(niwalk,1)
    nis_run,nig_run=valSignal(nirun,1)
    nis_up,nig_up=valSignal(niup,1)
    nis_down,nig_down=valSignal(nidown,1)
    nis_su,nig_su=valSignal(nisu,1)
    nis_cy,nig_cy=valSignal(nicy,1)
    nis_car,nig_car=valSignal(nicar,1)
    
#-----USER8-------------------------------------------------------------------------------------------
    user = 'tutty'
    #CARGAR DATOS
    tuiwalk, tumwalk, tuirun, tumrun, tuiup, tumup, tuidown, tumdown, tuicar, tumcar,tuisu,tumsu,tuicy,tumcy = loadData(user,2)
    tuis_walk,tuig_walk=valSignal(tuiwalk,1)
    tuis_run,tuig_run=valSignal(tuirun,1)
    tuis_up,tuig_up=valSignal(tuiup,1)
    tuis_down,tuig_down=valSignal(tuidown,1)
    tuis_su,tuig_su=valSignal(tuisu,1)
    tuis_cy,tuig_cy=valSignal(tuicy,1)
    tuis_car,tuig_car=valSignal(tuicar,1)
    
    #FEATURE 1:MAGNITUD
    mdw = np.sum(ig_walk)
    mew = np.sum(eig_walk)
    mtw = np.sum(tig_walk)
    mjw= np.sum(jig_walk)
    mjgw= np.sum(jgig_walk)
    mjtw=np.sum(jtig_walk)
    mnw=np.sum(nig_walk)
    mtuw=np.sum(tuig_walk)
    
    mdr = np.sum(ig_run)
    mer = np.sum(eig_run)
    mtr = np.sum(tig_run)
    mjr= np.sum(jig_run)
    mjgr= np.sum(jgig_run)
    mjtr=np.sum(jtig_run)
    mnr=np.sum(nig_run)
    mtur=np.sum(tuig_run)
    
    mdu = np.sum(ig_up)
    meu = np.sum(eig_up)
    mtu = np.sum(tig_up)
    mju= np.sum(jig_up)
    mjgu= np.sum(jgig_up)
    mjtu=np.sum(jtig_up)
    mnu=np.sum(nig_up)
    mtuu=np.sum(tuig_up)
    
    mdd = np.sum(ig_down)
    med = np.sum(eig_down)
    mtd = np.sum(tig_down)
    mjd= np.sum(jig_down)
    mjgd= np.sum(jgig_down)
    mjtd=np.sum(jtig_down)
    mnd=np.sum(nig_down)
    mtud=np.sum(tuig_down)
    
    mds = np.sum(ig_su)
    mes = np.sum(eig_su)
    mts = np.sum(tig_su)
    mjs= np.sum(jig_su)
    mjgs= np.sum(jgig_su)
    mjts=np.sum(jtig_su)
    mns=np.sum(nig_su)
    mtus=np.sum(tuig_su)
    
    mdcy = np.sum(ig_cy)
    mecy = np.sum(eig_cy)
    mtcy = np.sum(tig_cy)
    mjcy= np.sum(jig_cy)
    mjgcy= np.sum(jgig_cy)
    mjtcy=np.sum(jtig_cy)
    mncy=np.sum(nig_cy)
    mtucy=np.sum(tuig_cy)
    
    mdc = np.sum(ig_car)
    mec = np.sum(eig_car)
    mtc = np.sum(tig_car)
    mjc= np.sum(jig_car)
    mjgc= np.sum(jgig_car)
    mjtc=np.sum(jtig_car)
    mnc=np.sum(nig_car)
    mtuc=np.sum(tuig_car)
    
    mC_1=[mdw,mew,mtw,mjw,mjgw,mjtw,mnw,mtuw]
    mC_2=[mdr,mer,mtr,mjr,mjgr,mjtr,mnr,mtur]
    mC_3=[mdu,meu,mtu,mju,mjgu,mjtu,mnu,mtuu]
    mC_4=[mdd,med,mtd,mjd,mjgd,mjtd,mnd,mtud]
    mC_5=[mds,mes,mts,mjs,mjgs,mjts,mns,mtus]
    mC_6=[mdcy,mecy,mtcy,mjcy,mjgcy,mjtcy,mncy,mtucy]
    mC_7=[mdc,mec,mtc,mjc,mjgc,mjtc,mnc,mtuc]
    
    Magnitud= [mC_1,mC_2,mC_3,mC_4,mC_5,mC_6,mC_7]
    
    #FEATURE 2:MEDIA
    mndw = np.mean(ig_walk)
    mnew = np.mean(eig_walk)
    mntw = np.mean(tig_walk)
    mnjw= np.mean(jig_walk)
    mnjgw= np.mean(jgig_walk)
    mnjtw=np.mean(jtig_walk)
    mnnw=np.mean(nig_walk)
    mntuw=np.mean(tuig_walk)
    
    mndr = np.mean(ig_run)
    mner = np.mean(eig_run)
    mntr = np.mean(tig_run)
    mnjr= np.mean(jig_run)
    mnjgr= np.mean(jgig_run)
    mnjtr=np.mean(jtig_run)
    mnnr=np.mean(nig_run)
    mntur=np.mean(tuig_run)
    
    mndu = np.mean(ig_up)
    mneu = np.mean(eig_up)
    mntu = np.mean(tig_up)
    mnju= np.mean(jig_up)
    mnjgu= np.mean(jgig_up)
    mnjtu=np.mean(jtig_up)
    mnnu=np.mean(nig_up)
    mntuu=np.mean(tuig_up)
    
    mndd = np.mean(ig_down)
    mned = np.mean(eig_down)
    mntd = np.mean(tig_down)
    mnjd= np.mean(jig_down)
    mnjgd= np.mean(jgig_down)
    mnjtd=np.mean(jtig_down)
    mnnd=np.mean(nig_down)
    mntud=np.mean(tuig_down)
    
    mnds = np.mean(ig_su)
    mnes = np.mean(eig_su)
    mnts = np.mean(tig_su)
    mnjs= np.mean(jig_su)
    mnjgs= np.mean(jgig_su)
    mnjts=np.mean(jtig_su)
    mnns=np.mean(nig_su)
    mntus=np.mean(tuig_su)
    
    mndcy = np.mean(ig_cy)
    mnecy = np.mean(eig_cy)
    mntcy = np.mean(tig_cy)
    mnjcy= np.mean(jig_cy)
    mnjgcy= np.mean(jgig_cy)
    mnjtcy=np.mean(jtig_cy)
    mnncy=np.mean(nig_cy)
    mntucy=np.mean(tuig_cy)
    
    mndc = np.mean(ig_car)
    mnec = np.mean(eig_car)
    mntc = np.mean(tig_car)
    mnjc= np.mean(jig_car)
    mnjgc= np.mean(jgig_car)
    mnjtc=np.mean(jtig_car)
    mnnc=np.mean(nig_car)
    mntuc=np.mean(tuig_car)
    
    
    C_1=[mndw,mnew,mntw,mnjw,mnjgw,mnjtw,mnnw,mntuw]
    C_2=[mndr,mner,mntr,mnjr,mnjgr,mnjtr,mnnr,mntur]
    C_3=[mndu,mneu,mntu,mnju,mnjgu,mnjtu,mnnu,mntuu]
    C_4=[mndd,mned,mntd,mnjd,mnjgd,mnjtd,mnnd,mntud]
    C_5=[mnds,mnes,mnts,mnjs,mnjgs,mnjts,mnns,mntus]
    C_6=[mndcy,mnecy,mntcy,mnjcy,mnjgcy,mnjtcy,mnncy,mntucy]
    C_7=[mndc,mnec,mntc,mnjc,mnjgc,mnjtc,mnnc,mntuc]
    
    Media=[C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 3:STD
    stddw = np.std(ig_walk)
    stdew = np.std(eig_walk)
    stdtw = np.std(tig_walk)
    stdjw= np.std(jig_walk)
    stdjgw= np.std(jgig_walk)
    stdjtw=np.std(jtig_walk)
    stdnw=np.std(nig_walk)
    stdtuw=np.std(tuig_walk)
    
    stddr = np.std(ig_run)
    stder = np.std(eig_run)
    stdtr = np.std(tig_run)
    stdjr= np.std(jig_run)
    stdjgr= np.std(jgig_run)
    stdjtr=np.std(jtig_run)
    stdnr=np.std(nig_run)
    stdtur=np.std(tuig_run)
    
    stddu = np.std(ig_up)
    stdeu = np.std(eig_up)
    stdtu = np.std(tig_up)
    stdju= np.std(jig_up)
    stdjgu= np.std(jgig_up)
    stdjtu=np.std(jtig_up)
    stdnu=np.std(nig_up)
    stdtuu=np.std(tuig_up)
    
    stddd = np.std(ig_down)
    stded = np.std(eig_down)
    stdtd = np.std(tig_down)
    stdjd= np.std(jig_down)
    stdjgd= np.std(jgig_down)
    stdjtd=np.std(jtig_down)
    stdnd=np.std(nig_down)
    stdtud=np.std(tuig_down)
    
    stdds = np.std(ig_su)
    stdes = np.std(eig_su)
    stdts = np.std(tig_su)
    stdjs= np.std(jig_su)
    stdjgs= np.std(jgig_su)
    stdjts=np.std(jtig_su)
    stdns=np.std(nig_su)
    stdtus=np.std(tuig_su)
    
    stddcy = np.std(ig_cy)
    stdecy = np.std(eig_cy)
    stdtcy = np.std(tig_cy)
    stdjcy= np.std(jig_cy)
    stdjgcy= np.std(jgig_cy)
    stdjtcy=np.std(jtig_cy)
    stdncy=np.std(nig_cy)
    stdtucy=np.std(tuig_cy)
    
    stddc = np.std(ig_car)
    stdec = np.std(eig_car)
    stdtc = np.std(tig_car)
    stdjc= np.std(jig_car)
    stdjgc= np.std(jgig_car)
    stdjtc=np.std(jtig_car)
    stdnc=np.std(nig_car)
    stdtuc=np.std(tuig_car)
    
    
    C_1=[stddw,stdew,stdtw,stdjw,stdjgw,stdjtw,stdnw,stdtuw]
    C_2=[stddr,stder,stdtr,stdjr,stdjgr,stdjtr,stdnr,stdtur]
    C_3=[stddu,stdeu,stdtu,stdju,stdjgu,stdjtu,stdnu,stdtuu]
    C_4=[stddd,stded,stdtd,stdjd,stdjgd,stdjtd,stdnd,stdtud]
    C_5=[stdds,stdes,stdts,stdjs,stdjgs,stdjts,stdns,stdtus]
    C_6=[stddcy,stdecy,stdtcy,stdjcy,stdjgcy,stdjtcy,stdncy,stdtucy]
    C_7=[stddc,stdec,stdtc,stdjc,stdjgc,stdjtc,stdnc,stdtuc]
    
    std=[C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 4:Kurtosis
    kdw = kurtosis(ig_walk)
    kew = kurtosis(eig_walk)
    ktw = kurtosis(tig_walk)
    kjw= kurtosis(jig_walk)
    kjgw= kurtosis(jgig_walk)
    kjtw=kurtosis(jtig_walk)
    knw=kurtosis(nig_walk)
    ktuw=kurtosis(tuig_walk)
    
    kdr = kurtosis(ig_run)
    ker = kurtosis(eig_run)
    ktr = kurtosis(tig_run)
    kjr= kurtosis(jig_run)
    kjgr= kurtosis(jgig_run)
    kjtr=kurtosis(jtig_run)
    knr=kurtosis(nig_run)
    ktur=kurtosis(tuig_run)
    
    kdu = kurtosis(ig_up)
    keu = kurtosis(eig_up)
    ktu = kurtosis(tig_up)
    kju= kurtosis(jig_up)
    kjgu= kurtosis(jgig_up)
    kjtu=kurtosis(jtig_up)
    knu=kurtosis(nig_up)
    ktuu=kurtosis(tuig_up)
    
    kdd = kurtosis(ig_down)
    ked = kurtosis(eig_down)
    ktd = kurtosis(tig_down)
    kjd= kurtosis(jig_down)
    kjgd= kurtosis(jgig_down)
    kjtd=kurtosis(jtig_down)
    knd=kurtosis(nig_down)
    ktud=kurtosis(tuig_down)
    
    kds = kurtosis(ig_su)
    kes = kurtosis(eig_su)
    kts = kurtosis(tig_su)
    kjs= kurtosis(jig_su)
    kjgs= kurtosis(jgig_su)
    kjts=kurtosis(jtig_su)
    kns=kurtosis(nig_su)
    ktus=kurtosis(tuig_su)
    
    kdcy = kurtosis(ig_cy)
    kecy = kurtosis(eig_cy)
    ktcy = kurtosis(tig_cy)
    kjcy= kurtosis(jig_cy)
    kjgcy= kurtosis(jgig_cy)
    kjtcy=kurtosis(jtig_cy)
    kncy=kurtosis(nig_cy)
    ktucy=kurtosis(tuig_cy)
    
    kdc = kurtosis(ig_car)
    kec = kurtosis(eig_car)
    ktc = kurtosis(tig_car)
    kjc= kurtosis(jig_car)
    kjgc= kurtosis(jgig_car)
    kjtc=kurtosis(jtig_car)
    knc=kurtosis(nig_car)
    ktuc=kurtosis(tuig_car)
    
    
    C_1=[kdw,kew,ktw,kjw,kjgw,kjtw,knw,ktuw]
    C_2=[kdr,ker,ktr,kjr,kjgr,kjtr,knr,ktur]
    C_3=[kdu,keu,ktu,kju,kjgu,kjtu,knu,ktuu]
    C_4=[kdd,ked,ktd,kjd,kjgd,kjtd,knd,ktud]
    C_5=[kds,kes,kts,kjs,kjgs,kjts,kns,ktus]
    C_6=[kdcy,kecy,ktcy,kjcy,kjgcy,kjtcy,kncy,ktucy]
    C_7=[kdc,kec,ktc,kjc,kjgc,kjtc,knc,ktuc]
    
    Kurtosis= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 5:MAXIMO
    mxdw = np.max(ig_walk)
    mxew = np.max(eig_walk)
    mxtw = np.max(tig_walk)
    mxjw= np.max(jig_walk)
    mxjgw= np.max(jgig_walk)
    mxjtw=np.max(jtig_walk)
    mxnw=np.max(nig_walk)
    mxtuw=np.max(tuig_walk)
    
    mxdr = np.max(ig_run)
    mxer = np.max(eig_run)
    mxtr = np.max(tig_run)
    mxjr= np.max(jig_run)
    mxjgr= np.max(jgig_run)
    mxjtr=np.max(jtig_run)
    mxnr=np.max(nig_run)
    mxtur=np.max(tuig_run)
    
    mxdu = np.max(ig_up)
    mxeu = np.max(eig_up)
    mxtu = np.max(tig_up)
    mxju= np.max(jig_up)
    mxjgu= np.max(jgig_up)
    mxjtu=np.max(jtig_up)
    mxnu=np.max(nig_up)
    mxtuu=np.max(tuig_up)
    
    mxdd = np.max(ig_down)
    mxed = np.max(eig_down)
    mxtd = np.max(tig_down)
    mxjd= np.max(jig_down)
    mxjgd= np.max(jgig_down)
    mxjtd=np.max(jtig_down)
    mxnd=np.max(nig_down)
    mxtud=np.max(tuig_down)
    
    mxds = np.max(ig_su)
    mxes = np.max(eig_su)
    mxts = np.max(tig_su)
    mxjs= np.max(jig_su)
    mxjgs= np.max(jgig_su)
    mxjts=np.max(jtig_su)
    mxns=np.max(nig_su)
    mxtus=np.max(tuig_su)
    
    mxdcy = np.max(ig_cy)
    mxecy = np.max(eig_cy)
    mxtcy = np.max(tig_cy)
    mxjcy= np.max(jig_cy)
    mxjgcy= np.max(jgig_cy)
    mxjtcy=np.max(jtig_cy)
    mxncy=np.max(nig_cy)
    mxtucy=np.max(tuig_cy)
    
    mxdc = np.max(ig_car)
    mxec = np.max(eig_car)
    mxtc = np.max(tig_car)
    mxjc= np.max(jig_car)
    mxjgc= np.max(jgig_car)
    mxjtc=np.max(jtig_car)
    mxnc=np.max(nig_car)
    mxtuc=np.max(tuig_car)
    
    
    C_1=[mxdw,mxew,mxtw,mxjw,mxjgw,mxjtw,mxnw,mxtuw]
    C_2=[mxdr,mxer,mxtr,mxjr,mxjgr,mxjtr,mxnr,mxtur]
    C_3=[mxdu,mxeu,mxtu,mxju,mxjgu,mxjtu,mxnu,mxtuu]
    C_4=[mxdd,mxed,mxtd,mxjd,mxjgd,mxjtd,mxnd,mxtud]
    C_5=[mxds,mxes,mxts,mxjs,mxjgs,mxjts,mxns,mxtus]
    C_6=[mxdcy,mxecy,mxtcy,mxjcy,mxjgcy,mxjtcy,mxncy,mxtucy]
    C_7=[mxdc,mxec,mxtc,mxjc,mxjgc,mxjtc,mxnc,mxtuc]
    
    Max= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 6:MINIMO
    mindw = np.min(ig_walk)
    minew = np.min(eig_walk)
    mintw = np.min(tig_walk)
    minjw= np.min(jig_walk)
    minjgw= np.min(jgig_walk)
    minjtw=np.min(jtig_walk)
    minnw=np.min(nig_walk)
    mintuw=np.min(tuig_walk)
    
    mindr = np.min(ig_run)
    miner = np.min(eig_run)
    mintr = np.min(tig_run)
    minjr= np.min(jig_run)
    minjgr= np.min(jgig_run)
    minjtr=np.min(jtig_run)
    minnr=np.min(nig_run)
    mintur=np.min(tuig_run)
    
    mindu = np.min(ig_up)
    mineu = np.min(eig_up)
    mintu = np.min(tig_up)
    minju= np.min(jig_up)
    minjgu= np.min(jgig_up)
    minjtu=np.min(jtig_up)
    minnu=np.min(nig_up)
    mintuu=np.min(tuig_up)
    
    mindd = np.min(ig_down)
    mined = np.min(eig_down)
    mintd = np.min(tig_down)
    minjd= np.min(jig_down)
    minjgd= np.min(jgig_down)
    minjtd=np.min(jtig_down)
    minnd=np.min(nig_down)
    mintud=np.min(tuig_down)
    
    minds = np.min(ig_su)
    mines = np.min(eig_su)
    mints = np.min(tig_su)
    minjs= np.min(jig_su)
    minjgs= np.min(jgig_su)
    minjts=np.min(jtig_su)
    minns=np.min(nig_su)
    mintus=np.min(tuig_su)
    
    mindcy = np.min(ig_cy)
    minecy = np.min(eig_cy)
    mintcy = np.min(tig_cy)
    minjcy= np.min(jig_cy)
    minjgcy= np.min(jgig_cy)
    minjtcy=np.min(jtig_cy)
    minncy=np.min(nig_cy)
    mintucy=np.min(tuig_cy)
    
    mindc = np.min(ig_car)
    minec = np.min(eig_car)
    mintc = np.min(tig_car)
    minjc= np.min(jig_car)
    minjgc= np.min(jgig_car)
    minjtc=np.min(jtig_car)
    minnc=np.min(nig_car)
    mintuc=np.min(tuig_car)
    
    
    C_1=[mindw,minew,mintw,minjw,minjgw,minjtw,minnw,mintuw]
    C_2=[mindr,miner,mintr,minjr,minjgr,minjtr,minnr,mintur]
    C_3=[mindu,mineu,mintu,minju,minjgu,minjtu,minnu,mintuu]
    C_4=[mindd,mined,mintd,minjd,minjgd,minjtd,minnd,mintud]
    C_5=[minds,mines,mints,minjs,minjgs,minjts,minns,mintus]
    C_6=[mindcy,minecy,mintcy,minjcy,minjgcy,minjtcy,minncy,mintucy]
    C_7=[mindc,minec,mintc,minjc,minjgc,minjtc,minnc,mintuc]
    
    Min= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 7:RANGO
    rndw = mxdw-mindw
    rnew = mxew-minew
    rntw = mxtw-mintw
    rnjw = mxjw-minjw
    rnjgw = mxjgw-minjgw
    rnjtw = mxjtw-minjtw
    rnnw = mxnw-minnw
    rntuw = mxtuw-mintuw
    
    
    rndr = mxdr-mindr
    rner = mxer-miner
    rntr = mxtr-mintr
    rnjr = mxjr-minjr
    rnjgr = mxjgr-minjgr
    rnjtr = mxjtr-minjtr
    rnnr = mxnr-minnr
    rntur = mxtur-mintur
    
    rndu = mxdu-mindu
    rneu = mxeu-mineu
    rntu = mxtu-mintu
    rnju = mxju-minju
    rnjgu = mxjgu-minjgu
    rnjtu = mxjtu-minjtu
    rnnu = mxnu-minnu
    rntuu = mxtuu-mintuu
    
    rndd = mxdd-mindd
    rned = mxed-mined
    rntd = mxtd-mintd
    rnjd = mxjd-minjd
    rnjgd = mxjgd-minjgd
    rnjtd = mxjtd-minjtd
    rnnd = mxnd-minnd
    rntud = mxtud-mintud
    
    rnds = mxds-minds
    rnes = mxes-mines
    rnts = mxts-mints
    rnjs = mxjs-minjs
    rnjgs = mxjgs-minjgs
    rnjts = mxjts-minjts
    rnns = mxns-minns
    rntus = mxtus-mintus
    
    rndcy = mxdcy-mindcy
    rnecy = mxecy-minecy
    rntcy = mxtcy-mintcy
    rnjcy = mxjcy-minjcy
    rnjgcy = mxjgcy-minjgcy
    rnjtcy = mxjtcy-minjtcy
    rnncy = mxncy-minncy
    rntucy = mxtucy-mintucy
    
    rndc = mxdc-mindc
    rnec = mxec-minec
    rntc = mxtc-mintc
    rnjc = mxjc-minjc
    rnjgc = mxjgc-minjgc
    rnjtc = mxjtc-minjtc
    rnnc = mxnc-minnc
    rntuc = mxtuc-mintuc
    
    C_1=[rndw,rnew,rntw,rnjw,rnjgw,rnjtw,rnnw,rntuw]
    C_2=[rndr,rner,rntr,rnjr,rnjgr,rnjtr,rnnr,rntur]
    C_3=[rndu,rneu,rntu,rnju,rnjgu,rnjtu,rnnu,rntuu]
    C_4=[rndd,rned,rntd,rnjd,rnjgd,rnjtd,rnnd,rntud]
    C_5=[rnds,rnes,rnts,rnjs,rnjgs,rnjts,rnns,rntus]
    C_6=[rndcy,rnecy,rntcy,rnjcy,rnjgcy,rnjtcy,rnncy,rntucy]
    C_7=[rndc,rnec,rntc,rnjc,rnjgc,rnjtc,rnnc,rntuc]
    
    Range= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 8: MXFOURIER
    mfdw = feature8(diwalk,ig_walk)
    mfew = feature8(eiwalk,eig_walk)
    mftw = feature8(tiwalk,tig_walk)
    mfjw=  feature8(jiwalk,jig_walk)
    mfjgw = feature8(jgiwalk,jgig_walk)
    mfjtw = feature8(jtiwalk,jtig_walk)
    mfnw = feature8(niwalk,nig_walk)
    mftuw=  feature8(tuiwalk,tuig_walk)
    
    mfdr = feature8(dirun,ig_run)
    mfer = feature8(eirun,eig_run)
    mftr = feature8(tirun,tig_run)
    mfjr=  feature8(jirun,jig_run)
    mfjgr = feature8(jgirun,jgig_run)
    mfjtr = feature8(jtirun,jtig_run)
    mfnr = feature8(nirun,nig_run)
    mftur=  feature8(tuirun,tuig_run)
    
    mfdu = feature8(diup,ig_up)
    mfeu = feature8(eiup,eig_up)
    mftu = feature8(tiup,tig_up)
    mfju=  feature8(jiup,jig_up)
    mfjgu = feature8(jgiup,jgig_up)
    mfjtu = feature8(jtiup,jtig_up)
    mfnu = feature8(niup,nig_up)
    mftuu=  feature8(tuiup,tuig_up)
    
    mfdd = feature8(didown,ig_down)
    mfed = feature8(eidown,eig_down)
    mftd = feature8(tidown,tig_down)
    mfjd=  feature8(jidown,jig_down)
    mfjgd = feature8(jgidown,jgig_down)
    mfjtd = feature8(jtidown,jtig_down)
    mfnd = feature8(nidown,nig_down)
    mftud=  feature8(tuidown,tuig_down)
    
    mfds = feature8(disu,ig_su)
    mfes = feature8(eisu,eig_su)
    mfts = feature8(tisu,tig_su)
    mfjs=  feature8(jisu,jig_su)
    mfjgs = feature8(jgisu,jgig_su)
    mfjts = feature8(jtisu,jtig_su)
    mfns = feature8(nisu,nig_su)
    mftus=  feature8(tuisu,tuig_su)
    
    mfdcy = feature8(dicy,ig_cy)
    mfecy = feature8(eicy,eig_cy)
    mftcy = feature8(ticy,tig_cy)
    mfjcy=  feature8(jicy,jig_cy)
    mfjgcy = feature8(jgicy,jgig_cy)
    mfjtcy = feature8(jticy,jtig_cy)
    mfncy = feature8(nicy,nig_cy)
    mftucy=  feature8(tuicy,tuig_cy)
    
    mfdc = feature8(dicar,ig_car)
    mfec = feature8(eicar,eig_car)
    mftc = feature8(ticar,tig_car)
    mfjc=  feature8(jicar,jig_car)
    mfjgc = feature8(jgicar,jgig_car)
    mfjtc = feature8(jticar,jtig_car)
    mfnc = feature8(nicar,nig_car)
    mftuc=  feature8(tuicar,tuig_car)
    
    C_1=[mfdw,mfew,mftw,mfjw,mfjgw,mfjtw,mfnw,mftuw]
    C_2=[mfdr,mfer,mftr,mfjr,mfjgr,mfjtr,mfnr,mftur]
    C_3=[mfdu,mfeu,mftu,mfju,mfjgu,mfjtu,mfnu,mftuu]
    C_4=[mfdd,mfed,mftd,mfjd,mfjgd,mfjtd,mfnd,mftud]
    C_5=[mfds,mfes,mfts,mfjs,mfjgs,mfjts,mfns,mftus]
    C_6=[mfdcy,mfecy,mftcy,mfjcy,mfjgcy,mfjtcy,mfncy,mftucy]
    C_7=[mfdc,mfec,mftc,mfjc,mfjgc,mfjtc,mfnc,mftuc]
    
    mFourier= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 9 : Freq
    fdw = feature9(diwalk,ig_walk)
    few = feature9(eiwalk,eig_walk)
    ftw = feature9(tiwalk,tig_walk)
    fjw=  feature9(jiwalk,jig_walk)
    fjgw = feature9(jgiwalk,jgig_walk)
    fjtw = feature9(jtiwalk,jtig_walk)
    fnw = feature9(niwalk,nig_walk)
    ftuw=  feature9(tuiwalk,tuig_walk)
    
    fdr = feature9(dirun,ig_run)
    fer = feature9(eirun,eig_run)
    ftr = feature9(tirun,tig_run)
    fjr=  feature9(jirun,jig_run)
    fjgr = feature9(jgirun,jgig_run)
    fjtr = feature9(jtirun,jtig_run)
    fnr = feature9(nirun,nig_run)
    ftur=  feature9(tuirun,tuig_run)
    
    fdu = feature9(diup,ig_up)
    feu = feature9(eiup,eig_up)
    ftu = feature9(tiup,tig_up)
    fju=  feature9(jiup,jig_up)
    fjgu = feature9(jgiup,jgig_up)
    fjtu = feature9(jtiup,jtig_up)
    fnu = feature9(niup,nig_up)
    ftuu=  feature9(tuiup,tuig_up)
    
    fdd = feature9(didown,ig_down)
    fed = feature9(eidown,eig_down)
    ftd = feature9(tidown,tig_down)
    fjd=  feature9(jidown,jig_down)
    fjgd = feature9(jgidown,jgig_down)
    fjtd = feature9(jtidown,jtig_down)
    fnd = feature9(nidown,nig_down)
    ftud=  feature9(tuidown,tuig_down)
    
    fds = feature9(disu,ig_su)
    fes = feature9(eisu,eig_su)
    fts = feature9(tisu,tig_su)
    fjs=  feature9(jisu,jig_su)
    fjgs = feature9(jgisu,jgig_su)
    fjts = feature9(jtisu,jtig_su)
    fns = feature9(nisu,nig_su)
    ftus=  feature9(tuisu,tuig_su)
    
    fdcy = feature9(dicy,ig_cy)
    fecy = feature9(eicy,eig_cy)
    ftcy = feature9(ticy,tig_cy)
    fjcy=  feature9(jicy,jig_cy)
    fjgcy = feature9(jgicy,jgig_cy)
    fjtcy = feature9(jticy,jtig_cy)
    fncy = feature9(nicy,nig_cy)
    ftucy=  feature9(tuicy,tuig_cy)
    
    fdc = feature9(dicar,ig_car)
    fec = feature9(eicar,eig_car)
    ftc = feature9(ticar,tig_car)
    fjc=  feature9(jicar,jig_car)
    fjgc = feature9(jgicar,jgig_car)
    fjtc = feature9(jticar,jtig_car)
    fnc = feature9(nicar,nig_car)
    ftuc=  feature9(tuicar,tuig_car)
    
    C_1=[fdw,few,ftw,fjw,fjgw,fjtw,fnw,ftuw]
    C_2=[fdr,fer,ftr,fjr,fjgr,fjtr,fnr,ftur]
    C_3=[fdu,feu,ftu,fju,fjgu,fjtu,fnu,ftuu]
    C_4=[fdd,fed,ftd,fjd,fjgd,fjtd,fnd,ftud]
    C_5=[fds,fes,fts,fjs,fjgs,fjts,fns,ftus]
    C_6=[fdcy,fecy,ftcy,fjcy,fjgcy,fjtcy,fncy,ftucy]
    C_7=[fdc,fec,ftc,fjc,fjgc,fjtc,fnc,ftuc]
    
    Freq= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 10 : MediaFreq
    fmdw = feature10(diwalk,ig_walk)
    fmew = feature10(eiwalk,eig_walk)
    fmtw = feature10(tiwalk,tig_walk)
    fmjw=  feature10(jiwalk,jig_walk)
    fmjgw = feature10(jgiwalk,jgig_walk)
    fmjtw = feature10(jtiwalk,jtig_walk)
    fmnw = feature10(niwalk,nig_walk)
    fmtuw=  feature10(tuiwalk,tuig_walk)
    
    fmdr = feature10(dirun,ig_run)
    fmer = feature10(eirun,eig_run)
    fmtr = feature10(tirun,tig_run)
    fmjr=  feature10(jirun,jig_run)
    fmjgr = feature10(jgirun,jgig_run)
    fmjtr = feature10(jtirun,jtig_run)
    fmnr = feature10(nirun,nig_run)
    fmtur=  feature10(tuirun,tuig_run)
    
    fmdu = feature10(diup,ig_up)
    fmeu = feature10(eiup,eig_up)
    fmtu = feature10(tiup,tig_up)
    fmju=  feature10(jiup,jig_up)
    fmjgu = feature10(jgiup,jgig_up)
    fmjtu = feature10(jtiup,jtig_up)
    fmnu = feature10(niup,nig_up)
    fmtuu=  feature10(tuiup,tuig_up)
    
    fmdd = feature10(didown,ig_down)
    fmed = feature10(eidown,eig_down)
    fmtd = feature10(tidown,tig_down)
    fmjd=  feature10(jidown,jig_down)
    fmjgd = feature10(jgidown,jgig_down)
    fmjtd = feature10(jtidown,jtig_down)
    fmnd = feature10(nidown,nig_down)
    fmtud=  feature10(tuidown,tuig_down)
    
    fmds = feature10(disu,ig_su)
    fmes = feature10(eisu,eig_su)
    fmts = feature10(tisu,tig_su)
    fmjs=  feature10(jisu,jig_su)
    fmjgs = feature10(jgisu,jgig_su)
    fmjts = feature10(jtisu,jtig_su)
    fmns = feature10(nisu,nig_su)
    fmtus=  feature10(tuisu,tuig_su)
    
    fmdcy = feature10(dicy,ig_cy)
    fmecy = feature10(eicy,eig_cy)
    fmtcy = feature10(ticy,tig_cy)
    fmjcy=  feature10(jicy,jig_cy)
    fmjgcy = feature10(jgicy,jgig_cy)
    fmjtcy = feature10(jticy,jtig_cy)
    fmncy = feature10(nicy,nig_cy)
    fmtucy=  feature10(tuicy,tuig_cy)
    
    fmdc = feature10(dicar,ig_car)
    fmec = feature10(eicar,eig_car)
    fmtc = feature10(ticar,tig_car)
    fmjc=  feature10(jicar,jig_car)
    fmjgc = feature10(jgicar,jgig_car)
    fmjtc = feature10(jticar,jtig_car)
    fmnc = feature10(nicar,nig_car)
    fmtuc=  feature10(tuicar,tuig_car)
    
    C_1=[fmdw,fmew,fmtw,fmjw,fmjgw,fmjtw,fmnw,fmtuw]
    C_2=[fmdr,fmer,fmtr,fmjr,fmjgr,fmjtr,fmnr,fmtur]
    C_3=[fmdu,fmeu,fmtu,fmju,fmjgu,fmjtu,fmnu,fmtuu]
    C_4=[fmdd,fmed,fmtd,fmjd,fmjgd,fmjtd,fmnd,fmtud]
    C_5=[fmds,fmes,fmts,fmjs,fmjgs,fmjts,fmns,fmtus]
    C_6=[fmdcy,fmecy,fmtcy,fmjcy,fmjgcy,fmjtcy,fmncy,fmtucy]
    C_7=[fmdc,fmec,fmtc,fmjc,fmjgc,fmjtc,fmnc,fmtuc]
    freqMean= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 11 : FreqSTD
    fstddw = feature11(diwalk,ig_walk)
    fstdew = feature11(eiwalk,eig_walk)
    fstdtw = feature11(tiwalk,tig_walk)
    fstdjw=  feature11(jiwalk,jig_walk)
    fstdjgw = feature11(jgiwalk,jgig_walk)
    fstdjtw = feature11(jtiwalk,jtig_walk)
    fstdnw = feature11(niwalk,nig_walk)
    fstdtuw=  feature11(tuiwalk,tuig_walk)
    
    fstddr = feature11(dirun,ig_run)
    fstder = feature11(eirun,eig_run)
    fstdtr = feature11(tirun,tig_run)
    fstdjr=  feature11(jirun,jig_run)
    fstdjgr = feature11(jgirun,jgig_run)
    fstdjtr = feature11(jtirun,jtig_run)
    fstdnr = feature11(nirun,nig_run)
    fstdtur=  feature11(tuirun,tuig_run)
    
    fstddu = feature11(diup,ig_up)
    fstdeu = feature11(eiup,eig_up)
    fstdtu = feature11(tiup,tig_up)
    fstdju=  feature11(jiup,jig_up)
    fstdjgu = feature11(jgiup,jgig_up)
    fstdjtu = feature11(jtiup,jtig_up)
    fstdnu = feature11(niup,nig_up)
    fstdtuu=  feature11(tuiup,tuig_up)
    
    fstddd = feature11(didown,ig_down)
    fstded = feature11(eidown,eig_down)
    fstdtd = feature11(tidown,tig_down)
    fstdjd=  feature11(jidown,jig_down)
    fstdjgd = feature11(jgidown,jgig_down)
    fstdjtd = feature11(jtidown,jtig_down)
    fstdnd = feature11(nidown,nig_down)
    fstdtud=  feature11(tuidown,tuig_down)
    
    fstdds = feature11(disu,ig_su)
    fstdes = feature11(eisu,eig_su)
    fstdts = feature11(tisu,tig_su)
    fstdjs=  feature11(jisu,jig_su)
    fstdjgs = feature11(jgisu,jgig_su)
    fstdjts = feature11(jtisu,jtig_su)
    fstdns = feature11(nisu,nig_su)
    fstdtus=  feature11(tuisu,tuig_su)
    
    fstddcy = feature11(dicy,ig_cy)
    fstdecy = feature11(eicy,eig_cy)
    fstdtcy = feature11(ticy,tig_cy)
    fstdjcy=  feature11(jicy,jig_cy)
    fstdjgcy = feature11(jgicy,jgig_cy)
    fstdjtcy = feature11(jticy,jtig_cy)
    fstdncy = feature11(nicy,nig_cy)
    fstdtucy=  feature11(tuicy,tuig_cy)
    
    fstddc = feature11(dicar,ig_car)
    fstdec = feature11(eicar,eig_car)
    fstdtc = feature11(ticar,tig_car)
    fstdjc=  feature11(jicar,jig_car)
    fstdjgc = feature11(jgicar,jgig_car)
    fstdjtc = feature11(jticar,jtig_car)
    fstdnc = feature11(nicar,nig_car)
    fstdtuc=  feature11(tuicar,tuig_car)
    
    C_1=[fstddw,fstdew,fstdtw,fstdjw,fstdjgw,fstdjtw,fstdnw,fstdtuw]
    C_2=[fstddr,fstder,fstdtr,fstdjr,fstdjgr,fstdjtr,fstdnr,fstdtur]
    C_3=[fstddu,fstdeu,fstdtu,fstdju,fstdjgu,fstdjtu,fstdnu,fstdtuu]
    C_4=[fstddd,fstded,fstdtd,fstdjd,fstdjgd,fstdjtd,fstdnd,fstdtud]
    C_5=[fstdds,fstdes,fstdts,fstdjs,fstdjgs,fstdjts,fstdns,fstdtus]
    C_6=[fstddcy,fstdecy,fstdtcy,fstdjcy,fstdjgcy,fstdjtcy,fstdncy,fstdtucy]
    C_7=[fstddc,fstdec,fstdtc,fstdjc,fstdjgc,fstdjtc,fstdnc,fstdtuc]
    
    Fstd= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 12 : Media Velocidad
    vdw = lSpeed(diwalk)
    vew = lSpeed(eiwalk)
    vtw = lSpeed(tiwalk)
    vjw=  lSpeed(jiwalk)
    vjgw = lSpeed(jgiwalk)
    vjtw = lSpeed(jtiwalk)
    vnw = lSpeed(niwalk)
    vtuw=  lSpeed(tuiwalk)
    
    vdr = lSpeed(dirun)
    ver = lSpeed(eirun)
    vtr = lSpeed(tirun)
    vjr=  lSpeed(jirun)
    vjgr = lSpeed(jgirun)
    vjtr = lSpeed(jtirun)
    vnr = lSpeed(nirun)
    vtur=  lSpeed(tuirun)
    
    vdu = lSpeed(diup)
    veu = lSpeed(eiup)
    vtu = lSpeed(tiup)
    vju=  lSpeed(jiup)
    vjgu = lSpeed(jgiup)
    vjtu = lSpeed(jtiup)
    vnu = lSpeed(niup)
    vtuu=  lSpeed(tuiup)
    
    vdd = lSpeed(didown)
    ved = lSpeed(eidown)
    vtd = lSpeed(tidown)
    vjd=  lSpeed(jidown)
    vjgd = lSpeed(jgidown)
    vjtd = lSpeed(jtidown)
    vnd = lSpeed(nidown)
    vtud=  lSpeed(tuidown)
    
    vds = lSpeed(disu)
    ves = lSpeed(eisu)
    vts = lSpeed(tisu)
    vjs=  lSpeed(jisu)
    vjgs = lSpeed(jgisu)
    vjts = lSpeed(jtisu)
    vns = lSpeed(nisu)
    vtus=  lSpeed(tuisu)
    
    vdcy = lSpeed(dicy)
    vecy = lSpeed(eicy)
    vtcy = lSpeed(ticy)
    vjcy=  lSpeed(jicy)
    vjgcy = lSpeed(jgicy)
    vjtcy = lSpeed(jticy)
    vncy = lSpeed(nicy)
    vtucy=  lSpeed(tuicy)
    
    vdc = lSpeed(dicar)
    vec = lSpeed(eicar)
    vtc = lSpeed(ticar)
    vjc=  lSpeed(jicar)
    vjgc = lSpeed(jgicar)
    vjtc = lSpeed(jticar)
    vnc = lSpeed(nicar)
    vtuc=  lSpeed(tuicar)
    
    C_1=[vdw,vew,vtw,vjw,vjgw,vjtw,vnw,vtuw]
    C_2=[vdr,ver,vtr,vjr,vjgr,vjtr,vnr,vtur]
    C_3=[vdu,veu,vtu,vju,vjgu,vjtu,vnu,vtuu]
    C_4=[vdd,ved,vtd,vjd,vjgd,vjtd,vnd,vtud]
    C_5=[vds,ves,vts,vjs,vjgs,vjts,vns,vtus]
    C_6=[vdcy,vecy,vtcy,vjcy,vjgcy,vjtcy,vncy,vtucy]
    C_7=[vdc,vec,vtc,vjc,vjgc,vjtc,vnc,vtuc]
    
    v= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 13 : Media Derivada
    mddw = feature13(diwalk)
    mdew = feature13(eiwalk)
    mdtw = feature13(tiwalk)
    mdjw=  feature13(jiwalk)
    mdjgw = feature13(jgiwalk)
    mdjtw = feature13(jtiwalk)
    mdnw = feature13(niwalk)
    mdtuw=  feature13(tuiwalk)
    
    mddr = feature13(dirun)
    mder = feature13(eirun)
    mdtr = feature13(tirun)
    mdjr=  feature13(jirun)
    mdjgr = feature13(jgirun)
    mdjtr = feature13(jtirun)
    mdnr = feature13(nirun)
    mdtur=  feature13(tuirun)
    
    mddu = feature13(diup)
    mdeu = feature13(eiup)
    mdtu = feature13(tiup)
    mdju=  feature13(jiup)
    mdjgu = feature13(jgiup)
    mdjtu = feature13(jtiup)
    mdnu = feature13(niup)
    mdtuu=  feature13(tuiup)
    
    mddd = feature13(didown)
    mded = feature13(eidown)
    mdtd = feature13(tidown)
    mdjd=  feature13(jidown)
    mdjgd = feature13(jgidown)
    mdjtd = feature13(jtidown)
    mdnd = feature13(nidown)
    mdtud=  feature13(tuidown)
    
    mdds = feature13(disu)
    mdes = feature13(eisu)
    mdts = feature13(tisu)
    mdjs=  feature13(jisu)
    mdjgs = feature13(jgisu)
    mdjts = feature13(jtisu)
    mdns = feature13(nisu)
    mdtus=  feature13(tuisu)
    
    mddcy = feature13(dicy)
    mdecy = feature13(eicy)
    mdtcy = feature13(ticy)
    mdjcy=  feature13(jicy)
    mdjgcy = feature13(jgicy)
    mdjtcy = feature13(jticy)
    mdncy = feature13(nicy)
    mdtucy=  feature13(tuicy)
    
    mddc = feature13(dicar)
    mdec = feature13(eicar)
    mdtc = feature13(ticar)
    mdjc=  feature13(jicar)
    mdjgc = feature13(jgicar)
    mdjtc = feature13(jticar)
    mdnc = feature13(nicar)
    mdtuc=  feature13(tuicar)
    
    C_1=[mddw,mdew,mdtw,mdjw,mdjgw,mdjtw,mdnw,mdtuw]
    C_2=[mddr,mder,mdtr,mdjr,mdjgr,mdjtr,mdnr,mdtur]
    C_3=[mddu,mdeu,mdtu,mdju,mdjgu,mdjtu,mdnu,mdtuu]
    C_4=[mddd,mded,mdtd,mdjd,mdjgd,mdjtd,mdnd,mdtud]
    C_5=[mdds,mdes,mdts,mdjs,mdjgs,mdjts,mdns,mdtus]
    C_6=[mddcy,mdecy,mdtcy,mdjcy,mdjgcy,mdjtcy,mdncy,mdtucy]
    C_7=[mddc,mdec,mdtc,mdjc,mdjgc,mdjtc,mdnc,mdtuc]
    
    md= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 14 : STD Derivada
    mdstddw = feature14(diwalk)
    mdstdew = feature14(eiwalk)
    mdstdtw = feature14(tiwalk)
    mdstdjw=  feature14(jiwalk)
    mdstdjgw = feature14(jgiwalk)
    mdstdjtw = feature14(jtiwalk)
    mdstdnw = feature14(niwalk)
    mdstdtuw=  feature14(tuiwalk)
    
    mdstddr = feature14(dirun)
    mdstder = feature14(eirun)
    mdstdtr = feature14(tirun)
    mdstdjr=  feature14(jirun)
    mdstdjgr = feature14(jgirun)
    mdstdjtr = feature14(jtirun)
    mdstdnr = feature14(nirun)
    mdstdtur=  feature14(tuirun)
    
    mdstddu = feature14(diup)
    mdstdeu = feature14(eiup)
    mdstdtu = feature14(tiup)
    mdstdju=  feature14(jiup)
    mdstdjgu = feature14(jgiup)
    mdstdjtu = feature14(jtiup)
    mdstdnu = feature14(niup)
    mdstdtuu=  feature14(tuiup)
    
    mdstddd = feature14(didown)
    mdstded = feature14(eidown)
    mdstdtd = feature14(tidown)
    mdstdjd=  feature14(jidown)
    mdstdjgd = feature14(jgidown)
    mdstdjtd = feature14(jtidown)
    mdstdnd = feature14(nidown)
    mdstdtud=  feature14(tuidown)
    
    mdstdds = feature14(disu)
    mdstdes = feature14(eisu)
    mdstdts = feature14(tisu)
    mdstdjs=  feature14(jisu)
    mdstdjgs = feature14(jgisu)
    mdstdjts = feature14(jtisu)
    mdstdns = feature14(nisu)
    mdstdtus=  feature14(tuisu)
    
    mdstddcy = feature14(dicy)
    mdstdecy = feature14(eicy)
    mdstdtcy = feature14(ticy)
    mdstdjcy=  feature14(jicy)
    mdstdjgcy = feature14(jgicy)
    mdstdjtcy = feature14(jticy)
    mdstdncy = feature14(nicy)
    mdstdtucy=  feature14(tuicy)
    
    mdstddc = feature14(dicar)
    mdstdec = feature14(eicar)
    mdstdtc = feature14(ticar)
    mdstdjc=  feature14(jicar)
    mdstdjgc = feature14(jgicar)
    mdstdjtc = feature14(jticar)
    mdstdnc = feature14(nicar)
    mdstdtuc=  feature14(tuicar)
    
    C_1=[mdstddw,mdstdew,mdstdtw,mdstdjw,mdstdjgw,mdstdjtw,mdstdnw,mdstdtuw]
    C_2=[mdstddr,mdstder,mdstdtr,mdstdjr,mdstdjgr,mdstdjtr,mdstdnr,mdstdtur]
    C_3=[mdstddu,mdstdeu,mdstdtu,mdstdju,mdstdjgu,mdstdjtu,mdstdnu,mdstdtuu]
    C_4=[mdstddd,mdstded,mdstdtd,mdstdjd,mdstdjgd,mdstdjtd,mdstdnd,mdstdtud]
    C_5=[mdstdds,mdstdes,mdstdts,mdstdjs,mdstdjgs,mdstdjts,mdstdns,mdstdtus]
    C_6=[mdstddcy,mdstdecy,mdstdtcy,mdstdjcy,mdstdjgcy,mdstdjtcy,mdstdncy,mdstdtucy]
    C_7=[mdstddc,mdstdec,mdstdtc,mdstdjc,mdstdjgc,mdstdjtc,mdstdnc,mdstdtuc]
    
    mdstd= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    Mag_1mean=np.mean(Magnitud[0])
    Mag_2mean=np.mean(Magnitud[1])
    Mag_3mean=np.mean(Magnitud[2])
    Mag_4mean=np.mean(Magnitud[3])
    Mag_5mean=np.mean(Magnitud[4])
    Mag_6mean=np.mean(Magnitud[5])
    Mag_7mean=np.mean(Magnitud[6])
    
    Mag_1std=np.std(Magnitud[0])
    Mag_2std=np.std(Magnitud[1])
    Mag_3std=np.std(Magnitud[2])
    Mag_4std=np.std(Magnitud[3])
    Mag_5std=np.std(Magnitud[4])
    Mag_6std=np.std(Magnitud[5])
    Mag_7std=np.std(Magnitud[6])
   
    Mag_d0 = norm(loc = Mag_1mean, scale=Mag_1std)
    Mag_d1 = norm(loc = Mag_2mean, scale=Mag_2std)
    Mag_d2 = norm(loc = Mag_3mean, scale=Mag_3std)
    Mag_d3 = norm(loc = Mag_4mean, scale=Mag_4std)
    Mag_d4 = norm(loc = Mag_5mean, scale=Mag_5std)
    Mag_d5 = norm(loc = Mag_6mean, scale=Mag_6std)
    Mag_d6 = norm(loc = Mag_7mean, scale=Mag_7std)

    x1 = np.arange(-.5, 2000, 2000.5/100.0)
   
    #plot tee pdfs of teese normal distributions
    plt.figure(1, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, Mag_d0.pdf(x1),label='walk')
    plt.plot(x1 , Mag_d1.pdf(x1),label='run')
    plt.plot(x1, Mag_d2.pdf(x1),label='up')
    plt.plot(x1 , Mag_d3.pdf(x1),label='down')
    plt.plot(x1, Mag_d4.pdf(x1),label='static')
    plt.plot(x1 , Mag_d5.pdf(x1),label='cycle')
    plt.plot(x1 , Mag_d6.pdf(x1),label='car')    
    plt.title("Magnitude")
    plt.legend(loc="upper left") 
    plt.show()
    
    #GRAFICA 2
    Med_1mean=np.mean(Media[0])
    Med_2mean=np.mean(Media[1])
    Med_3mean=np.mean(Media[2])
    Med_4mean=np.mean(Media[3])
    Med_5mean=np.mean(Media[4])
    Med_6mean=np.mean(Media[5])
    Med_7mean=np.mean(Media[6])
    
    Med_1std=np.std(Media[0])
    Med_2std=np.std(Media[1])
    Med_3std=np.std(Media[2])
    Med_4std=np.std(Media[3])
    Med_5std=np.std(Media[4])
    Med_6std=np.std(Media[5])
    Med_7std=np.std(Media[6])
   
    Med_d0 = norm(loc = Med_1mean, scale=Med_1std)
    Med_d1 = norm(loc = Med_2mean, scale=Med_2std)
    Med_d2 = norm(loc = Med_3mean, scale=Med_3std)
    Med_d3 = norm(loc = Med_4mean, scale=Med_4std)
    Med_d4 = norm(loc = Med_5mean, scale=Med_5std)
    Med_d5 = norm(loc = Med_6mean, scale=Med_6std)
    Med_d6 = norm(loc = Med_7mean, scale=Med_7std)

    x1 = np.arange(-.5, 6, 6.5/1000.0)
   
    #plot the pdfs of these normal distributions
    plt.figure(2, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, Med_d0.pdf(x1),label='walk')
    plt.plot(x1 , Med_d1.pdf(x1),label='run')
    plt.plot(x1, Med_d2.pdf(x1),label='up')
    plt.plot(x1 , Med_d3.pdf(x1),label='down')
    plt.plot(x1, Med_d4.pdf(x1),label='static')
    plt.plot(x1 , Med_d5.pdf(x1),label='cycle')
    plt.plot(x1 , Med_d6.pdf(x1),label='car')    
    plt.title("Mean")
    plt.legend(loc="upper right") 
    plt.show()
    
    #FIGURA3
    std_1mean=np.mean(std[0])
    std_2mean=np.mean(std[1])
    std_3mean=np.mean(std[2])
    std_4mean=np.mean(std[3])
    std_5mean=np.mean(std[4])
    std_6mean=np.mean(std[5])
    std_7mean=np.mean(std[6])
    
    std_1std=np.std(std[0])
    std_2std=np.std(std[1])
    std_3std=np.std(std[2])
    std_4std=np.std(std[3])
    std_5std=np.std(std[4])
    std_6std=np.std(std[5])
    std_7std=np.std(std[6])
   
    std_d0 = norm(loc = std_1mean, scale=std_1std)
    std_d1 = norm(loc = std_2mean, scale=std_2std)
    std_d2 = norm(loc = std_3mean, scale=std_3std)
    std_d3 = norm(loc = std_4mean, scale=std_4std)
    std_d4 = norm(loc = std_5mean, scale=std_5std)
    std_d5 = norm(loc = std_6mean, scale=std_6std)
    std_d6 = norm(loc = std_7mean, scale=std_7std)

    x1 = np.arange(-.5, 3.5, 4.0/1000.0)
   
    #plot the pdfs of these normal distributions
    plt.figure(3, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, std_d0.pdf(x1),label='walk')
    plt.plot(x1 , std_d1.pdf(x1),label='run')
    plt.plot(x1, std_d2.pdf(x1),label='up')
    plt.plot(x1 , std_d3.pdf(x1),label='down')
    plt.plot(x1, std_d4.pdf(x1),label='static')
    plt.plot(x1 , std_d5.pdf(x1),label='cycle')
    plt.plot(x1 , std_d6.pdf(x1),label='car')    
    plt.title("std")
    plt.legend(loc="upper right") 
    plt.show()

    #FIGURA 4 
    K_1mean=np.mean(Kurtosis[0])
    K_2mean=np.mean(Kurtosis[1])
    K_3mean=np.mean(Kurtosis[2])
    K_4mean=np.mean(Kurtosis[3])
    K_5mean=np.mean(Kurtosis[4])
    K_6mean=np.mean(Kurtosis[5])
    K_7mean=np.mean(Kurtosis[6])
    
    K_1std=np.std(Kurtosis[0])
    K_2std=np.std(Kurtosis[1])
    K_3std=np.std(Kurtosis[2])
    K_4std=np.std(Kurtosis[3])
    K_5std=np.std(Kurtosis[4])
    K_6std=np.std(Kurtosis[5])
    K_7std=np.std(Kurtosis[6])
   
    K_d0 = norm(loc = K_1mean, scale=K_1std)
    K_d1 = norm(loc = K_2mean, scale=K_2std)
    K_d2 = norm(loc = K_3mean, scale=K_3std)
    K_d3 = norm(loc = K_4mean, scale=K_4std)
    K_d4 = norm(loc = K_5mean, scale=K_5std)
    K_d5 = norm(loc = K_6mean, scale=K_6std)
    K_d6 = norm(loc = K_7mean, scale=K_7std)

    x1 = np.arange(-5, 15, 20.0/1000.0)
   
    #plot the pdfs of these normal distributions
    plt.figure(4, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, K_d0.pdf(x1),label='walk')
    plt.plot(x1 , K_d1.pdf(x1),label='run')
    plt.plot(x1, K_d2.pdf(x1),label='up')
    plt.plot(x1 , K_d3.pdf(x1),label='down')
    plt.plot(x1, K_d4.pdf(x1),label='static')
    plt.plot(x1 , K_d5.pdf(x1),label='cycle')
    plt.plot(x1 , K_d6.pdf(x1),label='car')    
    plt.legend(loc="upper right") 
    plt.show()
    
 #FIGURA 5 
    Mx1mean=np.mean(Max[0])
    Mx2mean=np.mean(Max[1])
    Mx3mean=np.mean(Max[2])
    Mx4mean=np.mean(Max[3])
    Mx5mean=np.mean(Max[4])
    Mx6mean=np.mean(Max[5])
    Mx7mean=np.mean(Max[6])
    
    Mx1std=np.std(Max[0])
    Mx2std=np.std(Max[1])
    Mx3std=np.std(Max[2])
    Mx4std=np.std(Max[3])
    Mx5std=np.std(Max[4])
    Mx6std=np.std(Max[5])
    Mx7std=np.std(Max[6])
   
    Mxd0 = norm(loc = Mx1mean, scale=Mx1std)
    Mxd1 = norm(loc = Mx2mean, scale=Mx2std)
    Mxd2 = norm(loc = Mx3mean, scale=Mx3std)
    Mxd3 = norm(loc = Mx4mean, scale=Mx4std)
    Mxd4 = norm(loc = Mx5mean, scale=Mx5std)
    Mxd5 = norm(loc = Mx6mean, scale=Mx6std)
    Mxd6 = norm(loc = Mx7mean, scale=Mx7std)

    x1 = np.arange(-1, 18, 19.0/1000.0)
   
    #plot the pdfs of these normal distributions
    plt.figure(5, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, Mxd0.pdf(x1),label='walk')
    plt.plot(x1 , Mxd1.pdf(x1),label='run')
    plt.plot(x1, Mxd2.pdf(x1),label='up')
    plt.plot(x1 , Mxd3.pdf(x1),label='down')
    plt.plot(x1, Mxd4.pdf(x1),label='static')
    plt.plot(x1 , Mxd5.pdf(x1),label='cycle')
    plt.plot(x1 , Mxd6.pdf(x1),label='car')    
    plt.title("Max")
    plt.legend(loc="upper right") 
    plt.show()
    
    #FIGURA 6
    Min1mean=np.mean(Min[0])
    Min2mean=np.mean(Min[1])
    Min3mean=np.mean(Min[2])
    Min4mean=np.mean(Min[3])
    Min5mean=np.mean(Min[4])
    Min6mean=np.mean(Min[5])
    Min7mean=np.mean(Min[6])
    
    Min1std=np.std(Min[0])
    Min2std=np.std(Min[1])
    Min3std=np.std(Min[2])
    Min4std=np.std(Min[3])
    Min5std=np.std(Min[4])
    Min6std=np.std(Min[5])
    Min7std=np.std(Min[6])
   
    Mind0 = norm(loc = Min1mean, scale=Min1std)
    Mind1 = norm(loc = Min2mean, scale=Min2std)
    Mind2 = norm(loc = Min3mean, scale=Min3std)
    Mind3 = norm(loc = Min4mean, scale=Min4std)
    Mind4 = norm(loc = Min5mean, scale=Min5std)
    Mind5 = norm(loc = Min6mean, scale=Min6std)
    Mind6 = norm(loc = Min7mean, scale=Min7std)

    x1 = np.arange(-.5, 1.25, .75/100.0)
   
    #plot the pdfs of these normal distributions
    plt.figure(6, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, Mind0.pdf(x1),label='walk')
    plt.plot(x1 , Mind1.pdf(x1),label='run')
    plt.plot(x1, Mind2.pdf(x1),label='up')
    plt.plot(x1 , Mind3.pdf(x1),label='down')
    plt.plot(x1, Mind4.pdf(x1),label='static')
    plt.plot(x1 , Mind5.pdf(x1),label='cycle')
    plt.plot(x1 , Mind6.pdf(x1),label='car')    
    plt.title("Min")
    plt.legend(loc="upper right") 
    plt.show()
    
    #FIGURA 7
    rn1mean=np.mean(Range[0])
    rn2mean=np.mean(Range[1])
    rn3mean=np.mean(Range[2])
    rn4mean=np.mean(Range[3])
    rn5mean=np.mean(Range[4])
    rn6mean=np.mean(Range[5])
    rn7mean=np.mean(Range[6])
    
    rn1std=np.std(Range[0])
    rn2std=np.std(Range[1])
    rn3std=np.std(Range[2])
    rn4std=np.std(Range[3])
    rn5std=np.std(Range[4])
    rn6std=np.std(Range[5])
    rn7std=np.std(Range[6])
   
    rnd0 = norm(loc = rn1mean, scale=rn1std)
    rnd1 = norm(loc = rn2mean, scale=rn2std)
    rnd2 = norm(loc = rn3mean, scale=rn3std)
    rnd3 = norm(loc = rn4mean, scale=rn4std)
    rnd4 = norm(loc = rn5mean, scale=rn5std)
    rnd5 = norm(loc = rn6mean, scale=rn6std)
    rnd6 = norm(loc = rn7mean, scale=rn7std)

    x1 = np.arange(-1, 17, 18.0/1000.0)
   
    #plot the pdfs of these normal distributions
    plt.figure(7, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, rnd0.pdf(x1),label='walk')
    plt.plot(x1 , rnd1.pdf(x1),label='run')
    plt.plot(x1, rnd2.pdf(x1),label='up')
    plt.plot(x1 , rnd3.pdf(x1),label='down')
    plt.plot(x1, rnd4.pdf(x1),label='static')
    plt.plot(x1 , rnd5.pdf(x1),label='cycle')
    plt.plot(x1 , rnd6.pdf(x1),label='car')    
    plt.title("Range")
    plt.legend(loc="upper right") 
    plt.show()
    
    #FIGURA 8 
    mf1mean=np.mean(mFourier[0])
    mf2mean=np.mean(mFourier[1])
    mf3mean=np.mean(mFourier[2])
    mf4mean=np.mean(mFourier[3])
    mf5mean=np.mean(mFourier[4])
    mf6mean=np.mean(mFourier[5])
    mf7mean=np.mean(mFourier[6])
    
    mf1std=np.std(mFourier[0])
    mf2std=np.std(mFourier[1])
    mf3std=np.std(mFourier[2])
    mf4std=np.std(mFourier[3])
    mf5std=np.std(mFourier[4])
    mf6std=np.std(mFourier[5])
    mf7std=np.std(mFourier[6])
   
    mfd0 = norm(loc = mf1mean, scale=mf1std)
    mfd1 = norm(loc = mf2mean, scale=mf2std)
    mfd2 = norm(loc = mf3mean, scale=mf3std)
    mfd3 = norm(loc = mf4mean, scale=mf4std)
    mfd4 = norm(loc = mf5mean, scale=mf5std)
    mfd5 = norm(loc = mf6mean, scale=mf6std)
    mfd6 = norm(loc = mf7mean, scale=mf7std)

    x1 = np.arange(-1, 140, 141.0/1000.0)
   
    #plot the pdfs of these normal distributions
    plt.figure(8, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, mfd0.pdf(x1),label='walk')
    plt.plot(x1 , mfd1.pdf(x1),label='run')
    plt.plot(x1, mfd2.pdf(x1),label='up')
    plt.plot(x1 , mfd3.pdf(x1),label='down')
    plt.plot(x1, mfd4.pdf(x1),label='static')
    plt.plot(x1 , mfd5.pdf(x1),label='cycle')
    plt.plot(x1 , mfd6.pdf(x1),label='car')    
    plt.title("Fourier Magnitude")
    plt.legend(loc="upper right") 
    plt.show()
    
    #FIGURA 9 
    f1mean=np.mean(Freq[0])
    f2mean=np.mean(Freq[1])
    f3mean=np.mean(Freq[2])
    f4mean=np.mean(Freq[3])
    f5mean=np.mean(Freq[4])
    f6mean=np.mean(Freq[5])
    f7mean=np.mean(Freq[6])
    
    f1std=np.std(Freq[0])
    f2std=np.std(Freq[1])
    f3std=np.std(Freq[2])
    f4std=np.std(Freq[3])
    f5std=np.std(Freq[4])
    f6std=np.std(Freq[5])
    f7std=np.std(Freq[6])
   
    fd0 = norm(loc = f1mean, scale=f1std)
    fd1 = norm(loc = f2mean, scale=f2std)
    fd2 = norm(loc = f3mean, scale=f3std)
    fd3 = norm(loc = f4mean, scale=f4std)
    fd4 = norm(loc = f5mean, scale=f5std)
    fd5 = norm(loc = f6mean, scale=f6std)
    fd6 = norm(loc = f7mean, scale=f7std)

    x1 = np.arange(-1, 5.5, 6.5/1000.0)
   
    #plot the pdfs of these normal distributions
    plt.figure(9, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, fd0.pdf(x1),label='walk')
    plt.plot(x1 , fd1.pdf(x1),label='run')
    plt.plot(x1, fd2.pdf(x1),label='up')
    plt.plot(x1 , fd3.pdf(x1),label='down')
    plt.plot(x1, fd4.pdf(x1),label='static')
    plt.plot(x1 , fd5.pdf(x1),label='cycle')
    plt.plot(x1 , fd6.pdf(x1),label='car')    
    plt.title("Frequency Position")
    plt.legend(loc="upper right") 
    plt.show()
    
    #FIGURA 10 
    fm1mean=np.mean(freqMean[0])
    fm2mean=np.mean(freqMean[1])
    fm3mean=np.mean(freqMean[2])
    fm4mean=np.mean(freqMean[3])
    fm5mean=np.mean(freqMean[4])
    fm6mean=np.mean(freqMean[5])
    fm7mean=np.mean(freqMean[6])
    
    fm1std=np.std(freqMean[0])
    fm2std=np.std(freqMean[1])
    fm3std=np.std(freqMean[2])
    fm4std=np.std(freqMean[3])
    fm5std=np.std(freqMean[4])
    fm6std=np.std(freqMean[5])
    fm7std=np.std(freqMean[6])
   
    fmd0 = norm(loc = fm1mean, scale=fm1std)
    fmd1 = norm(loc = fm2mean, scale=fm2std)
    fmd2 = norm(loc = fm3mean, scale=fm3std)
    fmd3 = norm(loc = fm4mean, scale=fm4std)
    fmd4 = norm(loc = fm5mean, scale=fm5std)
    fmd5 = norm(loc = fm6mean, scale=fm6std)
    fmd6 = norm(loc = fm7mean, scale=fm7std)

    x1 = np.arange(-1, 18, 19.0/1000.0)
   
    #plot the pdfms of these normal distributions
    plt.figure(10, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, fmd0.pdf(x1),label='walk')
    plt.plot(x1 , fmd1.pdf(x1),label='run')
    plt.plot(x1, fmd2.pdf(x1),label='up')
    plt.plot(x1 , fmd3.pdf(x1),label='down')
    plt.plot(x1, fmd4.pdf(x1),label='static')
    plt.plot(x1 , fmd5.pdf(x1),label='cycle')
    plt.plot(x1 , fmd6.pdf(x1),label='car')    
    plt.title("freqMean")
    plt.legend(loc="upper right") 
    plt.show()
    
    #FIGURA 11 
    fs1mean=np.mean(Fstd[0])
    fs2mean=np.mean(Fstd[1])
    fs3mean=np.mean(Fstd[2])
    fs4mean=np.mean(Fstd[3])
    fs5mean=np.mean(Fstd[4])
    fs6mean=np.mean(Fstd[5])
    fs7mean=np.mean(Fstd[6])
    
    fs1std=np.std(Fstd[0])
    fs2std=np.std(Fstd[1])
    fs3std=np.std(Fstd[2])
    fs4std=np.std(Fstd[3])
    fs5std=np.std(Fstd[4])
    fs6std=np.std(Fstd[5])
    fs7std=np.std(Fstd[6])
   
    fsd0 = norm(loc = fs1mean, scale=fs1std)
    fsd1 = norm(loc = fs2mean, scale=fs2std)
    fsd2 = norm(loc = fs3mean, scale=fs3std)
    fsd3 = norm(loc = fs4mean, scale=fs4std)
    fsd4 = norm(loc = fs5mean, scale=fs5std)
    fsd5 = norm(loc = fs6mean, scale=fs6std)
    fsd6 = norm(loc = fs7mean, scale=fs7std)

    x1 = np.arange(-1, 22, 23.0/1000.0 )
   
    #plot the pdfs of these normal distributions
    plt.figure(11, figsize=(8, 6))
    plt.clf() 
    plt.plot(x1, fsd0.pdf(x1),label='walk')
    plt.plot(x1 , fsd1.pdf(x1),label='run')
    plt.plot(x1, fsd2.pdf(x1),label='up')
    plt.plot(x1 , fsd3.pdf(x1),label='down')
    plt.plot(x1, fsd4.pdf(x1),label='static')
    plt.plot(x1 , fsd5.pdf(x1),label='cycle')
    plt.plot(x1 , fsd6.pdf(x1),label='car')    
    plt.title("Fstd")
    plt.legend(loc="upper right") 
    plt.show()
    