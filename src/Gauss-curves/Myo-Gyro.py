# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 00:48:32 2019

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

def feature8(df,signal):
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
    f11=np.std(mf) #Desviacion estandar magnitud fourie

    return f8
    
def feature9(df,signal):
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
    f11=np.std(mf) #Desviacion estandar magnitud fourie

    return f9
    
def feature10(df,signal):
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
    f11=np.std(mf) #Desviacion estandar magnitud fourie
    
    return f10
    
def feature11(df,signal):
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
    f11=np.std(mf) #Desviacion estandar magnitud fourie

    return f11

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
    mg_walk,m_walk=valSignal(dmwalk,0)
    mg_run,m_run=valSignal(dmrun,0)
    mg_up,m_up=valSignal(dmup,0)
    mg_down,m_down=valSignal(dmdown,0)
    mg_su,m_su=valSignal(dmsu,0)
    mg_cy,m_cy=valSignal(dmcy,0)
    mg_car,m_car=valSignal(dmcar,0)
    
    user = 'eli'
    #CARGAR DATOS
    eiwalk, emwalk, eirun, emrun, eiup, emup, eidown, emdown, eicar, emcar,eisu,emsu,eicy,emcy = loadData(user,1)
    emg_walk,em_walk=valSignal(emwalk,0)
    emg_run,em_run=valSignal(emrun,0)
    emg_up,em_up=valSignal(emup,0)
    emg_down,em_down=valSignal(emdown,0)
    emg_su,em_su=valSignal(emsu,0)
    emg_cy,em_cy=valSignal(emcy,0)
    emg_car,em_car=valSignal(emcar,0)
#--------------------------------------------------------------------------------------------------------
#-----USER3----------------------------------------------------------------------------------------------
    user = 'guayara'
    #CARGAR DAgOS
    tiwalk, tmwalk, tirun, tmrun, tiup, tmup, tidown, tmdown, ticar, tmcar,tisu,tmsu,ticy,tmcy = loadData(user,0)
    tmg_walk,tm_walk=valSignal(tmwalk,0)
    tmg_run,tm_run=valSignal(tmrun,0)
    tmg_up,tm_up=valSignal(tmup,0)
    tmg_down,tm_down=valSignal(tmdown,0)
    tmg_su,tm_su=valSignal(tmsu,0)
    tmg_cy,tm_cy=valSignal(tmcy,0)
    tmg_car,tm_car=valSignal(tmcar,0)
    
#-----USER4-------------------------------------------------------------------------------------------
    user = 'janeth'
    #CARGAR DATOS
    jiwalk, jmwalk, jirun, jmrun, jiup, jmup, jidown, jmdown, jicar, jmcar,jisu,jmsu,jicy,jmcy = loadData(user,2)
    jmg_walk,jm_walk=valSignal(jmwalk,0)
    jmg_run,jm_run=valSignal(jmrun,0)
    jmg_up,jm_up=valSignal(jmup,0)
    jmg_down,jm_down=valSignal(jmdown,0)
    jmg_su,jm_su=valSignal(jmsu,0)
    jmg_cy,jm_cy=valSignal(jmcy,0)
    jmg_car,jm_car=valSignal(jmcar,0)
    
#-----USER5-------------------------------------------------------------------------------------------
    user = 'juan'
    #CARGAR DATOS
    jgiwalk, jgmwalk, jgirun, jgmrun, jgiup, jgmup, jgidown, jgmdown, jgicar, jgmcar,jgisu,jgmsu,jgicy,jgmcy = loadData(user,0)
    jgmg_walk,jgm_walk=valSignal(jgmwalk,0)
    jgmg_run,jgm_run=valSignal(jgmrun,0)
    jgmg_up,jgm_up=valSignal(jgmup,0)
    jgmg_down,jgm_down=valSignal(jgmdown,0)
    jgmg_su,jgm_su=valSignal(jgmsu,0)
    jgmg_cy,jgm_cy=valSignal(jgmcy,0)
    jgmg_car,jgm_car=valSignal(jgmcar,0)
    
#-----USER6-------------------------------------------------------------------------------------------
    user = 'juana'
    #CARGAR DATOS
    jtiwalk, jtmwalk, jtirun, jtmrun, jtiup, jtmup, jtidown, jtmdown, jticar, jtmcar,jtisu,jtmsu,jticy,jtmcy = loadData(user,1)
    jtmg_walk,jtm_walk=valSignal(jtmwalk,0)
    jtmg_run,jtm_run=valSignal(jtmrun,0)
    jtmg_up,jtm_up=valSignal(jtmup,0)
    jtmg_down,jtm_down=valSignal(jtmdown,0)
    jtmg_su,jtm_su=valSignal(jtmsu,0)
    jtmg_cy,jtm_cy=valSignal(jtmcy,0)
    jtmg_car,jtm_car=valSignal(jtmcar,0)
#-----USER7-------------------------------------------------------------------------------------------
    user = 'nicolas'
    #CARGAR DATOS
    niwalk, nmwalk, nirun, nmrun, niup, nmup, nidown, nmdown, nicar, nmcar,nisu,nmsu,nicy,nmcy = loadData(user,2)
    nmg_walk,nm_walk=valSignal(nmwalk,0)
    nmg_run,nm_run=valSignal(nmrun,0)
    nmg_up,nm_up=valSignal(nmup,0)
    nmg_down,nm_down=valSignal(nmdown,0)
    nmg_su,nm_su=valSignal(nmsu,0)
    nmg_cy,nm_cy=valSignal(nmcy,0)
    nmg_car,nm_car=valSignal(nmcar,0)
    
#-----USER8-------------------------------------------------------------------------------------------
    user = 'tutty'
    #CARGAR DATOS
    tuiwalk, tumwalk, tuirun, tumrun, tuiup, tumup, tuidown, tumdown, tuicar, tumcar,tuisu,tumsu,tuicy,tumcy = loadData(user,2)
    tumg_walk,tum_walk=valSignal(tumwalk,0)
    tumg_run,tum_run=valSignal(tumrun,0)
    tumg_up,tum_up=valSignal(tumup,0)
    tumg_down,tum_down=valSignal(tumdown,0)
    tumg_su,tum_su=valSignal(tumsu,0)
    tumg_cy,tum_cy=valSignal(tumcy,0)
    tumg_car,tum_car=valSignal(tumcar,0)
    
    #FEATURE 1:MAGNITUD
    mdw = np.sum(m_walk)
    mew = np.sum(em_walk)
    mtw = np.sum(tm_walk)
    mjw= np.sum(jm_walk)
    mjgw= np.sum(jgm_walk)
    mjtw=np.sum(jtm_walk)
    mnw=np.sum(nm_walk)
    mtuw=np.sum(tum_walk)
    
    mdr = np.sum(m_run)
    mer = np.sum(em_run)
    mtr = np.sum(tm_run)
    mjr= np.sum(jm_run)
    mjgr= np.sum(jgm_run)
    mjtr=np.sum(jtm_run)
    mnr=np.sum(nm_run)
    mtur=np.sum(tum_run)
    
    mdu = np.sum(m_up)
    meu = np.sum(em_up)
    mtu = np.sum(tm_up)
    mju= np.sum(jm_up)
    mjgu= np.sum(jgm_up)
    mjtu=np.sum(jtm_up)
    mnu=np.sum(nm_up)
    mtuu=np.sum(tum_up)
    
    mdd = np.sum(m_down)
    med = np.sum(em_down)
    mtd = np.sum(tm_down)
    mjd= np.sum(jm_down)
    mjgd= np.sum(jgm_down)
    mjtd=np.sum(jtm_down)
    mnd=np.sum(nm_down)
    mtud=np.sum(tum_down)
    
    mds = np.sum(m_su)
    mes = np.sum(em_su)
    mts = np.sum(tm_su)
    mjs= np.sum(jm_su)
    mjgs= np.sum(jgm_su)
    mjts=np.sum(jtm_su)
    mns=np.sum(nm_su)
    mtus=np.sum(tum_su)
    
    mdcy = np.sum(m_cy)
    mecy = np.sum(em_cy)
    mtcy = np.sum(tm_cy)
    mjcy= np.sum(jm_cy)
    mjgcy= np.sum(jgm_cy)
    mjtcy=np.sum(jtm_cy)
    mncy=np.sum(nm_cy)
    mtucy=np.sum(tum_cy)
    
    mdc = np.sum(m_car)
    mec = np.sum(em_car)
    mtc = np.sum(tm_car)
    mjc= np.sum(jm_car)
    mjgc= np.sum(jgm_car)
    mjtc=np.sum(jtm_car)
    mnc=np.sum(nm_car)
    mtuc=np.sum(tum_car)
    
    mC_1=[mdw,mew,mtw,mjw,mjgw,mjtw,mnw,mtuw]
    mC_2=[mdr,mer,mtr,mjr,mjgr,mjtr,mnr,mtur]
    mC_3=[mdu,meu,mtu,mju,mjgu,mjtu,mnu,mtuu]
    mC_4=[mdd,med,mtd,mjd,mjgd,mjtd,mnd,mtud]
    mC_5=[mds,mes,mts,mjs,mjgs,mjts,mns,mtus]
    mC_6=[mdcy,mecy,mtcy,mjcy,mjgcy,mjtcy,mncy,mtucy]
    mC_7=[mdc,mec,mtc,mjc,mjgc,mjtc,mnc,mtuc]
    
    Magnitud= [mC_1,mC_2,mC_3,mC_4,mC_5,mC_6,mC_7]
    
    #FEATURE 2:MEDIA
    mndw = np.mean(m_walk)
    mnew = np.mean(em_walk)
    mntw = np.mean(tm_walk)
    mnjw= np.mean(jm_walk)
    mnjgw= np.mean(jgm_walk)
    mnjtw=np.mean(jtm_walk)
    mnnw=np.mean(nm_walk)
    mntuw=np.mean(tum_walk)
    
    mndr = np.mean(m_run)
    mner = np.mean(em_run)
    mntr = np.mean(tm_run)
    mnjr= np.mean(jm_run)
    mnjgr= np.mean(jgm_run)
    mnjtr=np.mean(jtm_run)
    mnnr=np.mean(nm_run)
    mntur=np.mean(tum_run)
    
    mndu = np.mean(m_up)
    mneu = np.mean(em_up)
    mntu = np.mean(tm_up)
    mnju= np.mean(jm_up)
    mnjgu= np.mean(jgm_up)
    mnjtu=np.mean(jtm_up)
    mnnu=np.mean(nm_up)
    mntuu=np.mean(tum_up)
    
    mndd = np.mean(m_down)
    mned = np.mean(em_down)
    mntd = np.mean(tm_down)
    mnjd= np.mean(jm_down)
    mnjgd= np.mean(jgm_down)
    mnjtd=np.mean(jtm_down)
    mnnd=np.mean(nm_down)
    mntud=np.mean(tum_down)
    
    mnds = np.mean(m_su)
    mnes = np.mean(em_su)
    mnts = np.mean(tm_su)
    mnjs= np.mean(jm_su)
    mnjgs= np.mean(jgm_su)
    mnjts=np.mean(jtm_su)
    mnns=np.mean(nm_su)
    mntus=np.mean(tum_su)
    
    mndcy = np.mean(m_cy)
    mnecy = np.mean(em_cy)
    mntcy = np.mean(tm_cy)
    mnjcy= np.mean(jm_cy)
    mnjgcy= np.mean(jgm_cy)
    mnjtcy=np.mean(jtm_cy)
    mnncy=np.mean(nm_cy)
    mntucy=np.mean(tum_cy)
    
    mndc = np.mean(m_car)
    mnec = np.mean(em_car)
    mntc = np.mean(tm_car)
    mnjc= np.mean(jm_car)
    mnjgc= np.mean(jgm_car)
    mnjtc=np.mean(jtm_car)
    mnnc=np.mean(nm_car)
    mntuc=np.mean(tum_car)
    
    
    C_1=[mndw,mnew,mntw,mnjw,mnjgw,mnjtw,mnnw,mntuw]
    C_2=[mndr,mner,mntr,mnjr,mnjgr,mnjtr,mnnr,mntur]
    C_3=[mndu,mneu,mntu,mnju,mnjgu,mnjtu,mnnu,mntuu]
    C_4=[mndd,mned,mntd,mnjd,mnjgd,mnjtd,mnnd,mntud]
    C_5=[mnds,mnes,mnts,mnjs,mnjgs,mnjts,mnns,mntus]
    C_6=[mndcy,mnecy,mntcy,mnjcy,mnjgcy,mnjtcy,mnncy,mntucy]
    C_7=[mndc,mnec,mntc,mnjc,mnjgc,mnjtc,mnnc,mntuc]
    
    Media=[C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 3:STD
    stddw = np.std(m_walk)
    stdew = np.std(em_walk)
    stdtw = np.std(tm_walk)
    stdjw= np.std(jm_walk)
    stdjgw= np.std(jgm_walk)
    stdjtw=np.std(jtm_walk)
    stdnw=np.std(nm_walk)
    stdtuw=np.std(tum_walk)
    
    stddr = np.std(m_run)
    stder = np.std(em_run)
    stdtr = np.std(tm_run)
    stdjr= np.std(jm_run)
    stdjgr= np.std(jgm_run)
    stdjtr=np.std(jtm_run)
    stdnr=np.std(nm_run)
    stdtur=np.std(tum_run)
    
    stddu = np.std(m_up)
    stdeu = np.std(em_up)
    stdtu = np.std(tm_up)
    stdju= np.std(jm_up)
    stdjgu= np.std(jgm_up)
    stdjtu=np.std(jtm_up)
    stdnu=np.std(nm_up)
    stdtuu=np.std(tum_up)
    
    stddd = np.std(m_down)
    stded = np.std(em_down)
    stdtd = np.std(tm_down)
    stdjd= np.std(jm_down)
    stdjgd= np.std(jgm_down)
    stdjtd=np.std(jtm_down)
    stdnd=np.std(nm_down)
    stdtud=np.std(tum_down)
    
    stdds = np.std(m_su)
    stdes = np.std(em_su)
    stdts = np.std(tm_su)
    stdjs= np.std(jm_su)
    stdjgs= np.std(jgm_su)
    stdjts=np.std(jtm_su)
    stdns=np.std(nm_su)
    stdtus=np.std(tum_su)
    
    stddcy = np.std(m_cy)
    stdecy = np.std(em_cy)
    stdtcy = np.std(tm_cy)
    stdjcy= np.std(jm_cy)
    stdjgcy= np.std(jgm_cy)
    stdjtcy=np.std(jtm_cy)
    stdncy=np.std(nm_cy)
    stdtucy=np.std(tum_cy)
    
    stddc = np.std(m_car)
    stdec = np.std(em_car)
    stdtc = np.std(tm_car)
    stdjc= np.std(jm_car)
    stdjgc= np.std(jgm_car)
    stdjtc=np.std(jtm_car)
    stdnc=np.std(nm_car)
    stdtuc=np.std(tum_car)
    
    
    C_1=[stddw,stdew,stdtw,stdjw,stdjgw,stdjtw,stdnw,stdtuw]
    C_2=[stddr,stder,stdtr,stdjr,stdjgr,stdjtr,stdnr,stdtur]
    C_3=[stddu,stdeu,stdtu,stdju,stdjgu,stdjtu,stdnu,stdtuu]
    C_4=[stddd,stded,stdtd,stdjd,stdjgd,stdjtd,stdnd,stdtud]
    C_5=[stdds,stdes,stdts,stdjs,stdjgs,stdjts,stdns,stdtus]
    C_6=[stddcy,stdecy,stdtcy,stdjcy,stdjgcy,stdjtcy,stdncy,stdtucy]
    C_7=[stddc,stdec,stdtc,stdjc,stdjgc,stdjtc,stdnc,stdtuc]
    
    std=[C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 4:Kurtosm
    kdw = kurtosis(m_walk)
    kew = kurtosis(em_walk)
    ktw = kurtosis(tm_walk)
    kjw= kurtosis(jm_walk)
    kjgw= kurtosis(jgm_walk)
    kjtw=kurtosis(jtm_walk)
    knw=kurtosis(nm_walk)
    ktuw=kurtosis(tum_walk)
    
    kdr = kurtosis(m_run)
    ker = kurtosis(em_run)
    ktr = kurtosis(tm_run)
    kjr= kurtosis(jm_run)
    kjgr= kurtosis(jgm_run)
    kjtr=kurtosis(jtm_run)
    knr=kurtosis(nm_run)
    ktur=kurtosis(tum_run)
    
    kdu = kurtosis(m_up)
    keu = kurtosis(em_up)
    ktu = kurtosis(tm_up)
    kju= kurtosis(jm_up)
    kjgu= kurtosis(jgm_up)
    kjtu=kurtosis(jtm_up)
    knu=kurtosis(nm_up)
    ktuu=kurtosis(tum_up)
    
    kdd = kurtosis(m_down)
    ked = kurtosis(em_down)
    ktd = kurtosis(tm_down)
    kjd= kurtosis(jm_down)
    kjgd= kurtosis(jgm_down)
    kjtd=kurtosis(jtm_down)
    knd=kurtosis(nm_down)
    ktud=kurtosis(tum_down)
    
    kds = kurtosis(m_su)
    kes = kurtosis(em_su)
    kts = kurtosis(tm_su)
    kjs= kurtosis(jm_su)
    kjgs= kurtosis(jgm_su)
    kjts=kurtosis(jtm_su)
    kns=kurtosis(nm_su)
    ktus=kurtosis(tum_su)
    
    kdcy = kurtosis(m_cy)
    kecy = kurtosis(em_cy)
    ktcy = kurtosis(tm_cy)
    kjcy= kurtosis(jm_cy)
    kjgcy= kurtosis(jgm_cy)
    kjtcy=kurtosis(jtm_cy)
    kncy=kurtosis(nm_cy)
    ktucy=kurtosis(tum_cy)
    
    kdc = kurtosis(m_car)
    kec = kurtosis(em_car)
    ktc = kurtosis(tm_car)
    kjc= kurtosis(jm_car)
    kjgc= kurtosis(jgm_car)
    kjtc=kurtosis(jtm_car)
    knc=kurtosis(nm_car)
    ktuc=kurtosis(tum_car)
    
    
    C_1=[kdw,kew,ktw,kjw,kjgw,kjtw,knw,ktuw]
    C_2=[kdr,ker,ktr,kjr,kjgr,kjtr,knr,ktur]
    C_3=[kdu,keu,ktu,kju,kjgu,kjtu,knu,ktuu]
    C_4=[kdd,ked,ktd,kjd,kjgd,kjtd,knd,ktud]
    C_5=[kds,kes,kts,kjs,kjgs,kjts,kns,ktus]
    C_6=[kdcy,kecy,ktcy,kjcy,kjgcy,kjtcy,kncy,ktucy]
    C_7=[kdc,kec,ktc,kjc,kjgc,kjtc,knc,ktuc]
    
    Kurtosis= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 5:MAXIMO
    mxdw = np.max(m_walk)
    mxew = np.max(em_walk)
    mxtw = np.max(tm_walk)
    mxjw= np.max(jm_walk)
    mxjgw= np.max(jgm_walk)
    mxjtw=np.max(jtm_walk)
    mxnw=np.max(nm_walk)
    mxtuw=np.max(tum_walk)
    
    mxdr = np.max(m_run)
    mxer = np.max(em_run)
    mxtr = np.max(tm_run)
    mxjr= np.max(jm_run)
    mxjgr= np.max(jgm_run)
    mxjtr=np.max(jtm_run)
    mxnr=np.max(nm_run)
    mxtur=np.max(tum_run)
    
    mxdu = np.max(m_up)
    mxeu = np.max(em_up)
    mxtu = np.max(tm_up)
    mxju= np.max(jm_up)
    mxjgu= np.max(jgm_up)
    mxjtu=np.max(jtm_up)
    mxnu=np.max(nm_up)
    mxtuu=np.max(tum_up)
    
    mxdd = np.max(m_down)
    mxed = np.max(em_down)
    mxtd = np.max(tm_down)
    mxjd= np.max(jm_down)
    mxjgd= np.max(jgm_down)
    mxjtd=np.max(jtm_down)
    mxnd=np.max(nm_down)
    mxtud=np.max(tum_down)
    
    mxds = np.max(m_su)
    mxes = np.max(em_su)
    mxts = np.max(tm_su)
    mxjs= np.max(jm_su)
    mxjgs= np.max(jgm_su)
    mxjts=np.max(jtm_su)
    mxns=np.max(nm_su)
    mxtus=np.max(tum_su)
    
    mxdcy = np.max(m_cy)
    mxecy = np.max(em_cy)
    mxtcy = np.max(tm_cy)
    mxjcy= np.max(jm_cy)
    mxjgcy= np.max(jgm_cy)
    mxjtcy=np.max(jtm_cy)
    mxncy=np.max(nm_cy)
    mxtucy=np.max(tum_cy)
    
    mxdc = np.max(m_car)
    mxec = np.max(em_car)
    mxtc = np.max(tm_car)
    mxjc= np.max(jm_car)
    mxjgc= np.max(jgm_car)
    mxjtc=np.max(jtm_car)
    mxnc=np.max(nm_car)
    mxtuc=np.max(tum_car)
    
    
    C_1=[mxdw,mxew,mxtw,mxjw,mxjgw,mxjtw,mxnw,mxtuw]
    C_2=[mxdr,mxer,mxtr,mxjr,mxjgr,mxjtr,mxnr,mxtur]
    C_3=[mxdu,mxeu,mxtu,mxju,mxjgu,mxjtu,mxnu,mxtuu]
    C_4=[mxdd,mxed,mxtd,mxjd,mxjgd,mxjtd,mxnd,mxtud]
    C_5=[mxds,mxes,mxts,mxjs,mxjgs,mxjts,mxns,mxtus]
    C_6=[mxdcy,mxecy,mxtcy,mxjcy,mxjgcy,mxjtcy,mxncy,mxtucy]
    C_7=[mxdc,mxec,mxtc,mxjc,mxjgc,mxjtc,mxnc,mxtuc]
    
    Max= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 6:MINIMO
    mindw = np.min(m_walk)
    minew = np.min(em_walk)
    mintw = np.min(tm_walk)
    minjw= np.min(jm_walk)
    minjgw= np.min(jgm_walk)
    minjtw=np.min(jtm_walk)
    minnw=np.min(nm_walk)
    mintuw=np.min(tum_walk)
    
    mindr = np.min(m_run)
    miner = np.min(em_run)
    mintr = np.min(tm_run)
    minjr= np.min(jm_run)
    minjgr= np.min(jgm_run)
    minjtr=np.min(jtm_run)
    minnr=np.min(nm_run)
    mintur=np.min(tum_run)
    
    mindu = np.min(m_up)
    mineu = np.min(em_up)
    mintu = np.min(tm_up)
    minju= np.min(jm_up)
    minjgu= np.min(jgm_up)
    minjtu=np.min(jtm_up)
    minnu=np.min(nm_up)
    mintuu=np.min(tum_up)
    
    mindd = np.min(m_down)
    mined = np.min(em_down)
    mintd = np.min(tm_down)
    minjd= np.min(jm_down)
    minjgd= np.min(jgm_down)
    minjtd=np.min(jtm_down)
    minnd=np.min(nm_down)
    mintud=np.min(tum_down)
    
    minds = np.min(m_su)
    mines = np.min(em_su)
    mints = np.min(tm_su)
    minjs= np.min(jm_su)
    minjgs= np.min(jgm_su)
    minjts=np.min(jtm_su)
    minns=np.min(nm_su)
    mintus=np.min(tum_su)
    
    mindcy = np.min(m_cy)
    minecy = np.min(em_cy)
    mintcy = np.min(tm_cy)
    minjcy= np.min(jm_cy)
    minjgcy= np.min(jgm_cy)
    minjtcy=np.min(jtm_cy)
    minncy=np.min(nm_cy)
    mintucy=np.min(tum_cy)
    
    mindc = np.min(m_car)
    minec = np.min(em_car)
    mintc = np.min(tm_car)
    minjc= np.min(jm_car)
    minjgc= np.min(jgm_car)
    minjtc=np.min(jtm_car)
    minnc=np.min(nm_car)
    mintuc=np.min(tum_car)
    
    
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
    
     #FEATURE 8: MXFOURmER
    mfdw = feature8(dmwalk,m_walk)
    mfew = feature8(emwalk,em_walk)
    mftw = feature8(tmwalk,tm_walk)
    mfjw=  feature8(jmwalk,jm_walk)
    mfjgw = feature8(jgmwalk,jgm_walk)
    mfjtw = feature8(jtmwalk,jtm_walk)
    mfnw = feature8(nmwalk,nm_walk)
    mftuw=  feature8(tumwalk,tum_walk)
    
    mfdr = feature8(dmrun,m_run)
    mfer = feature8(emrun,em_run)
    mftr = feature8(tmrun,tm_run)
    mfjr=  feature8(jmrun,jm_run)
    mfjgr = feature8(jgmrun,jgm_run)
    mfjtr = feature8(jtmrun,jtm_run)
    mfnr = feature8(nmrun,nm_run)
    mftur=  feature8(tumrun,tum_run)
    
    mfdu = feature8(dmup,m_up)
    mfeu = feature8(emup,em_up)
    mftu = feature8(tmup,tm_up)
    mfju=  feature8(jmup,jm_up)
    mfjgu = feature8(jgmup,jgm_up)
    mfjtu = feature8(jtmup,jtm_up)
    mfnu = feature8(nmup,nm_up)
    mftuu=  feature8(tumup,tum_up)
    
    mfdd = feature8(dmdown,m_down)
    mfed = feature8(emdown,em_down)
    mftd = feature8(tmdown,tm_down)
    mfjd=  feature8(jmdown,jm_down)
    mfjgd = feature8(jgmdown,jgm_down)
    mfjtd = feature8(jtmdown,jtm_down)
    mfnd = feature8(nmdown,nm_down)
    mftud=  feature8(tumdown,tum_down)
    
    mfds = feature8(dmsu,m_su)
    mfes = feature8(emsu,em_su)
    mfts = feature8(tmsu,tm_su)
    mfjs=  feature8(jmsu,jm_su)
    mfjgs = feature8(jgmsu,jgm_su)
    mfjts = feature8(jtmsu,jtm_su)
    mfns = feature8(nmsu,nm_su)
    mftus=  feature8(tumsu,tum_su)
    
    mfdcy = feature8(dmcy,m_cy)
    mfecy = feature8(emcy,em_cy)
    mftcy = feature8(tmcy,tm_cy)
    mfjcy=  feature8(jmcy,jm_cy)
    mfjgcy = feature8(jgmcy,jgm_cy)
    mfjtcy = feature8(jtmcy,jtm_cy)
    mfncy = feature8(nmcy,nm_cy)
    mftucy=  feature8(tumcy,tum_cy)
    
    mfdc = feature8(dmcar,m_car)
    mfec = feature8(emcar,em_car)
    mftc = feature8(tmcar,tm_car)
    mfjc=  feature8(jmcar,jm_car)
    mfjgc = feature8(jgmcar,jgm_car)
    mfjtc = feature8(jtmcar,jtm_car)
    mfnc = feature8(nmcar,nm_car)
    mftuc=  feature8(tumcar,tum_car)
    
    C_1=[mfdw,mfew,mftw,mfjw,mfjgw,mfjtw,mfnw,mftuw]
    C_2=[mfdr,mfer,mftr,mfjr,mfjgr,mfjtr,mfnr,mftur]
    C_3=[mfdu,mfeu,mftu,mfju,mfjgu,mfjtu,mfnu,mftuu]
    C_4=[mfdd,mfed,mftd,mfjd,mfjgd,mfjtd,mfnd,mftud]
    C_5=[mfds,mfes,mfts,mfjs,mfjgs,mfjts,mfns,mftus]
    C_6=[mfdcy,mfecy,mftcy,mfjcy,mfjgcy,mfjtcy,mfncy,mftucy]
    C_7=[mfdc,mfec,mftc,mfjc,mfjgc,mfjtc,mfnc,mftuc]
    
    mFourier= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 9 : Freq
    fdw = feature9(dmwalk,m_walk)
    few = feature9(emwalk,em_walk)
    ftw = feature9(tmwalk,tm_walk)
    fjw=  feature9(jmwalk,jm_walk)
    fjgw = feature9(jgmwalk,jgm_walk)
    fjtw = feature9(jtmwalk,jtm_walk)
    fnw = feature9(nmwalk,nm_walk)
    ftuw=  feature9(tumwalk,tum_walk)
    
    fdr = feature9(dmrun,m_run)
    fer = feature9(emrun,em_run)
    ftr = feature9(tmrun,tm_run)
    fjr=  feature9(jmrun,jm_run)
    fjgr = feature9(jgmrun,jgm_run)
    fjtr = feature9(jtmrun,jtm_run)
    fnr = feature9(nmrun,nm_run)
    ftur=  feature9(tumrun,tum_run)
    
    fdu = feature9(dmup,m_up)
    feu = feature9(emup,em_up)
    ftu = feature9(tmup,tm_up)
    fju=  feature9(jmup,jm_up)
    fjgu = feature9(jgmup,jgm_up)
    fjtu = feature9(jtmup,jtm_up)
    fnu = feature9(nmup,nm_up)
    ftuu=  feature9(tumup,tum_up)
    
    fdd = feature9(dmdown,m_down)
    fed = feature9(emdown,em_down)
    ftd = feature9(tmdown,tm_down)
    fjd=  feature9(jmdown,jm_down)
    fjgd = feature9(jgmdown,jgm_down)
    fjtd = feature9(jtmdown,jtm_down)
    fnd = feature9(nmdown,nm_down)
    ftud=  feature9(tumdown,tum_down)
    
    fds = feature9(dmsu,m_su)
    fes = feature9(emsu,em_su)
    fts = feature9(tmsu,tm_su)
    fjs=  feature9(jmsu,jm_su)
    fjgs = feature9(jgmsu,jgm_su)
    fjts = feature9(jtmsu,jtm_su)
    fns = feature9(nmsu,nm_su)
    ftus=  feature9(tumsu,tum_su)
    
    fdcy = feature9(dmcy,m_cy)
    fecy = feature9(emcy,em_cy)
    ftcy = feature9(tmcy,tm_cy)
    fjcy=  feature9(jmcy,jm_cy)
    fjgcy = feature9(jgmcy,jgm_cy)
    fjtcy = feature9(jtmcy,jtm_cy)
    fncy = feature9(nmcy,nm_cy)
    ftucy=  feature9(tumcy,tum_cy)
    
    fdc = feature9(dmcar,m_car)
    fec = feature9(emcar,em_car)
    ftc = feature9(tmcar,tm_car)
    fjc=  feature9(jmcar,jm_car)
    fjgc = feature9(jgmcar,jgm_car)
    fjtc = feature9(jtmcar,jtm_car)
    fnc = feature9(nmcar,nm_car)
    ftuc=  feature9(tumcar,tum_car)
    
    C_1=[fdw,few,ftw,fjw,fjgw,fjtw,fnw,ftuw]
    C_2=[fdr,fer,ftr,fjr,fjgr,fjtr,fnr,ftur]
    C_3=[fdu,feu,ftu,fju,fjgu,fjtu,fnu,ftuu]
    C_4=[fdd,fed,ftd,fjd,fjgd,fjtd,fnd,ftud]
    C_5=[fds,fes,fts,fjs,fjgs,fjts,fns,ftus]
    C_6=[fdcy,fecy,ftcy,fjcy,fjgcy,fjtcy,fncy,ftucy]
    C_7=[fdc,fec,ftc,fjc,fjgc,fjtc,fnc,ftuc]
    
    Freq= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 10 : MedmaFreq
    fmdw = feature10(dmwalk,m_walk)
    fmew = feature10(emwalk,em_walk)
    fmtw = feature10(tmwalk,tm_walk)
    fmjw=  feature10(jmwalk,jm_walk)
    fmjgw = feature10(jgmwalk,jgm_walk)
    fmjtw = feature10(jtmwalk,jtm_walk)
    fmnw = feature10(nmwalk,nm_walk)
    fmtuw=  feature10(tumwalk,tum_walk)
    
    fmdr = feature10(dmrun,m_run)
    fmer = feature10(emrun,em_run)
    fmtr = feature10(tmrun,tm_run)
    fmjr=  feature10(jmrun,jm_run)
    fmjgr = feature10(jgmrun,jgm_run)
    fmjtr = feature10(jtmrun,jtm_run)
    fmnr = feature10(nmrun,nm_run)
    fmtur=  feature10(tumrun,tum_run)
    
    fmdu = feature10(dmup,m_up)
    fmeu = feature10(emup,em_up)
    fmtu = feature10(tmup,tm_up)
    fmju=  feature10(jmup,jm_up)
    fmjgu = feature10(jgmup,jgm_up)
    fmjtu = feature10(jtmup,jtm_up)
    fmnu = feature10(nmup,nm_up)
    fmtuu=  feature10(tumup,tum_up)
    
    fmdd = feature10(dmdown,m_down)
    fmed = feature10(emdown,em_down)
    fmtd = feature10(tmdown,tm_down)
    fmjd=  feature10(jmdown,jm_down)
    fmjgd = feature10(jgmdown,jgm_down)
    fmjtd = feature10(jtmdown,jtm_down)
    fmnd = feature10(nmdown,nm_down)
    fmtud=  feature10(tumdown,tum_down)
    
    fmds = feature10(dmsu,m_su)
    fmes = feature10(emsu,em_su)
    fmts = feature10(tmsu,tm_su)
    fmjs=  feature10(jmsu,jm_su)
    fmjgs = feature10(jgmsu,jgm_su)
    fmjts = feature10(jtmsu,jtm_su)
    fmns = feature10(nmsu,nm_su)
    fmtus=  feature10(tumsu,tum_su)
    
    fmdcy = feature10(dmcy,m_cy)
    fmecy = feature10(emcy,em_cy)
    fmtcy = feature10(tmcy,tm_cy)
    fmjcy=  feature10(jmcy,jm_cy)
    fmjgcy = feature10(jgmcy,jgm_cy)
    fmjtcy = feature10(jtmcy,jtm_cy)
    fmncy = feature10(nmcy,nm_cy)
    fmtucy=  feature10(tumcy,tum_cy)
    
    fmdc = feature10(dmcar,m_car)
    fmec = feature10(emcar,em_car)
    fmtc = feature10(tmcar,tm_car)
    fmjc=  feature10(jmcar,jm_car)
    fmjgc = feature10(jgmcar,jgm_car)
    fmjtc = feature10(jtmcar,jtm_car)
    fmnc = feature10(nmcar,nm_car)
    fmtuc=  feature10(tumcar,tum_car)
    
    C_1=[fmdw,fmew,fmtw,fmjw,fmjgw,fmjtw,fmnw,fmtuw]
    C_2=[fmdr,fmer,fmtr,fmjr,fmjgr,fmjtr,fmnr,fmtur]
    C_3=[fmdu,fmeu,fmtu,fmju,fmjgu,fmjtu,fmnu,fmtuu]
    C_4=[fmdd,fmed,fmtd,fmjd,fmjgd,fmjtd,fmnd,fmtud]
    C_5=[fmds,fmes,fmts,fmjs,fmjgs,fmjts,fmns,fmtus]
    C_6=[fmdcy,fmecy,fmtcy,fmjcy,fmjgcy,fmjtcy,fmncy,fmtucy]
    C_7=[fmdc,fmec,fmtc,fmjc,fmjgc,fmjtc,fmnc,fmtuc]
    freqMean= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FEATURE 11 : FreqSTD
    fstddw = feature11(dmwalk,m_walk)
    fstdew = feature11(emwalk,em_walk)
    fstdtw = feature11(tmwalk,tm_walk)
    fstdjw=  feature11(jmwalk,jm_walk)
    fstdjgw = feature11(jgmwalk,jgm_walk)
    fstdjtw = feature11(jtmwalk,jtm_walk)
    fstdnw = feature11(nmwalk,nm_walk)
    fstdtuw=  feature11(tumwalk,tum_walk)
    
    fstddr = feature11(dmrun,m_run)
    fstder = feature11(emrun,em_run)
    fstdtr = feature11(tmrun,tm_run)
    fstdjr=  feature11(jmrun,jm_run)
    fstdjgr = feature11(jgmrun,jgm_run)
    fstdjtr = feature11(jtmrun,jtm_run)
    fstdnr = feature11(nmrun,nm_run)
    fstdtur=  feature11(tumrun,tum_run)
    
    fstddu = feature11(dmup,m_up)
    fstdeu = feature11(emup,em_up)
    fstdtu = feature11(tmup,tm_up)
    fstdju=  feature11(jmup,jm_up)
    fstdjgu = feature11(jgmup,jgm_up)
    fstdjtu = feature11(jtmup,jtm_up)
    fstdnu = feature11(nmup,nm_up)
    fstdtuu=  feature11(tumup,tum_up)
    
    fstddd = feature11(dmdown,m_down)
    fstded = feature11(emdown,em_down)
    fstdtd = feature11(tmdown,tm_down)
    fstdjd=  feature11(jmdown,jm_down)
    fstdjgd = feature11(jgmdown,jgm_down)
    fstdjtd = feature11(jtmdown,jtm_down)
    fstdnd = feature11(nmdown,nm_down)
    fstdtud=  feature11(tumdown,tum_down)
    
    fstdds = feature11(dmsu,m_su)
    fstdes = feature11(emsu,em_su)
    fstdts = feature11(tmsu,tm_su)
    fstdjs=  feature11(jmsu,jm_su)
    fstdjgs = feature11(jgmsu,jgm_su)
    fstdjts = feature11(jtmsu,jtm_su)
    fstdns = feature11(nmsu,nm_su)
    fstdtus=  feature11(tumsu,tum_su)
    
    fstddcy = feature11(dmcy,m_cy)
    fstdecy = feature11(emcy,em_cy)
    fstdtcy = feature11(tmcy,tm_cy)
    fstdjcy=  feature11(jmcy,jm_cy)
    fstdjgcy = feature11(jgmcy,jgm_cy)
    fstdjtcy = feature11(jtmcy,jtm_cy)
    fstdncy = feature11(nmcy,nm_cy)
    fstdtucy=  feature11(tumcy,tum_cy)
    
    fstddc = feature11(dmcar,m_car)
    fstdec = feature11(emcar,em_car)
    fstdtc = feature11(tmcar,tm_car)
    fstdjc=  feature11(jmcar,jm_car)
    fstdjgc = feature11(jgmcar,jgm_car)
    fstdjtc = feature11(jtmcar,jtm_car)
    fstdnc = feature11(nmcar,nm_car)
    fstdtuc=  feature11(tumcar,tum_car)
    
    C_1=[fstddw,fstdew,fstdtw,fstdjw,fstdjgw,fstdjtw,fstdnw,fstdtuw]
    C_2=[fstddr,fstder,fstdtr,fstdjr,fstdjgr,fstdjtr,fstdnr,fstdtur]
    C_3=[fstddu,fstdeu,fstdtu,fstdju,fstdjgu,fstdjtu,fstdnu,fstdtuu]
    C_4=[fstddd,fstded,fstdtd,fstdjd,fstdjgd,fstdjtd,fstdnd,fstdtud]
    C_5=[fstdds,fstdes,fstdts,fstdjs,fstdjgs,fstdjts,fstdns,fstdtus]
    C_6=[fstddcy,fstdecy,fstdtcy,fstdjcy,fstdjgcy,fstdjtcy,fstdncy,fstdtucy]
    C_7=[fstddc,fstdec,fstdtc,fstdjc,fstdjgc,fstdjtc,fstdnc,fstdtuc]
    
    Fstd= [C_1,C_2,C_3,C_4,C_5,C_6,C_7]
    
    #FIGURA1
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

    x1 = np.arange(0, 300, 300.0/1000)
   
    #plot the pdfs of these normal distributmons
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
    plt.legend(loc="upper right") 
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

    x1 = np.arange(0, 1.5, 1.5/1000)
   
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
    plt.title("Media")
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

    x1 = np.arange(-.05, .15, 0.1/1000)
   
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

    x1 = np.arange(-4, 5, 9.0/1000)
   
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
    plt.title("Kurtosis")
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

    x1 = np.arange(0, 1.5, 1.5/1000)
   
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

    x1 = np.arange(0, 1.5, 1.5/1000)
   
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

    x1 = np.arange(-.1, .5, 0.06/1000)
   
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

    x1 = np.arange(-1, 5, 6.0/1000)
   
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
    plt.title("mFourier")
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

    x1 = np.arange(-1, 2, 3.0/1000)
   
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
    plt.title("Freq")
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

    x1 = np.arange(-.1, 0.5, 0.6/1000)
   
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

    x1 = np.arange(-.2, .8, 1.0/1000)
   
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
