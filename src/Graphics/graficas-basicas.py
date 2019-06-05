# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:43:24 2019

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
        signal = df
        tS = (max(df.aTime)-min(df.aTime))/len(signal)

    elif source == 0:
        signal = df
        tS = (max(df.sec) - min(df.sec))/(len(signal)*1.0)
        
	# fragment 3 secs
    N = round(5.0/tS)
    N = int(N)
    if N<len(signal):
        signal = signal[0:N]
    else:
        print ("lenght error, signal < 3 secs")
        
    return signal
if __name__ == '__main__':
    user = 'david'
    #CARGAR DATOS
    iwalk, mwalk, irun, mrun, iup, mup, idown, mdown, icar, mcar,isu,msu,icy,mcy = loadData(user,2)
    iwalk=valSignal(iwalk,1)    
    irun=valSignal(irun,1)
    iup=valSignal(iup,1)
    idown=valSignal(idown,1)
    isu=valSignal(isu,1)
    icy=valSignal(icy,1)
    icar=valSignal(icar,1)
    
    plt.figure(1)
    plt.clf()
    plt.plot(iwalk.AX, label='AX')
    plt.plot(iwalk.AY, label='AY')
    plt.plot(iwalk.AZ, label='AZ')
    plt.title("Walking")
    plt.legend(loc="upper right")
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()
    
    plt.figure(2)
    plt.clf()
    plt.plot(irun.AX, label='aX')
    plt.plot(irun.AY, label='aY')
    plt.plot(irun.AZ, label='aZ')
    plt.title("Running")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()
        
    plt.figure(3)
    plt.clf()
    plt.plot(iup.AX, label='aX')
    plt.plot(iup.AY, label='aY')
    plt.plot(iup.AZ, label='aZ')
    plt.title("Upstairs")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()

    plt.figure(4)
    plt.clf()
    plt.plot(idown.AX, label='aX')
    plt.plot(idown.AY, label='aY')
    plt.plot(idown.AZ, label='aZ')
    plt.title("Downstairs")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()
    
    plt.figure(5)
    plt.clf()
    plt.plot(isu.AX, label='aX')
    plt.plot(isu.AY, label='aY')
    plt.plot(isu.AZ, label='aZ')
    plt.title("Static")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()
    
    plt.figure(6)
    plt.clf()
    plt.plot(icy.AX, label='aX')
    plt.plot(icy.AY, label='aY')
    plt.plot(icy.AZ, label='aZ')
    plt.title("Cycle")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()    
    
    plt.figure(7)
    plt.clf()
    plt.plot(icar.AX, label='aX')
    plt.plot(icar.AY, label='aY')
    plt.plot(icar.AZ, label='aZ')
    plt.title("Car")
    plt.legend(loc="upper right")
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show() 
    
    print("-----------------------------MYO-------------------------")
    mwalk=valSignal(mwalk,0)    
    mrun=valSignal(mrun,0)
    mup=valSignal(mup,0)
    mdown=valSignal(mdown,0)
    msu=valSignal(msu,0)
    mcy=valSignal(mcy,0)
    mcar=valSignal(mcar,0)
    
    plt.figure(8)
    plt.clf()
    plt.plot(mwalk.aX, label='aX')
    plt.plot(mwalk.aY, label='aY')
    plt.plot(mwalk.aZ, label='aZ')
    plt.title("Walking")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()
    
    plt.figure(9)
    plt.clf()
    plt.plot(mrun.aX, label='aX')
    plt.plot(mrun.aY, label='aY')
    plt.plot(mrun.aZ, label='aZ')
    plt.title("Running")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()
        
    plt.figure(10)
    plt.clf()
    plt.plot(mup.aX, label='aX')
    plt.plot(mup.aY, label='aY')
    plt.plot(mup.aZ, label='aZ')
    plt.title("Upstairs")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()

    plt.figure(11)
    plt.clf()
    plt.plot(mdown.aX, label='aX')
    plt.plot(mdown.aY, label='aY')
    plt.plot(mdown.aZ, label='aZ')
    plt.title("Downstairs")
    plt.legend(loc="upper right")
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()
    
    plt.figure(12)
    plt.clf()
    plt.plot(msu.aX, label='aX')
    plt.plot(msu.aY, label='aY')
    plt.plot(msu.aZ, label='aZ')
    plt.title("Static")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()
    
    plt.figure(13)
    plt.clf()
    plt.plot(mcy.aX, label='aX')
    plt.plot(mcy.aY, label='aY')
    plt.plot(mcy.aZ, label='aZ')
    plt.title("Cycle")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()    
    
    plt.figure(14)
    plt.clf()
    plt.plot(mcar.aX, label='aX')
    plt.plot(mcar.aY, label='aY')
    plt.plot(mcar.aZ, label='aZ')
    plt.title("Car")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()
