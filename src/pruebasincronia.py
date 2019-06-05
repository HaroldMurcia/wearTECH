# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:28:56 2019

@author: juanita
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import kurtosis
from numpy.fft import fft, fftfreq
import os

main_path = os.path.abspath(__file__+"/../..")+"/"
def loadData(user):
    ## CAMINANDO
    # iPhone Header
    inames=['loggingTime',    'loggingSample',    'identifierForVendor',    'deviceID',    'locationTimestamp_since1970',    'locationLatitude',    'locationLongitude',    'locationAltitude',    'lSpeed',    'locationCourse',    'locationVerticalAccuracy',    'locationHorizontalAccuracy',    'locationFloor',    'locationHeadingTimestamp_since1970',    'locationHeadingX',    'locationHeadingY',    'locationHeadingZ',    'locationTrueHeading',    'locationMagneticHeading',    'locationHeadingAccuracy',    'aTime',    'AX',    'AY',    'AZ',    'gTime',    'gX',    'gY',    'gZ',    'magnetometerTimestamp_sinceReboot',    'magnetometerX',    'magnetometerY',    'magnetometerZ',    'motionTimestamp_sinceReboot',    'motionYaw',    'motionRoll',    'motionPitch',    'motionRotationRateX',    'motionRotationRateY',    'motionRotationRateZ',    'motionUserAccelerationX',    'motionUserAccelerationY',    'motionUserAccelerationZ',    'motionAttitudeReferenceFrame',    'motionQuaternionX',    'motionQuaternionY',    'motionQuaternionZ',    'motionQuaternionW',    'motionGravityX',    'motionGravityY',    'motionGravityZ',    'motionMagneticFieldX',    'motionMagneticFieldY',    'motionMagneticFieldZ',    'motionMagneticFieldCalibrationAccuracy',    'activityTimestamp_sinceReboot',    'activity',    'activityActivityConfidence',    'activityActivityStartDate',    'altimeterTimestamp_sinceReboot',    'altimeterReset',    'altimeter',    'altimeterPressure',    'IP_en0',    'IP_pdp_ip0',    'deviceOrientation',    'batteryState',    'batteryLevel',    'avAudioRecorderPeakPower',    'avAudioRecorderAveragePower',    'label']
    # iPhone dataFrame
    file = main_path+'data/iPhone/i_'+user+'_rut.csv'
    irut = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
    # myo dataFrame
    file = main_path+'data/Myo/m_'+user+'_rut.txt'
    mrut= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
    return irut,mrut
def magSignal(df,source):
    # source = 1 for iPhone
    # source = 0 for Myo
    if source == 1:
        signal = np.sqrt(pow(df.AX,2)+pow(df.AY,2)+pow(df.AZ,2))
        tS = (max(df.aTime)-min(df.aTime))/len(signal)
        #Gyro
        signalg =np.sqrt(pow(df.gX,2)+pow(df.gY,2)+pow(df.gZ,2))

    elif source == 0:
        signal = np.sqrt(pow(df.aX,2)+pow(df.aY,2)+pow(df.aZ,2))
        #tS = (max(df.sec) - min(df.sec))/(len(signal)*1.0)
        #Gyro
        signalg =np.sqrt(pow(df.oX,2)+pow(df.oY,2)+pow(df.oZ,2))
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
        #tS = (max(df.sec) - min(df.sec))/(len(signal)*1.0)
        Fs=60.0
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

def gFourier(df,signalg,source):
    # source = 1 for iPhone
    # source = 0 for Myo
    if source == 1:
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

        F_v=np.array([f_8,f_9,f_10,f_11])

    elif source == 0:
        signalg=signalg-np.mean(signalg)
        gfourier= np.fft.fft(signalg)
        gmf=np.abs(gfourier/2)
        Lg=(len(gmf)/2)
        gmf=gmf[0:Lg]
        #t_S = (max(df.sec) - min(df.sec))/(len(signalg)*1.0)
        F_s=60.0
        gxf=np.linspace(0, F_s/2.0,  Lg)
        f_8=max(gmf)
        gpos= np.where(gmf == f_8)
        f_9=gxf[gpos] #Frecuencia donde se encuentra el pico mas grande
        f_10= np.mean(gmf) #Media Magnitud fourier
        f_11=np.std(gmf) #Desviacion estandar magnitud fourier

        F_v=np.array([f_8,f_9,f_10,f_11])
    return F_v

def lSpeed(df):
    sp=df.lSpeed
    sa=np.abs(sp)
    f12=np.mean(sa)
    return f12

def sigAltimeter(df):
    alt=df.altimeter
    #Derivada
    delta=alt.diff()
    f13=np.mean(delta)
    return f13

def val_Scores(Y,pred_1,pred_2,pred_3,pred_4,pred_5):
    f1_RF   = f1_score(Y, pred_1, average=None)
    f1_SVM  = f1_score(Y, pred_2, average=None)
    f1_NN   = f1_score(Y, pred_3, average=None)
    f1_bdt  = f1_score(Y, pred_4, average=None)
    f1_GAUSS= f1_score(Y, pred_5, average=None)
    F1_scores=np.matrix([f1_RF, f1_SVM, f1_NN, f1_bdt, f1_GAUSS])
    return F1_scores

if __name__ == '__main__':
#----USER1------------------------------------------------------------------
    user = 'juanita'
    #CARGAR DATOS
    irut,mrut = loadData(user)
    is_rut,ig_rut=magSignal(irut,1)
    m_rut, mg_rut=magSignal(mrut,0)
    tf=256
    ti=np.linspace(0,tf,25016)
    tm=np.linspace(0,tf,12506)
    plt.plot(ti, is_rut, label='iPhone')
    plt.plot(tm,m_rut,label='Myo')
    plt.title("Rutina")
    plt.legend(loc="upper right") 
    plt.xlabel('Time')
    plt.ylabel('accelerometer')
    plt.show()    
    
