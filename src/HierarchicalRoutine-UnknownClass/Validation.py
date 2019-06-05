import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import kurtosis
from numpy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        #UNKNOWN
        file = main_path+'data/iPhone/unknown/i_'+user+'_unk_0.csv'
        iunk = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/unknown/m_'+user+'_unk_0.txt'
        munk= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
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
        #UNKNOWN
        file = main_path+'data/iPhone/unknown/i_'+user+'_unk_1.csv'
        iunk = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/unknown/m_'+user+'_unk_1.txt'
        munk= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
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
        #UNKNOWN
        file = main_path+'data/iPhone/unknown/i_'+user+'_unk_2.csv'
        iunk = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/unknown/m_'+user+'_unk_2.txt'
        munk= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
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
        #UNKNOWN
        file = main_path+'data/iPhone/unknown/i_'+user+'_unk_3.csv'
        iunk = pd.read_table(file, engine='python', sep=';', header=None, names=inames)
        file = main_path+'data/Myo/unknown/m_'+user+'_unk_3.txt'
        munk= pd.read_csv(file, sep="\t", header=None, names=['year','month','day','hour','min','sec','oX','oY','oZ','aX','aY','aZ','vX','vY','vZ'])
    return iwalk, mwalk, irun, mrun, iup, mup, idown, mdown, icar, mcar,isu,msu,icy,mcy,iunk,munk

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

    N = round(4.0/tS)
    N = int(N)

    if N<len(signal):
        signal = signal[0:N]
    else:
        print ("lenght error, signal < 4 secs")
    #Gyro
    if N<len(signalg):
        signalg = signalg[0:N]
    else:
        print ("lenght error, signalg < 4 secs")

    return signal,signalg

def sigFeatures(signal,source):
    # source = 1 for iPhone
    # source = 0 for Myo
    if source == 1:
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
        #range
        f7=f5-f6

        fV = np.array([f2,f4,f6])

    elif source == 0:
        # sum
        f1 = np.sum(signal)
        # mean
        f2 = np.mean(signal)
        #std
        f3= np.std(signal)
        # kurtosis
        f4 = kurtosis(signal)
        #valor maximo
        f5= max(signal)
        #valor minimo
        f6= min(signal)
        #rango
        f7=f5-f6

        fV = np.array([f1,f4,f6])

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
        f11=np.std(mf)

        Fv=np.array([f8,f9,f10])

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
        f10=np.mean(mf)
        f11=np.std(mf) #Desviacion estandar magnitud fourier

        Fv=np.array([f10,f11])
    return Fv

def gFeatures(signalg,source):
    if source == 1:
        # sum
        f_1 = np.sum(signalg)
        # mean
        f_2 = np.mean(signalg)
        # std
        f_3 = np.std(signalg)
        #kurtosis
        f_4=kurtosis(signalg)
        #valor maximo
        f_5= max(signalg)
        #valor minimo
        f_6= min(signalg)
        #rango
        f_7=f_5-f_6

        f_V = np.array([f_4,f_6,f_7])

    elif source == 0:
        # sum
        f_1 = np.sum(signalg)
        # mean
        f_2 = np.mean(signalg)
        # std
        f_3 = np.std(signalg)
        #kurtosis
        f_4=kurtosis(signalg)
        #valor maximo
        f_5= max(signalg)
        #valor minimo
        f_6= min(signalg)
        #rango
        f_7=f_5-f_6

        f_V = np.array([f_1,f_4,f_6,f_7])

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
        f_10= np.mean(gmf)
        f_11=np.std(gmf)

        F_v=np.array([f_8,f_9])

    elif source == 0:
        signalg=signalg-np.mean(signalg)
        gfourier= np.fft.fft(signalg)
        gmf=np.abs(gfourier/2)
        Lg=(len(gmf)/2)
        gmf=gmf[0:Lg]
        t_S = (max(df.sec) - min(df.sec))/(len(signalg)*1.0)
        F_s=1.0/t_S
        gxf=np.linspace(0, F_s/2.0,  Lg)
        f_8=max(gmf)
        gpos= np.where(gmf == f_8)
        f_9=gxf[gpos] #Frecuencia donde se encuentra el pico mas grande
        f_10= np.mean(gmf) #Media Magnitud fourier
        f_11=np.std(gmf)

        F_v=np.array([f_9,f_10])
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
    f14=np.std(delta)
    Fa=np.array([f13,f14])
    return Fa

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
    user = 'david'
    #CARGAR DATOS
    iwalk, mwalk, irun, mrun, iup, mup, idown, mdown, icar, mcar,isu,msu,icy,mcy,iunk,munk = loadData(user,2)
#########IPHONE########################################################################
    #SUBIENDO ESCALERAS
    is_up,ig_up=valSignal(iup,1)
    feat = sigFeatures(is_up,1)
    ifourier=tFourier(iup,is_up,1)
    feat1= lSpeed(iup)
    feat2= sigAltimeter(iup)
    #GYRO
    i_feat= gFeatures(ig_up,1)
    i_fourier=gFourier(iup,ig_up,1)
    gfeat1= lSpeed(iup)
    gfeat2= sigAltimeter(iup)
    #########MYO###########################################################
    #SUBIENDO ESCALERAS
    m_up,mg_up = valSignal(mup,0)
    mfeat = sigFeatures(m_up,0)
    mfourier=tFourier(mup,m_up,0)
    #GYRO
    m_feat= gFeatures(mg_up,0)
    m_fourier=gFourier(mup,mg_up,0)

    ifeat1 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat1 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
    #----------------------------------------------------------------------------------------------------------------------------------------
    #UNKNOWN
    is_unk,ig_unk=valSignal(iunk,1)
    feat = sigFeatures(is_unk,1)
    ifourier=tFourier(iunk,is_unk,1)
    feat1= lSpeed(iunk)
    feat2= sigAltimeter(iunk)
    #GYRO
    i_feat= gFeatures(ig_unk,1)
    i_fourier=gFourier(iunk,ig_unk,1)
    #########MYO###########################################################
    #UNKNOWN
    m_unk,mg_unk = valSignal(munk,0)
    mfeat = sigFeatures(m_unk,0)
    mfourier=tFourier(munk,m_unk,0)
    #GYRO
    m_feat= gFeatures(mg_unk,0)
    m_fourier=gFourier(munk,mg_unk,0)

    ifeat1_0= np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat1_0 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#-------------------------------------------------------------------------------------------------------------------------
    #ESTATICO
    is_su,ig_su=valSignal(isu,1)
    feat = sigFeatures(is_su,1)
    ifourier=tFourier(isu,is_su,1)
    feat1= lSpeed(isu)
    feat2= sigAltimeter(isu)
    #GYRO
    i_feat= gFeatures(ig_su,1)
    i_fourier=gFourier(isu,ig_su,1)
    gfeat1= lSpeed(isu)
    gfeat2= sigAltimeter(isu)
    #########MYO###########################################################
    m_su,mg_su = valSignal(msu,0)
    mfeat = sigFeatures(m_su,0)
    mfourier=tFourier(msu,m_su,0)
    #GYRO
    m_feat= gFeatures(mg_su,0)
    m_fourier=gFourier(msu,mg_su,0)

    ifeat1_1 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat1_1 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #CAMINANDO
    is_walk,ig_walk=valSignal(iwalk,1)
    feat = sigFeatures(is_walk,1)
    ifourier=tFourier(iwalk,is_walk,1)
    feat1= lSpeed(iwalk)
    feat2= sigAltimeter(iwalk)
    #GYRO
    i_feat= gFeatures(ig_walk,1)
    i_fourier=gFourier(iwalk,ig_walk,1)
    gfeat1= lSpeed(iwalk)
    gfeat2= sigAltimeter(iwalk)
 ########MYO##########################################################################
    #CAMINANDO
    m_walk,mg_walk = valSignal(mwalk,0)
    mfeat = sigFeatures(m_walk,0)
    mfourier=tFourier(mwalk,m_walk,0)
    #GYRO
    m_feat= gFeatures(mg_walk,0)
    m_fourier=gFourier(mwalk,mg_walk,0)

    ifeat1_2 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat1_2 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #CORRIENDO
    is_run,ig_run=valSignal(irun,1)
    feat = sigFeatures(is_run,1)
    ifourier=tFourier(irun,is_run,1)
    feat1= lSpeed(irun)
    feat2= sigAltimeter(irun)
    #GYRO
    i_feat= gFeatures(ig_run,1)
    i_fourier=gFourier(irun,ig_run,1)
    gfeat1= lSpeed(irun)
    gfeat2= sigAltimeter(irun)
#############MYO###########################################################
    #CORRIENDO
    m_run,mg_run = valSignal(mrun,0)
    mfeat = sigFeatures(m_run,0)
    mfourier=tFourier(mrun,m_run,0)
    #GYRO
    m_feat= gFeatures(mg_run,0)
    m_fourier=gFourier(mrun,mg_run,0)

    ifeat1_3 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat1_3 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #BICICLETA
    is_cy,ig_cy=valSignal(icy,1)
    feat = sigFeatures(is_cy,1)
    ifourier=tFourier(icy,is_cy,1)
    feat1= lSpeed(icy)
    feat2= sigAltimeter(icy)
    #GYRO
    i_feat= gFeatures(ig_cy,1)
    i_fourier=gFourier(icy,ig_cy,1)
    gfeat1= lSpeed(icy)
    gfeat2= sigAltimeter(icy)
    ###########MYO###############
    #BICICLETA
    m_cy,mg_cy = valSignal(mcy,0)
    mfeat = sigFeatures(m_cy,0)
    mfourier=tFourier(mcy,m_cy,0)
    #GYRO
    m_feat= gFeatures(mg_cy,0)
    m_fourier=gFourier(mcy,mg_cy,0)

    ifeat1_4 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat1_4 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #CARRO
    is_car,ig_car=valSignal(icar,1)
    feat = sigFeatures(is_car,1)
    ifourier=tFourier(icar,is_car,1)
    feat1= lSpeed(icar)
    feat2= sigAltimeter(icar)
    #GYRO
    i_feat= gFeatures(ig_car,1)
    i_fourier=gFourier(icar,ig_car,1)
    gfeat1= lSpeed(icar)
    gfeat2= sigAltimeter(icar)
    #######MYO##################
    #CARRO
    m_car,mg_car = valSignal(mcar,0)
    mfeat = sigFeatures(m_car,0)
    mfourier=tFourier(mcar,m_car,0)
    #GYRO
    m_feat= gFeatures(mg_car,0)
    m_fourier=gFourier(mcar,mg_car,0)

    ifeat1_5 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat1_5 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#-------------------------------------------------------------------------------------------------------------------------------------------
    #BAJANDO ESCALERAS
    is_down,ig_down=valSignal(idown,1)
    feat = sigFeatures(is_down,1)
    ifourier=tFourier(idown,is_down,1)
    feat1= lSpeed(idown)
    feat2= sigAltimeter(idown)
    #GYRO
    i_feat= gFeatures(ig_down,1)
    i_fourier=gFourier(idown,ig_down,1)
    gfeat1= lSpeed(idown)
    gfeat2= sigAltimeter(idown)
    #########MYO###########################################################
    #BAJANDO ESCALERAS
    m_down,mg_down = valSignal(mdown,0)
    mfeat = sigFeatures(m_down,0)
    mfourier=tFourier(mdown,m_down,0)
    #GYRO
    m_feat= gFeatures(mg_down,0)
    m_fourier=gFourier(mdown,mg_down,0)

    ifeat1_6 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat1_6 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
    #----------------------------------------------------------------------------------------------------------------------------------------
    #IPHONE
    user1 = np.vstack((ifeat1, ifeat1_0,ifeat1_1, ifeat1_2, ifeat1_3, ifeat1_4, ifeat1_5, ifeat1_6))

    #IPHONE+MYO
    mUser1 = np.vstack((mfeat1, mfeat1_0,mfeat1_1, mfeat1_2, mfeat1_3, mfeat1_4, mfeat1_5, mfeat1_6))
#--------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# ------USER2------------------------------------------------------------------------------------
    user = 'helman'
    #CARGAR DATOS
    iwalk, mwalk, irun, mrun, iup, mup, idown, mdown, icar, mcar,isu,msu,icy,mcy,iunk,munk = loadData(user,0)
#########IPHONE########################################################################
    #SUBIENDO ESCALERAS
    is_up,ig_up=valSignal(iup,1)
    feat = sigFeatures(is_up,1)
    ifourier=tFourier(iup,is_up,1)
    feat1= lSpeed(iup)
    feat2= sigAltimeter(iup)
    #GYRO
    i_feat= gFeatures(ig_up,1)
    i_fourier=gFourier(iup,ig_up,1)
    gfeat1= lSpeed(iup)
    gfeat2= sigAltimeter(iup)
    #########MYO###########################################################
    #SUBIENDO ESCALERAS
    m_up,mg_up = valSignal(mup,0)
    mfeat = sigFeatures(m_up,0)
    mfourier=tFourier(mup,m_up,0)
    #GYRO
    m_feat= gFeatures(mg_up,0)
    m_fourier=gFourier(mup,mg_up,0)

    ifeat2 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat2 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
    #----------------------------------------------------------------------------------------------------------------------------------------
    #UNKNOWN
    is_unk,ig_unk=valSignal(iunk,1)
    feat = sigFeatures(is_unk,1)
    ifourier=tFourier(iunk,is_unk,1)
    feat1= lSpeed(iunk)
    feat2= sigAltimeter(iunk)
    #GYRO
    i_feat= gFeatures(ig_unk,1)
    i_fourier=gFourier(iunk,ig_unk,1)
    #########MYO###########################################################
    #UNKNOWN
    m_unk,mg_unk = valSignal(munk,0)
    mfeat = sigFeatures(m_unk,0)
    mfourier=tFourier(munk,m_unk,0)
    #GYRO
    m_feat= gFeatures(mg_unk,0)
    m_fourier=gFourier(munk,mg_unk,0)

    ifeat2_0= np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat2_0 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#-------------------------------------------------------------------------------------------------------------------------
    #ESTATICO
    is_su,ig_su=valSignal(isu,1)
    feat = sigFeatures(is_su,1)
    ifourier=tFourier(isu,is_su,1)
    feat1= lSpeed(isu)
    feat2= sigAltimeter(isu)
    #GYRO
    i_feat= gFeatures(ig_su,1)
    i_fourier=gFourier(isu,ig_su,1)
    gfeat1= lSpeed(isu)
    gfeat2= sigAltimeter(isu)
    #########MYO###########################################################
    m_su,mg_su = valSignal(msu,0)
    mfeat = sigFeatures(m_su,0)
    mfourier=tFourier(msu,m_su,0)
    #GYRO
    m_feat= gFeatures(mg_su,0)
    m_fourier=gFourier(msu,mg_su,0)

    ifeat2_1 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat2_1 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #CAMINANDO
    is_walk,ig_walk=valSignal(iwalk,1)
    feat = sigFeatures(is_walk,1)
    ifourier=tFourier(iwalk,is_walk,1)
    feat1= lSpeed(iwalk)
    feat2= sigAltimeter(iwalk)
    #GYRO
    i_feat= gFeatures(ig_walk,1)
    i_fourier=gFourier(iwalk,ig_walk,1)
    gfeat1= lSpeed(iwalk)
    gfeat2= sigAltimeter(iwalk)
 ########MYO##########################################################################
    #CAMINANDO
    m_walk,mg_walk = valSignal(mwalk,0)
    mfeat = sigFeatures(m_walk,0)
    mfourier=tFourier(mwalk,m_walk,0)
    #GYRO
    m_feat= gFeatures(mg_walk,0)
    m_fourier=gFourier(mwalk,mg_walk,0)

    ifeat2_2 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat2_2 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #CORRIENDO
    is_run,ig_run=valSignal(irun,1)
    feat = sigFeatures(is_run,1)
    ifourier=tFourier(irun,is_run,1)
    feat1= lSpeed(irun)
    feat2= sigAltimeter(irun)
    #GYRO
    i_feat= gFeatures(ig_run,1)
    i_fourier=gFourier(irun,ig_run,1)
    gfeat1= lSpeed(irun)
    gfeat2= sigAltimeter(irun)
#############MYO###########################################################
    #CORRIENDO
    m_run,mg_run = valSignal(mrun,0)
    mfeat = sigFeatures(m_run,0)
    mfourier=tFourier(mrun,m_run,0)
    #GYRO
    m_feat= gFeatures(mg_run,0)
    m_fourier=gFourier(mrun,mg_run,0)

    ifeat2_3 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat2_3 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #BICICLETA
    is_cy,ig_cy=valSignal(icy,1)
    feat = sigFeatures(is_cy,1)
    ifourier=tFourier(icy,is_cy,1)
    feat1= lSpeed(icy)
    feat2= sigAltimeter(icy)
    #GYRO
    i_feat= gFeatures(ig_cy,1)
    i_fourier=gFourier(icy,ig_cy,1)
    gfeat1= lSpeed(icy)
    gfeat2= sigAltimeter(icy)
    ###########MYO###############
    #BICICLETA
    m_cy,mg_cy = valSignal(mcy,0)
    mfeat = sigFeatures(m_cy,0)
    mfourier=tFourier(mcy,m_cy,0)
    #GYRO
    m_feat= gFeatures(mg_cy,0)
    m_fourier=gFourier(mcy,mg_cy,0)

    ifeat2_4 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat2_4 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #CARRO
    is_car,ig_car=valSignal(icar,1)
    feat = sigFeatures(is_car,1)
    ifourier=tFourier(icar,is_car,1)
    feat1= lSpeed(icar)
    feat2= sigAltimeter(icar)
    #GYRO
    i_feat= gFeatures(ig_car,1)
    i_fourier=gFourier(icar,ig_car,1)
    gfeat1= lSpeed(icar)
    gfeat2= sigAltimeter(icar)
    #######MYO##################
    #CARRO
    m_car,mg_car = valSignal(mcar,0)
    mfeat = sigFeatures(m_car,0)
    mfourier=tFourier(mcar,m_car,0)
    #GYRO
    m_feat= gFeatures(mg_car,0)
    m_fourier=gFourier(mcar,mg_car,0)

    ifeat2_5 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat2_5 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#-------------------------------------------------------------------------------------------------------------------------------------------
    #BAJANDO ESCALERAS
    is_down,ig_down=valSignal(idown,1)
    feat = sigFeatures(is_down,1)
    ifourier=tFourier(idown,is_down,1)
    feat1= lSpeed(idown)
    feat2= sigAltimeter(idown)
    #GYRO
    i_feat= gFeatures(ig_down,1)
    i_fourier=gFourier(idown,ig_down,1)
    gfeat1= lSpeed(idown)
    gfeat2= sigAltimeter(idown)
    #########MYO###########################################################
    #BAJANDO ESCALERAS
    m_down,mg_down = valSignal(mdown,0)
    mfeat = sigFeatures(m_down,0)
    mfourier=tFourier(mdown,m_down,0)
    #GYRO
    m_feat= gFeatures(mg_down,0)
    m_fourier=gFourier(mdown,mg_down,0)

    ifeat2_6 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat2_6 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
    #----------------------------------------------------------------------------------------------------------------------------------------

    #IPHONE
    user2 = np.vstack((ifeat2, ifeat2_0, ifeat2_1, ifeat2_2, ifeat2_3, ifeat2_4, ifeat2_5, ifeat2_6))

    #IPHONE+MYO
    mUser2 = np.vstack((mfeat2, mfeat2_0, mfeat2_1, mfeat2_2, mfeat2_3, mfeat2_4, mfeat2_5, mfeat2_6))
#--------------------------------------------------------------------------------------------------------
#-----USER3----------------------------------------------------------------------------------------------
    user = 'tique'
    #CARGAR DATOS
    iwalk, mwalk, irun, mrun, iup, mup, idown, mdown, icar, mcar,isu,msu,icy,mcy,iunk,munk = loadData(user,0)
#########IPHONE########################################################################
    #SUBIENDO ESCALERAS
    is_up,ig_up=valSignal(iup,1)
    feat = sigFeatures(is_up,1)
    ifourier=tFourier(iup,is_up,1)
    feat1= lSpeed(iup)
    feat2= sigAltimeter(iup)
    #GYRO
    i_feat= gFeatures(ig_up,1)
    i_fourier=gFourier(iup,ig_up,1)
    gfeat1= lSpeed(iup)
    gfeat2= sigAltimeter(iup)
    #########MYO###########################################################
    #SUBIENDO ESCALERAS
    m_up,mg_up = valSignal(mup,0)
    mfeat = sigFeatures(m_up,0)
    mfourier=tFourier(mup,m_up,0)
    #GYRO
    m_feat= gFeatures(mg_up,0)
    m_fourier=gFourier(mup,mg_up,0)

    ifeat3 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat3 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
    #----------------------------------------------------------------------------------------------------------------------------------------
    #UNKNOWN
    is_unk,ig_unk=valSignal(iunk,1)
    feat = sigFeatures(is_unk,1)
    ifourier=tFourier(iunk,is_unk,1)
    feat1= lSpeed(iunk)
    feat2= sigAltimeter(iunk)
    #GYRO
    i_feat= gFeatures(ig_unk,1)
    i_fourier=gFourier(iunk,ig_unk,1)
    #########MYO###########################################################
    #UNKNOWN
    m_unk,mg_unk = valSignal(munk,0)
    mfeat = sigFeatures(m_unk,0)
    mfourier=tFourier(munk,m_unk,0)
    #GYRO
    m_feat= gFeatures(mg_unk,0)
    m_fourier=gFourier(munk,mg_unk,0)

    ifeat3_0= np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat3_0 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#-------------------------------------------------------------------------------------------------------------------------
    #ESTATICO
    is_su,ig_su=valSignal(isu,1)
    feat = sigFeatures(is_su,1)
    ifourier=tFourier(isu,is_su,1)
    feat1= lSpeed(isu)
    feat2= sigAltimeter(isu)
    #GYRO
    i_feat= gFeatures(ig_su,1)
    i_fourier=gFourier(isu,ig_su,1)
    gfeat1= lSpeed(isu)
    gfeat2= sigAltimeter(isu)
    #########MYO###########################################################
    m_su,mg_su = valSignal(msu,0)
    mfeat = sigFeatures(m_su,0)
    mfourier=tFourier(msu,m_su,0)
    #GYRO
    m_feat= gFeatures(mg_su,0)
    m_fourier=gFourier(msu,mg_su,0)

    ifeat3_1 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat3_1 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #CAMINANDO
    is_walk,ig_walk=valSignal(iwalk,1)
    feat = sigFeatures(is_walk,1)
    ifourier=tFourier(iwalk,is_walk,1)
    feat1= lSpeed(iwalk)
    feat2= sigAltimeter(iwalk)
    #GYRO
    i_feat= gFeatures(ig_walk,1)
    i_fourier=gFourier(iwalk,ig_walk,1)
    gfeat1= lSpeed(iwalk)
    gfeat2= sigAltimeter(iwalk)
 ########MYO##########################################################################
    #CAMINANDO
    m_walk,mg_walk = valSignal(mwalk,0)
    mfeat = sigFeatures(m_walk,0)
    mfourier=tFourier(mwalk,m_walk,0)
    #GYRO
    m_feat= gFeatures(mg_walk,0)
    m_fourier=gFourier(mwalk,mg_walk,0)

    ifeat3_2 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat3_2 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #CORRIENDO
    is_run,ig_run=valSignal(irun,1)
    feat = sigFeatures(is_run,1)
    ifourier=tFourier(irun,is_run,1)
    feat1= lSpeed(irun)
    feat2= sigAltimeter(irun)
    #GYRO
    i_feat= gFeatures(ig_run,1)
    i_fourier=gFourier(irun,ig_run,1)
    gfeat1= lSpeed(irun)
    gfeat2= sigAltimeter(irun)
#############MYO###########################################################
    #CORRIENDO
    m_run,mg_run = valSignal(mrun,0)
    mfeat = sigFeatures(m_run,0)
    mfourier=tFourier(mrun,m_run,0)
    #GYRO
    m_feat= gFeatures(mg_run,0)
    m_fourier=gFourier(mrun,mg_run,0)

    ifeat3_3 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat3_3 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #BICICLETA
    is_cy,ig_cy=valSignal(icy,1)
    feat = sigFeatures(is_cy,1)
    ifourier=tFourier(icy,is_cy,1)
    feat1= lSpeed(icy)
    feat2= sigAltimeter(icy)
    #GYRO
    i_feat= gFeatures(ig_cy,1)
    i_fourier=gFourier(icy,ig_cy,1)
    gfeat1= lSpeed(icy)
    gfeat2= sigAltimeter(icy)
    ###########MYO###############
    #BICICLETA
    m_cy,mg_cy = valSignal(mcy,0)
    mfeat = sigFeatures(m_cy,0)
    mfourier=tFourier(mcy,m_cy,0)
    #GYRO
    m_feat= gFeatures(mg_cy,0)
    m_fourier=gFourier(mcy,mg_cy,0)

    ifeat3_4 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat3_4 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#----------------------------------------------------------------------------------------------------------------------------------------
    #CARRO
    is_car,ig_car=valSignal(icar,1)
    feat = sigFeatures(is_car,1)
    ifourier=tFourier(icar,is_car,1)
    feat1= lSpeed(icar)
    feat2= sigAltimeter(icar)
    #GYRO
    i_feat= gFeatures(ig_car,1)
    i_fourier=gFourier(icar,ig_car,1)
    gfeat1= lSpeed(icar)
    gfeat2= sigAltimeter(icar)
    #######MYO##################
    #CARRO
    m_car,mg_car = valSignal(mcar,0)
    mfeat = sigFeatures(m_car,0)
    mfourier=tFourier(mcar,m_car,0)
    #GYRO
    m_feat= gFeatures(mg_car,0)
    m_fourier=gFourier(mcar,mg_car,0)

    ifeat3_5 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat3_5 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
#-------------------------------------------------------------------------------------------------------------------------------------------
    #BAJANDO ESCALERAS
    is_down,ig_down=valSignal(idown,1)
    feat = sigFeatures(is_down,1)
    ifourier=tFourier(idown,is_down,1)
    feat1= lSpeed(idown)
    feat2= sigAltimeter(idown)
    #GYRO
    i_feat= gFeatures(ig_down,1)
    i_fourier=gFourier(idown,ig_down,1)
    gfeat1= lSpeed(idown)
    gfeat2= sigAltimeter(idown)
    #########MYO###########################################################
    #BAJANDO ESCALERAS
    m_down,mg_down = valSignal(mdown,0)
    mfeat = sigFeatures(m_down,0)
    mfourier=tFourier(mdown,m_down,0)
    #GYRO
    m_feat= gFeatures(mg_down,0)
    m_fourier=gFourier(mdown,mg_down,0)

    ifeat3_6 = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
    mfeat3_6 = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
    #----------------------------------------------------------------------------------------------------------------------------------------
    #IPHONE
    user3 = np.vstack((ifeat3, ifeat3_0, ifeat3_1, ifeat3_2, ifeat3_3, ifeat3_4, ifeat3_5, ifeat3_6))

    #IPHONE+MYO
    mUser3 = np.vstack((mfeat3, mfeat3_0, mfeat3_1, mfeat3_2, mfeat3_3, mfeat3_4, mfeat3_5, mfeat3_6))
#CLASIFICACION
    print("-------------------------------------------------------------------")
    print("-----------------------------IPHONE------------------------------")
    print("-------------------------------------------------------------------")
    #CARGAR DATOS DE ENTRENAMIENTO
    #FIRST LEVEL
    #IPHONE
    i_RF= pickle.load(open(main_path+'data/LearningData/Jerarquico/Rutina/Normal/iPhone/i_RF.sav'))
    i_SVM=pickle.load(open(main_path+'data/LearningData/Jerarquico/Rutina/Normal/iPhone/i_SVM.sav'))
    i_NN=pickle.load(open(main_path+'data/LearningData/Jerarquico/Rutina/Normal/iPhone/i_NN.sav'))
    i_BDT_REAL=pickle.load(open(main_path+'data/LearningData/Jerarquico/Rutina/Normal/iPhone/i_BDT.sav'))
    i_GAUSS=pickle.load(open(main_path+'data/LearningData/Jerarquico/Rutina/Normal/iPhone/i_GAUSS.sav'))
    #IPHONE+Myo
    i_m_RF= pickle.load(open(main_path+'data/LearningData/Jerarquico/Rutina/Normal/iPhone+Myo/i_m_RF.sav'))
    i_m_SVM=pickle.load(open(main_path+'data/LearningData/Jerarquico/Rutina/Normal/iPhone+Myo/i_m_SVM.sav'))
    i_m_NN=pickle.load(open(main_path+'data/LearningData/Jerarquico/Rutina/Normal/iPhone+Myo/i_m_NN.sav'))
    i_m_BDT_REAL=pickle.load(open(main_path+'data/LearningData/Jerarquico/Rutina/Normal/iPhone+Myo/i_m_BDT.sav'))
    i_m_GAUSS=pickle.load(open(main_path+'data/LearningData/Jerarquico/Rutina/Normal/iPhone+Myo/i_m_GAUSS.sav'))
#----------------------------------------------------------------------------------------------
    #SECONDLEVEL
    si_RF= pickle.load(open(main_path+'data/LearningData/Jerarquico/DeepL/Rutina/Normal/iPhone/i_RF.sav'))
    si_SVM=pickle.load(open(main_path+'data/LearningData/Jerarquico/DeepL/Rutina/Normal/iPhone/i_SVM.sav'))
    si_NN=pickle.load(open(main_path+'data/LearningData/Jerarquico/DeepL/Rutina/Normal/iPhone/i_NN.sav'))
    si_BDT_REAL=pickle.load(open(main_path+'data/LearningData/Jerarquico/DeepL/Rutina/Normal/iPhone/i_BDT.sav'))
    si_GAUSS=pickle.load(open(main_path+'data/LearningData/Jerarquico/DeepL/Rutina/Normal/iPhone/i_GAUSS.sav'))
    #IPHONE+Myo
    si_m_RF= pickle.load(open(main_path+'data/LearningData/Jerarquico/DeepL/Rutina/Normal/iPhone+Myo/i_m_RF.sav'))
    si_m_SVM=pickle.load(open(main_path+'data/LearningData/Jerarquico/DeepL/Rutina/Normal/iPhone+Myo/i_m_SVM.sav'))
    si_m_NN=pickle.load(open(main_path+'data/LearningData/Jerarquico/DeepL/Rutina/Normal/iPhone+Myo/i_m_NN.sav'))
    si_m_BDT_REAL=pickle.load(open(main_path+'data/LearningData/Jerarquico/DeepL/Rutina/Normal/iPhone+Myo/i_m_BDT.sav'))
    si_m_GAUSS=pickle.load(open(main_path+'data/LearningData/Jerarquico/DeepL/Rutina/Normal/iPhone+Myo/i_m_GAUSS.sav'))
#----------------------------------------------------------------------------------------------
    datos=np.vstack((user1,user2,user3))
    mdatos=np.vstack((mUser1,mUser2,mUser3))
    print(mdatos.shape)
    Validacion=np.zeros([0,5])
    mValidacion=np.zeros([0,5])
    target=np.array([[0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]]).T
    target=np.ravel(target)
    one=[]
    two=[]
    three=[]
    four=[]
    five=[]
    mone=[]
    mtwo=[]
    mthree=[]
    mfour=[]
    mfive=[]

    for k in range(0,24):
        row=datos[k,:]
        row=row.reshape(1, -1)
        mrow=mdatos[k,:]
        mrow=mrow.reshape(1, -1)
        #--------VALIDACION----------------------------------------------------------------------------------------------------------
        #PREDICTION IPHONE SIN PCA
        i_pred_RF=i_RF.predict(row)
        if i_pred_RF == 3:
            i_pred_RF= si_RF.predict(row)
        elif i_pred_RF == 4:
            i_pred_RF= si_RF.predict(row)
#-----------------------------------------------------------------------------
        i_pred_SVM=i_SVM.predict(row)

        if i_pred_SVM == 3:
            i_pred_SVM= si_SVM.predict(row)

        elif i_pred_SVM == 4:
            i_pred_SVM= si_SVM.predict(row)
#-----------------------------------------------------------------------------
        i_pred_NN=i_NN.predict(row)

        if i_pred_NN == 3:
            i_pred_NN= si_NN.predict(row)

        elif i_pred_NN == 4:
            i_pred_NN= si_NN.predict(row)
#-----------------------------------------------------------------------------
        i_pred_BDT_REAL=i_BDT_REAL.predict(row)

        if i_pred_BDT_REAL == 3:
            i_pred_BDT_REAL= si_BDT_REAL.predict(row)

        elif i_pred_BDT_REAL == 4:
            i_pred_BDT_REAL= si_BDT_REAL.predict(row)
#-----------------------------------------------------------------------------
        i_pred_GAUSS=i_GAUSS.predict(row)

        if i_pred_GAUSS == 3:
            i_pred_GAUSS= si_GAUSS.predict(row)

        elif i_pred_GAUSS == 4:
            i_pred_GAUSS= si_GAUSS.predict(row)

        Val =np.concatenate((i_pred_RF,i_pred_SVM,i_pred_NN,i_pred_BDT_REAL,i_pred_GAUSS))
        Validacion= np.vstack((Validacion, Val))

        one=np.concatenate((one,i_pred_RF))
        two=np.concatenate((two,i_pred_SVM))
        three=np.concatenate((three,i_pred_NN))
        four=np.concatenate((four,i_pred_BDT_REAL))
        five=np.concatenate((five,i_pred_GAUSS))
        #---------------------------------------------------------------------
        #-----------------------------IPHONE+MYO------------------------------
        i_m_pred_RF=i_m_RF.predict(mrow)

        if i_m_pred_RF == 3:
            i_m_pred_RF=si_m_RF.predict(mrow)
        elif i_m_pred_RF == 4:
            i_m_pred_RF= si_m_RF.predict(mrow)
        #----------------------------------------------------------------------
        i_m_pred_SVM=i_m_SVM.predict(mrow)

        if i_m_pred_SVM == 3:
            i_m_pred_SVM=si_m_SVM.predict(mrow)
        elif i_m_pred_SVM == 4:
            i_m_pred_SVM= si_m_SVM.predict(mrow)
        #----------------------------------------------------------------------
        i_m_pred_NN=i_m_NN.predict(mrow)

        if i_m_pred_NN == 3:
            i_m_pred_NN=si_m_NN.predict(mrow)
        elif i_m_pred_NN == 4:
            i_m_pred_NN= si_m_NN.predict(mrow)
        #----------------------------------------------------------------------
        i_m_pred_BDT_REAL=i_m_BDT_REAL.predict(mrow)

        if i_m_pred_BDT_REAL == 3:
            i_m_pred_BDT_REAL=si_m_BDT_REAL.predict(mrow)
        elif i_m_pred_BDT_REAL == 4:
            i_m_pred_BDT_REAL= si_m_BDT_REAL.predict(mrow)
        #----------------------------------------------------------------------
        i_m_pred_GAUSS=i_m_GAUSS.predict(mrow)

        if i_m_pred_GAUSS == 3:
            i_m_pred_GAUSS=si_m_GAUSS.predict(mrow)
        elif i_m_pred_GAUSS == 4:
            i_m_pred_GAUSS= si_m_GAUSS.predict(mrow)
        #----------------------------------------------------------------------
        mVal =np.concatenate((i_m_pred_RF,i_m_pred_SVM,i_m_pred_NN,i_m_pred_BDT_REAL,i_m_pred_GAUSS))
        mValidacion= np.vstack((mValidacion, mVal))
        mone=np.concatenate((mone,i_m_pred_RF))
        mtwo=np.concatenate((mtwo,i_m_pred_SVM))
        mthree=np.concatenate((mthree,i_m_pred_NN))
        mfour=np.concatenate((mfour,i_m_pred_BDT_REAL))
        mfive=np.concatenate((mfive,i_m_pred_GAUSS))

    ValScores = val_Scores(target,one,two,three,four,five)
    mValScores = val_Scores(target,mone,mtwo,mthree,mfour,mfive)
    print("IPHONE")
    print(Validacion)
    print("ValScores")
    print(ValScores)
    print("IPHONE+MYO")
    print(mValidacion)
    print("mValScores")
    print(mValScores)

#-----------------------------------------------------------------------------
