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

def ival_Scores(Y,pred_1):
    f1_iPred   = f1_score(Y, pred_1, average=None)
    F1_scores=f1_iPred
    return F1_scores
if __name__ == '__main__':
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
#----USER1------------------------------------------------------------------
    user = 'jgm'
    #CARGAR DATOS
    irut,mrut = loadData(user)
    n_idata=500
    n_mdata=250
    N_idata=len(irut)
    N_mdata=len(mrut)
    Validacion=np.zeros([0,5])
    mValidacion=np.zeros([0,5])
    delta=[]
    counter = 0
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
    i_Pred=[]
   
    for k in range(0,N_idata,n_idata):
        segment=irut[k:k+n_idata]
        is_rut,ig_rut=magSignal(segment,1)
        feat = sigFeatures(is_rut,1)
        ifourier=tFourier(segment,is_rut,1)
        feat1= lSpeed(segment)
        feat2= sigAltimeter(segment)
        #GYRO
        i_feat= gFeatures(ig_rut,1)
        i_fourier=gFourier(segment,ig_rut,1)
        user1=np.zeros([1,14])
        user1[0,:] = np.concatenate((feat,ifourier,i_feat, i_fourier,feat1,feat2), axis=None)
        user1=user1.reshape(1, -1)
        #CLASIFICACION
        #--------VALIDACION----------------------------------------------------------------------------------------------------------
        #PREDICTION IPHONE SIN PCA
        i_pred_RF=i_RF.predict(user1)
        if i_pred_RF == 3:
            i_pred_RF= si_RF.predict(user1)
        elif i_pred_RF == 4:
            i_pred_RF= si_RF.predict(user1)
#-----------------------------------------------------------------------------
        i_pred_SVM=i_SVM.predict(user1)

        if i_pred_SVM == 3:
            i_pred_SVM= si_SVM.predict(user1)

        elif i_pred_SVM == 4:
            i_pred_SVM= si_SVM.predict(user1)
#-----------------------------------------------------------------------------
        i_pred_NN=i_NN.predict(user1)

        if i_pred_NN == 3:
            i_pred_NN= si_NN.predict(user1)

        elif i_pred_NN == 4:
            i_pred_NN= si_NN.predict(user1)
#-----------------------------------------------------------------------------
        i_pred_BDT_REAL=i_BDT_REAL.predict(user1)

        if i_pred_BDT_REAL == 3:
            i_pred_BDT_REAL= si_BDT_REAL.predict(user1)

        elif i_pred_BDT_REAL == 4:
            i_pred_BDT_REAL= si_BDT_REAL.predict(user1)
#-----------------------------------------------------------------------------
        i_pred_GAUSS=i_GAUSS.predict(user1)

        if i_pred_GAUSS == 3:
            i_pred_GAUSS= si_GAUSS.predict(user1)

        elif i_pred_GAUSS == 4:
            i_pred_GAUSS= si_GAUSS.predict(user1)

        Val =np.concatenate((i_pred_RF,i_pred_SVM,i_pred_NN,i_pred_BDT_REAL,i_pred_GAUSS))
        Validacion= np.vstack((Validacion, Val))

        one=np.concatenate((one,i_pred_RF))
        two=np.concatenate((two,i_pred_SVM))
        three=np.concatenate((three,i_pred_NN))
        four=np.concatenate((four,i_pred_BDT_REAL))
        five=np.concatenate((five,i_pred_GAUSS))
        
        # ----------------------------------------------------------------------------------------
        #-------------MYO-------------------------------------------------------------------------
        if k==0:
            segment_m = mrut[k:k+n_mdata]
        else:
            segment_m = mrut[k-250*counter:k-250*counter+n_mdata]
        counter=counter+1
        if len(segment_m) > 0:
            #DESCRIPTORES
            m_rut,mg_rut = magSignal(segment_m,0)
            m_rut=list(filter(None,m_rut))
            mfeat = sigFeatures(m_rut,0)
            mfourier=tFourier(segment_m,m_rut,0)
            #GYRO
            m_feat= gFeatures(mg_rut,0)
            m_fourier=gFourier(segment_m,mg_rut,0)
            #USUARIO
            mUser1=np.zeros([1,25])
            mUser1[0,:] = np.concatenate((feat,mfeat,ifourier,mfourier,i_feat,m_feat,i_fourier,m_fourier,feat1,feat2), axis=None)
            #---------------------------------------------------------------------------------------------------------------------------
            #CLASIFICACION
            #--------VALIDACION----------------------------------------------------------------------------------------------------------
            #PREDICTION IPHONE+MYO SIN PCA
            i_m_pred_RF=i_m_RF.predict(mUser1)

            if i_m_pred_RF == 3:
                i_m_pred_RF=si_m_RF.predict(mUser1)
            elif i_m_pred_RF == 4:
                i_m_pred_RF= si_m_RF.predict(mUser1)
            #----------------------------------------------------------------------
            i_m_pred_SVM=i_m_SVM.predict(mUser1)

            if i_m_pred_SVM == 3:
                i_m_pred_SVM=si_m_SVM.predict(mUser1)
            elif i_m_pred_SVM == 4:
                i_m_pred_SVM= si_m_SVM.predict(mUser1)
            #----------------------------------------------------------------------
            i_m_pred_NN=i_m_NN.predict(mUser1)

            if i_m_pred_NN == 3:
                i_m_pred_NN=si_m_NN.predict(mUser1)
            elif i_m_pred_NN == 4:
                i_m_pred_NN= si_m_NN.predict(mUser1)
            #----------------------------------------------------------------------
            i_m_pred_BDT_REAL=i_m_BDT_REAL.predict(mUser1)

            if i_m_pred_BDT_REAL == 3:
                i_m_pred_BDT_REAL=si_m_BDT_REAL.predict(mUser1)
            elif i_m_pred_BDT_REAL == 4:
                i_m_pred_BDT_REAL= si_m_BDT_REAL.predict(mUser1)
            #----------------------------------------------------------------------
            i_m_pred_GAUSS=i_m_GAUSS.predict(mUser1)

            if i_m_pred_GAUSS == 3:
                i_m_pred_GAUSS=si_m_GAUSS.predict(mUser1)
            elif i_m_pred_GAUSS == 4:
                i_m_pred_GAUSS= si_m_GAUSS.predict(mUser1)
            #----------------------------------------------------------------------
            mVal =np.concatenate((i_m_pred_RF,i_m_pred_SVM,i_m_pred_NN,i_m_pred_BDT_REAL,i_m_pred_GAUSS))
            mValidacion= np.vstack((mValidacion, mVal))
            mone=np.concatenate((mone,i_m_pred_RF))
            mtwo=np.concatenate((mtwo,i_m_pred_SVM))
            mthree=np.concatenate((mthree,i_m_pred_NN))
            mfour=np.concatenate((mfour,i_m_pred_BDT_REAL))
            mfive=np.concatenate((mfive,i_m_pred_GAUSS))
        else:
            print('segmento menor a 150')
    print(delta)
    print('iPhone: '+str(Validacion))
    print('iPhone+Myo: '+str(mValidacion))
    tfinal=415
    iPred=irut.activity
    A= 6*np.ones([1,105])
    B= 1*np.ones([1,10])
    C= 3*np.ones([1,85])
    D= 0*np.ones([1,35])
    E= 7*np.ones([1,35])
    G= 2*np.ones([1,45])
    H= 1*np.ones([1,5])
    I= 5*np.ones([1,55])
    J= 1*np.ones([1,5])
    K= 4*np.ones([1,40])
    Target = np.concatenate((A,B,C,D,E,G,H,I,J,K),axis=None)
    time_1 = np.linspace(0,tfinal,len(iPred))
    time_2 = np.linspace(0,tfinal,len(Validacion[:,0]))
    time_target = np.linspace(0,tfinal,len(Target))
    time_alt=np.linspace(0,tfinal,len(delta))
    target= [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,0,7,7,7,7,7,7,7,2,2,2,2,2,2,2,2,2,1,5,5,5,5,5,5,5,5,5,5,5,1,4,4,4,4,4,4,4,4]
    target=np.ravel(target)
    
    iA= 6*np.ones([1,10500])
    iB= 1*np.ones([1,1000])
    iC= 3*np.ones([1,8500])
    iD= 0*np.ones([1,3500])
    iE= 7*np.ones([1,3500])
    iG= 2*np.ones([1,4500])
    iH= 1*np.ones([1,500])
    iI= 5*np.ones([1,5500])
    iJ= 1*np.ones([1,500])
    iK= 4*np.ones([1,3633])
    itarget = np.concatenate((iA,iB,iC,iD,iE,iG,iH,iI,iJ,iK),axis=None)
    itarget=np.ravel(itarget)
    ValScores = val_Scores(target,one,two,three,four,five)
    mValScores = val_Scores(target,mone,mtwo,mthree,mfour,mfive)
    iValScores= ival_Scores(itarget,iPred)
    print("IPHONE")
    print("ValScores")
    print(ValScores)
    print("IPHONE+MYO")
    print("mValScores")
    print(mValScores)
    

    plt.plot(time_target,Target, label='Target')
    plt.plot(time_2,Validacion[:,0],'*', label='Prediction')
    plt.plot(time_1,iPred,'-',label='iPhone Prediction')
    plt.legend(loc="upper right") 
    plt.ylabel('Prediction')
    plt.xlabel('Number of Samples')
    plt.show()

    plt.figure(2)
    plt.plot(time_target,Target)
    plt.plot(time_2,Validacion[:,1],'*')
    plt.plot(time_1,iPred,'-')
    plt.title("iPhone:SVM")
    plt.show()

    plt.figure(3)
    plt.plot(time_target,Target)
    plt.plot(time_2,Validacion[:,2],'*')
    plt.plot(time_1,iPred,'-')
    plt.title("iPhone:NN")
    plt.show()

    plt.figure(4)
    plt.plot(time_target,Target)
    plt.plot(time_2,Validacion[:,3],'*')
    plt.plot(time_1,iPred,'-')
    plt.title("iPhone:AdaBoost")
    plt.show()

    plt.figure(5)
    plt.plot(time_target,Target)
    plt.plot(time_2,Validacion[:,4],'*')
    plt.plot(time_1,iPred,'-')
    plt.title("iPhone:Gauss")
    plt.show()

    plt.figure(6)
    plt.plot(time_target,Target, label='Target')
    plt.plot(time_2,mValidacion[:,0],'*', label='Prediction')
    plt.plot(time_1,iPred,'-', label='iPhone Prediction')
    plt.legend(loc="upper right")
    plt.ylabel('Prediction')
    plt.xlabel('Number of Samples')
    plt.show()

    plt.figure(7)
    plt.plot(time_target,Target, label='Target')
    plt.plot(time_2,mValidacion[:,1],'*', label='Prediction')
    plt.plot(time_1,iPred,'-', label= 'iPhone Target')
    plt.title("iPhone+Myo:SVM")
    plt.show()

    plt.figure(8)
    plt.plot(time_target,Target)
    plt.plot(time_2,mValidacion[:,2],'*')
    plt.plot(time_1,iPred,'-')
    plt.title("iPhone+Myo:NN")
    plt.show()

    plt.figure(9)
    plt.plot(time_target,Target)
    plt.plot(time_2,mValidacion[:,3],'*')
    plt.plot(time_1,iPred,'-')
    plt.title("iPhone+Myo:AdaBoost")
    plt.show()

    plt.figure(10)
    plt.plot(time_target,Target)
    plt.plot(time_2,mValidacion[:,4],'*')
    plt.plot(time_1,iPred,'-')
    plt.title("iPhone+Myo:Gauss")
    plt.show()

    plt.figure(11)
    plt.plot(time_1,irut.AX, label='aX')
    plt.plot(time_1,irut.AY, label= 'aY')
    plt.plot(time_1,irut.AZ, label= 'aZ')
    plt.legend(loc="upper right")
    plt.show()
    
    print("iPhone Prediction")
    print(iValScores)