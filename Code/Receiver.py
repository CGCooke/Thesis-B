import matplotlib.pyplot as plt
import time

import numpy as np
import TrackingLoop
import GenSignal
import Scenario

def runReceiver(sampleTimes,SignalIn,PLLBW,FLLBW):
    GpsSignal = TrackingLoop.InitGpsSignal(PLLBW,FLLBW)

    dT = sampleTimes[1]-sampleTimes[0]  # Dump period of 1 ms

    numInputSamples = sampleTimes.size

    # DCO Freq & DCO Phase
    DcoAccumPhase = 0
    
    # Pre-allocate arrays
    numOutputSamples = int(np.floor(numInputSamples/GpsSignal.coherentIntegrationTime)+1)
    DcoFreqArray = np.zeros(numOutputSamples)
    LockStateValues = np.zeros(numOutputSamples)
    
    Z = np.zeros(numInputSamples,dtype=np.complex128)
    ZCoh = np.zeros(numOutputSamples,dtype=np.complex128)

    Zi = Z[0]

    SampleCount = 0
    BatchCount = 0
    
    for i in range(0,numInputSamples):  
        DcoFreq = GpsSignal.DopplerHistHzS[2]
        DcoAccumPhase += DcoFreq*dT
        
        Z[i] = SignalIn[i]*np.exp(-1J*2*np.pi*DcoAccumPhase)
        
        if (SampleCount==0):
            IPBuffer = np.zeros(GpsSignal.coherentIntegrationTime,dtype=np.complex128)
            
        IPBuffer[SampleCount] = Z[i]
        SampleCount += 1               
        
        # A full integration period has been obtained, so perform feedback
        if (SampleCount==GpsSignal.coherentIntegrationTime):
            SampleCount = 0

            # Integrate and dump        
            Zi = np.sum(IPBuffer)/float(GpsSignal.coherentIntegrationTime)

            # Run the tracking loop
            TrackingLoop.TrackingLoop(Zi,GpsSignal)
            
            # Record values for reference
            ZCoh[BatchCount] = Zi
            
            LockStateValues[BatchCount] = GpsSignal.TrackMode
            DcoFreqArray[BatchCount] = GpsSignal.DopplerHistHzS[0]
            
            BatchCount += 1
            
        #Kind of suspicious about this delay, it looks like it's delaying 2ms,  
        # Emulate delays in writing from the SW firmware to the HW correlator 
        GpsSignal.DopplerHistHzS[2] = GpsSignal.DopplerHistHzS[1]
        GpsSignal.DopplerHistHzS[1] = GpsSignal.DopplerHistHzS[0]
    
    return(Z,DcoFreqArray,LockStateValues)

def runTrial():
    GpsSignal = TrackingLoop.InitGpsSignal(1,1)

    minPLLBW = 0
    maxPLLBW = 32
    minFLLBW = 0
    maxFLLBW = 15
    for CNR in range(38,39):
        manifold = np.zeros((maxPLLBW,maxFLLBW))
        T,SignalIn,stateHistory = GenSignal.createStepAccelerationScenario(CNR)        
        for PLLBW in range(minPLLBW,maxPLLBW):
            for FLLBW in range(minFLLBW,maxFLLBW):
                startTime = time.time()
                Z,DcoFreqArray,LockStateValues = runReceiver(T,SignalIn,PLLBW,FLLBW)
                breakingPoint =  100*(np.mean(LockStateValues)-2) #m/s^2
                #freqError = (5.25*stateHistory[::GpsSignal.coherentIntegrationTime,1]-DcoFreqArray[:100000/GpsSignal.coherentIntegrationTime])
                manifold[PLLBW,FLLBW] = breakingPoint
                print(PLLBW,FLLBW,time.time()-startTime)

        plt.title('Breaking point (m/s^2)')
        #plt.title('Total RMS Error')
        plt.ylabel('PLL Bandwidth (Hz)')
        plt.xlabel('FLL Bandwidth (Hz)')

        plt.imshow(manifold, interpolation='nearest',cmap ='RdYlGn_r',origin='lower')
        plt.grid()
        plt.clim(0,100)
        plt.colorbar()
        plt.savefig(str(CNR)+'A.png',dpi=450)
        plt.close()

def compareRecievers():
    channelDict = Scenario.readNMEAFile()
    satelliteDict = Scenario.readSimulatorMotionFile()
    startTime = 406800 #second of week

    CNR = 48
    PLLBW = 32
    FLLBW = 10

    for SVNum in [1,3,6,11,16,18,19,22,31]:
        T,SignalIn,stateHistory = GenSignal.createScenarioFromSpirentFile(SVNum,CNR)
        coherentIntegrationTime = 4
        numOutputSamples = T.size/coherentIntegrationTime

        print('Running Reciever')
        Z,DcoFreqArray,LockStateValues = runReceiver(T,SignalIn,PLLBW,FLLBW)
        PolarisFreq = DcoFreqArray

        plt.plot(DcoFreqArray)
        plt.plot(LockStateValues)
        plt.show()

        '''
        time,losX,losY,losZ = satelliteDict[SVNum]
        SprientTime = time[:-1]/1000
        SpirentVelocity = np.diff(np.sqrt(losX**2+losY**2+losZ**2))
        NamuruTime = channelDict[SVNum][:,0]-406800
        NamuruVelocity = channelDict[SVNum][:,1]

        SpirentVelocity-=SpirentVelocity.mean()
        NamuruVelocity-=NamuruVelocity.mean()
        PolarisFreq-=PolarisFreq.mean()        

        plt.title(str(SVNum))
        plt.plot(SprientTime,5.25*SpirentVelocity,label='SPIRENT')
        plt.plot(NamuruTime,5.25*NamuruVelocity,label='NAMURU')
        
        polarisTime = np.arange(0,PolarisFreq.size/250,0.004)
        
        plt.plot(polarisTime+SprientTime[0],PolarisFreq ,label='POLARIS')
        plt.legend()

        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.show()
        '''
        f = open(str(SVNum)+'PLLFLL.npy',"wb")
        np.save(f,DcoFreqArray)

def compareNumpyArrays():
    PurePLL = np.load(open('1PurePLL.npy',"rb"))
    PLLFLL = np.load(open('1PLLFLL.npy',"rb"))
    numSamples = 1000
    time = 0.004*np.arange(0,1000)#np.linspace(0,numSamples,0.004)
    plt.xlabel('Time (s)')
    plt.ylabel('NCO Frequency (Hz)')
    plt.plot(time,PurePLL[0:numSamples],color='k',alpha=0.7,label='Cooke PLL')
    plt.plot(time,PLLFLL[0:numSamples],color='r',alpha=0.7,label='Kaplan PLL + FLL')
    plt.grid()
    plt.legend()
    plt.savefig('LoopComparison.png',dpi=900)
    plt.close()

    numSamples = 250
    time = 0.004*np.arange(0,numSamples)#np.linspace(0,numSamples,0.004)
    plt.xlabel('Time (s)')
    plt.ylabel('NCO Frequency (Hz)')
    plt.plot(time,PurePLL[250:250+numSamples],color='k',alpha=0.7,label='Cooke PLL')
    plt.plot(time,PLLFLL[250:250+numSamples],color='r',alpha=0.7,label='Kaplan PLL + FLL')
    plt.grid()
    plt.legend()
    plt.savefig('LoopComparison2.png',dpi=900)

    
    #plt.show()




if __name__ == '__main__':
    #At burnout for first stage, accelerations are as follows:
    #40 m/s^2 X,1.5 m/s^2 Y, 20 m/s^2 Z
    #Must do 10, please do 15
    #runTrial()
    #compareRecievers()
    compareNumpyArrays()

    '''
    CNR = 38
    T,SignalIn,stateHistory = GenSignal.createStepAccelerationScenario(CNR)        
    PLLBW = 32
    FLLBW = 10
    GpsSignal = TrackingLoop.InitGpsSignal(FLLBW,PLLBW)
    Z,DcoFreqArray,LockStateValues = runReceiver(T,SignalIn,PLLBW,FLLBW)
    breakingPoint =  100*(np.mean(LockStateValues)-2) #m/s^2
    plt.plot(5.25*stateHistory[::GpsSignal.coherentIntegrationTime,1])
    plt.plot(DcoFreqArray[:100000/GpsSignal.coherentIntegrationTime])
    plt.show()
    #freqError = (5.25*stateHistory[::GpsSignal.coherentIntegrationTime,1]-DcoFreqArray[:100000/GpsSignal.coherentIntegrationTime])
    #manifold[PLLBW,FLLBW] = breakingPoint
    #print(PLLBW,FLLBW,time.time()-startTime)
    '''