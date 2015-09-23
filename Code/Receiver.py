import matplotlib.pyplot as plt
import time

import numpy as np
import TrackingLoop
import GenSignal

def runReceiver(sampleTimes,SignalIn,PLLBW,FLLBW):
    GpsSignal = TrackingLoop.InitGpsSignal(PLLBW,FLLBW)

    dT = sampleTimes[1]-sampleTimes[0]             # Dump period of 1 ms

    numInputSamples = sampleTimes.size

    # DCO Freq & DCO Phase
    DcoAccumPhase = 0
    DcoFreqHz = GpsSignal.CarrierDcoFreqHz
    
    # Pre-allocate arrays
    numOutputSamples= int(np.floor(numInputSamples/GpsSignal.coherentIntegrationTime)+1)
    PhaseCycles = np.zeros(numOutputSamples)
    DeltaPhaseCycles = np.zeros(numOutputSamples)
    DeltaFreqHzValues = np.zeros(numOutputSamples)
    DcoFreqHzValues = np.zeros(numOutputSamples)
    CarrierDcoFreqHzValues = np.zeros(numOutputSamples)
    LockStateValues = np.zeros(numOutputSamples)
    
    Z = np.zeros(numInputSamples,dtype=np.complex128)
    ZCoh = np.zeros(numOutputSamples,dtype=np.complex128)

    Zi = Z[0]

    SampleCount = 0
    BatchCount = 0

    for i in range(0,numInputSamples):  
        DcoFreqHz = GpsSignal.DopplerHistHzS[2]
        DcoAccumPhase   = DcoAccumPhase + DcoFreqHz*dT
        
        Z[i]  = SignalIn[i]*np.exp(-1J*2*np.pi*DcoAccumPhase)
        
        if (BatchCount>=numOutputSamples):
            break
        
        if (SampleCount==0):
            IPBuffer = np.zeros(GpsSignal.coherentIntegrationTime,dtype=np.complex128)
            
        IPBuffer[SampleCount] = Z[i]
        SampleCount = SampleCount + 1                
        
        # A full integration period has been obtained, so perform feedback
        if (SampleCount==GpsSignal.coherentIntegrationTime):
            SampleCount = 0        
            Zi = np.sum(IPBuffer)/float(GpsSignal.coherentIntegrationTime)
            # Run the tracking loop
            TrackingLoop.TrackingLoop(Zi,GpsSignal)
            
            # Record values for reference
            ZCoh[BatchCount] = Zi
            PhaseCycles[BatchCount] = GpsSignal.Phase
            DeltaPhaseCycles[BatchCount] = GpsSignal.DeltaPhase2Quad
            DeltaFreqHzValues[BatchCount] = GpsSignal.DeltaFreqHz
            CarrierDcoFreqHzValues[BatchCount] = GpsSignal.DopplerHistHzS[2]
            
            LockStateValues[BatchCount] = GpsSignal.TrackMode
            DcoFreqHzValues[BatchCount] = GpsSignal.DopplerHistHzS[0]
            
            BatchCount = BatchCount + 1
            
        # Emulate delays in writing from the SW firmware to the HW correlator 
        GpsSignal.DopplerHistHzS[2] = GpsSignal.DopplerHistHzS[1]
        GpsSignal.DopplerHistHzS[1] = GpsSignal.DopplerHistHzS[0]
    
    return(Z,DcoFreqHzValues,LockStateValues)

def runTrial():
    for CNR in range(38,39):
        manifold = np.zeros((32,15))
        T,SignalIn,stateHistory = GenSignal.createStepAccelerationScenario(CNR)        
        for PLLBW in range(1,32):
            for FLLBW in range(1,15):
                startTime = time.time()
                Z,DcoFreqHzValues,LockStateValues = runReceiver(T,SignalIn,PLLBW,FLLBW)
                breakingPoint =  100*(np.mean(LockStateValues)-2) #m/s^2
                freqError = (5.25*stateHistory[::4,1]-DcoFreqHzValues[:25000])
                manifold[PLLBW,FLLBW] =breakingPoint
                print(PLLBW,FLLBW,time.time()-startTime)

        plt.title('Breaking point (m/s^2)')
        #plt.title('Total RMS Error')
        plt.ylabel('PLL Bandwidth (Hz)')
        plt.xlabel('FLL Bandwidth (Hz)')

        plt.imshow(manifold, interpolation='nearest',cmap ='RdYlGn_r',origin='lower')
        
        plt.clim(0,100)
        plt.colorbar()
        plt.savefig(str(CNR)+'8ms.png',dpi=450)
        plt.close()

if __name__ == '__main__':
    #At burnout for first stage, accelerations are as follows:
    #40 m/s^2 X,1.5 m/s^2 Y, 20 m/s^2 Z
    #Must do 10, please do 15
    
    CNR = 48
    PLLBW = 32
    FLLBW = 10

    T,SignalIn,stateHistory = GenSignal.createScenarioFromSpirentFile(CNR)
    coherentIntegrationTime = 4
    numOutputSamples = T.size/coherentIntegrationTime

    print('Running Reciever')
    startTime = time.time()
    Z,DcoFreqHzValues,LockStateValues = runReceiver(T,SignalIn,PLLBW,FLLBW)
    simulatedFreq = (5.25*stateHistory[::coherentIntegrationTime,1])
    
    plt.plot(DcoFreqHzValues)
    plt.plot(simulatedFreq,color='k')
    
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Sample Number')
    plt.show()

    '''
    CNR = 38
    PLLBW = 18
    FLLBW = 1

    coherentIntegrationTime = 4
    numOutputSamples = 100000/coherentIntegrationTime

    T,SignalIn,stateHistory = GenSignal.createStepAccelerationScenario(CNR)        
    startTime = time.time()
    Z,DcoFreqHzValues,LockStateValues = runReceiver(T,SignalIn,)
    breakingPoint =  100*(np.mean(LockStateValues)-2) #m/s^2
    simulatedFreq = (5.25*stateHistory[::coherentIntegrationTime,1])
    freqError = (simulatedFreq-DcoFreqHzValues[:numOutputSamples])  
    
    plt.plot(freqError,color='k')
    plt.title('Freq Error (Hz)')
    plt.ylabel('Frequency error (Hz)')
    plt.xlabel('Sample Number')
    plt.savefig('out.png')
    plt.show()
    '''