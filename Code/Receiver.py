import matplotlib.pyplot as plt
import numpy as np
import time

import TrackingLoop
import GenSignal
import Scenario
import TrackingLoopCameron

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def runReceiver(sampleTimes,SignalIn,PLLBW,FLLBW):
    GpsSignal = TrackingLoopCameron.InitGpsSignal(PLLBW,FLLBW)
    dT = sampleTimes[1]-sampleTimes[0]  # Dump period of 1 ms
    numInputSamples = sampleTimes.size

    # DCO Freq & DCO Phase
    DcoAccumPhase = 0
    
    # Pre-allocate arrays
    numOutputSamples = int(np.floor(numInputSamples/ \
        GpsSignal.coherentIntegrationTime)+1)
    DcoFreqArray = np.zeros(numOutputSamples)
    LockStateValues = np.zeros(numOutputSamples)
    PhaseError = np.zeros(numOutputSamples)

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
            IPBuffer = np.zeros(GpsSignal.coherentIntegrationTime, \
                dtype=np.complex128)
            
        IPBuffer[SampleCount] = Z[i]
        SampleCount += 1               
        
        # A full integration period has been obtained,
        # so perform feedback
        if (SampleCount==GpsSignal.coherentIntegrationTime):
            SampleCount = 0

            # Integrate and dump        
            Zi = np.sum(IPBuffer)/float(GpsSignal.coherentIntegrationTime)

            # Run the tracking loop
            TrackingLoopCameron.TrackingLoop(Zi,GpsSignal)
            
            # Record values for reference
            ZCoh[BatchCount] = Zi
            
            LockStateValues[BatchCount] = GpsSignal.TrackMode
            DcoFreqArray[BatchCount] = GpsSignal.DopplerHistHzS[0]
            PhaseError[BatchCount] = GpsSignal.Phase2Quadrant

            BatchCount += 1
        
        # Emulate delays in writing from the SW 
        # firmware to the HW correlator 
        GpsSignal.DopplerHistHzS[2] = GpsSignal.DopplerHistHzS[1]
        GpsSignal.DopplerHistHzS[1] = GpsSignal.DopplerHistHzS[0]
    
    return(Z,DcoFreqArray,LockStateValues,PhaseError)

def runTrial():
    GpsSignal = TrackingLoop.InitGpsSignal(1,1)

    minPLLBW = 10
    maxPLLBW = 32
    minFLLBW = 0
    maxFLLBW = 15
    PLLRange = np.arange(minPLLBW,maxPLLBW)
    FLLRange = np.arange(minFLLBW,maxFLLBW)

    for CNO in range(45,46):
        manifold = np.zeros((1*(maxPLLBW-minPLLBW),1*(maxFLLBW-minFLLBW)))
        T,SignalIn,stateHistory = GenSignal.constantJerk(1,CNO,200)

        for PLLBW in PLLRange:
            for FLLBW in FLLRange:
                startTime = time.time()
                Z,DcoFreqArray,LockStateValues,PhaseError 
                = runReceiver(T,SignalIn,PLLBW,FLLBW)
                LockStateValues = np.where(LockStateValues == 3, 1, 0)
                breakingPoint = 200*(np.mean(LockStateValues)) #m/s^2
                
                if breakingPoint<0:
                    breakingPoint=0

                manifold[1*(PLLBW-minPLLBW),1*(FLLBW-minFLLBW)] 
                = breakingPoint
                
        plt.title('Breaking point $(m/s^2)$')
        plt.ylabel('PLL Bandwidth (Hz)')
        plt.xlabel('FLL Bandwidth (Hz)')

        CS = plt.contour(FLLRange,PLLRange,manifold,colors='k')
        
        plt.clabel(CS,CS.levels,inline=True,fontsize=10)

        levels =[-1,0]
        plt.contourf(FLLRange,PLLRange, \
        manifold,levels,colors = ('r'),alpha=0.7)
        
        plt.grid()
        plt.savefig('Acceleration.eps', format='eps', dpi=1000)
        plt.close()

def compareRecievers():
    channelDict = Scenario.readNMEAFile()
    satelliteDict = Scenario.readSimulatorMotionFile()
    startTime = 406800 #second of week

    CNO = 48
    PLLBW = 2
    FLLBW = 10

    for SVNum in [1,3,6,11,16,18,19,22,31]:
        T,SignalIn,stateHistory = \
        GenSignal.createScenarioFromSpirentFile(SVNum,CNO)
        coherentIntegrationTime = 4
        numOutputSamples = T.size/coherentIntegrationTime

        print('Running Reciever')
        Z,DcoFreqArray,LockStateValues,PhaseError = \
        runReceiver(T,SignalIn,18,FLLBW)
        PolarisFreq = DcoFreqArray[500:1000]
        plt.plot(PolarisFreq,'r')

        plt.ylabel('Doppler Shift (Hz)')
        plt.xlabel('Time (s)')
        plt.ylim(-4,4)
        plt.title('Bn=18Hz')
        plt.savefig(str(SVNum)+'18Polaris.eps',format='eps', dpi=1000)
        plt.close()


        Z,DcoFreqArray,LockStateValues,PhaseError = \
        runReceiver(T,SignalIn,32,FLLBW)
        PolarisFreq = DcoFreqArray[500:1000]
        plt.plot(PolarisFreq,'r')
        plt.ylim(-4,4)
        plt.ylabel('Doppler Shift (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Bn=32Hz')
        
        plt.savefig(str(SVNum)+'32Polaris.eps',format='eps', dpi=1000)
        plt.close()

def performAnalysisSingleSetting():
    PLLBW = 32
    FLLBW = 0
    CNO = 45
    
    t,SignalIn,stateHistory = GenSignal.constantSnap(1,CNO,200)
    Z,DcoFreqArray,LockStateValues,PhaseError = \
    runReceiver(t,SignalIn,PLLBW,FLLBW)
    LockStateValues = np.where(LockStateValues == 3, 1, 0)
    breakingPoint = 200*(np.mean(LockStateValues)) #m/s^2
    
    plt.title('Doppler frequency')
    t = 0.004*np.arange(0,DcoFreqArray.size)
    plt.plot(t,DcoFreqArray,color='k')
    plt.xlabel('Time (s)')
    plt.ylabel('Doppler frequency (Hz)')
    plt.savefig('DcoFreqArray.eps', format='eps', dpi=1000)
        
    plt.close()
    
    plt.title('PLL lock state')
    t = 0.004*np.arange(0,LockStateValues.size)
    print(breakingPoint)            

    plt.plot(t,LockStateValues,color='k')
    plt.fill_between(t,LockStateValues,color='r',alpha=0.5)

    plt.xlabel('Time (s)')
    plt.ylabel('Phase lock')
    plt.ylim(0,1.5)
    plt.savefig('LockState.eps', format='eps', dpi=1000)
    plt.close()

    t = 0.004*np.arange(0,PhaseError.size)

    plt.title('Phase Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Phase error (degrees)')
    plt.plot(t,360*PhaseError,color='k',alpha=0.5)
    plt.savefig('PhaseError.eps', format='eps', dpi=1000)
    plt.close()
    
    plt.title('Phase error standard deviation')    
    stdList =[]
    for i in range(0,PhaseError.size):
        stdList.append((360*PhaseError[i:i+100]).std())

    t = 0.004*np.arange(0,len(stdList))
    
    plt.plot(t,stdList,color='k',alpha=0.5)
    plt.grid()
    plt.ylabel('Window standard deviation (degrees)')
    plt.xlabel('Time (s)')
    plt.savefig('PhaseErrorSTD.eps', format='eps', dpi=1000)
    plt.close()

def monteCarloAnalysis():
    PLLBW = 32
    FLLBW = 10
    CNO = 45
    
    breakingPoints1 = np.zeros((100))
    averagePhaseJitter1 = np.zeros((100))
    for i in range(0,100):
        t,SignalIn,stateHistory = GenSignal.constantSnap(1,CNO,200)
        Z,DcoFreqArray,LockStateValues,PhaseError = \
        runReceiver(t,SignalIn,PLLBW,FLLBW)
        LockStateValues = np.where(LockStateValues == 3, 1, 0)
        breakingPoint = 200*(np.mean(LockStateValues)) #m/s^2
        breakingPoints1[i] = breakingPoint
        stdArray = np.zeros((250*70))
        for j in range(0,250*70):
            stdArray[j] = (360*PhaseError[j:j+100]).std()
        averagePhaseJitter1[i] = stdArray.mean()
        print(i,stdArray.mean())

    print('A')
    print(np.mean(averagePhaseJitter1),np.std(averagePhaseJitter1)) 

    PLLBW = 32
    FLLBW = 0
    CNO = 45
    
    breakingPoints2 = np.zeros((100))
    averagePhaseJitter2 = np.zeros((100))
    for i in range(0,100):
        t,SignalIn,stateHistory = GenSignal.constantSnap(1,CNO,200)
        Z,DcoFreqArray,LockStateValues,PhaseError = \
        runReceiver(t,SignalIn,PLLBW,FLLBW)
        LockStateValues = np.where(LockStateValues == 3, 1, 0)
        breakingPoint = 200*(np.mean(LockStateValues)) #m/s^2
        breakingPoints2[i] = breakingPoint
        stdArray = np.zeros((250*70))
        for j in range(0,250*70):
            stdArray[j] = (360*PhaseError[j:j+100]).std()
        averagePhaseJitter2[i] = stdArray.mean()
        print(i,stdArray.mean())

    print('B')
    print(np.mean(averagePhaseJitter2),np.std(averagePhaseJitter2)) 

    ## Create data
    np.random.seed(10)
    
    ## combine these different collections into a list    
    data_to_plot = [averagePhaseJitter1, averagePhaseJitter2]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create the boxplot
    bp = plt.boxplot(data_to_plot)
    plt.setp(bp['boxes'], color='k')
    plt.setp(bp['whiskers'], color='k')
    plt.setp(bp['fliers'], color='k', marker='+')
    plt.setp(bp['medians'], color='k')
    
    plt.title('Average phase jitter jitter')
    plt.ylabel('Average phase jitter $(m/s^3)$')
    plt.savefig('BoxplotPhaseJitter.eps', format='eps', dpi=1000)
   
if __name__ == '__main__':
    monteCarloAnalysis()   
