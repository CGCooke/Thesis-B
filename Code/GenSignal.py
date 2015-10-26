import matplotlib.pyplot as plt
import numpy as np
import Scenario

class motion():
    def __init__(self,Time,targetAcceleration,Jerk):
        self.State = 0
        self.Time = Time
        self.targetAcceleration =targetAcceleration
        self.Jerk = Jerk

def GenerateTrajectory(TStart, TEnd, initialState, Manoeuvres,dT=0.001):
    manoeuvereNumber = 0
    decayConstant = 1
    
    #Set up initial state
    X = initialState
    sampleTimes = np.arange(TStart,TEnd,dT)
    Y = np.zeros((sampleTimes.size))
    
    stateHistory = np.zeros((sampleTimes.size,3))
    for i in range(0,sampleTimes.size):
        #Check whether next manoeuvre is ready to be enacted. 
        currentManoeuvre = Manoeuvres[manoeuvereNumber]
        if sampleTimes[i]>currentManoeuvre.Time:
            if (manoeuvereNumber<len(Manoeuvres)-1):
                manoeuvereNumber+=1
        
        if X[2]<currentManoeuvre.targetAcceleration:        
            if abs(X[2]-currentManoeuvre.targetAcceleration)<0.2:
                decayConstant*=0.5
                X[2] += currentManoeuvre.Jerk*dT*decayConstant
            else:
                X[2] += currentManoeuvre.Jerk*dT
                decayConstant = 1
        else: 
            if abs(X[2]-currentManoeuvre.targetAcceleration)<0.2:
                decayConstant*=0.5
                X[2] -= currentManoeuvre.Jerk*dT*decayConstant
            else:
                X[2] -= currentManoeuvre.Jerk*dT
                decayConstant = 1

        #Velocity += Acceleration x delta time
        X[1] = X[1] + X[2]*dT
        #Position += Velocity x delta time
        X[0] = X[0] + X[1]*dT
        #Save position    
        Y[i] = X[0]
        
        stateHistory[i,:] = X
    return(sampleTimes,stateHistory)
    
def GenerateSignal(T,stateHistory,CNO=41,Noise=True,seedNum=1337):
    L1Freq = 1.57542*10**9 #Hz
    SpeedOfLight = 2.99792*10**8 #M/s
    CyclesPerMeter = L1Freq/SpeedOfLight

    Y = stateHistory[:,0]

    SignalPhase = Y*CyclesPerMeter  #m to cycles
    Amp0 = np.sqrt((10.0**(CNO/10.0))/1000.0)
    
    #np.random.seed(seedNum)
    #Generating noise
    if(Noise==True):
        NoiseIn = np.random.randn(SignalPhase.size) + 1j*np.random.randn(SignalPhase.size)
    else:
        NoiseIn = np.zeros(SignalPhase.size)
    
    #Adding noise to signal
    SignalIn = Amp0*np.exp(1j*2*np.pi*SignalPhase)+ NoiseIn/np.sqrt(2.0)
    
    BitAlignOS = 0
    for i in np.arange(0,T.size-40,40):
        SignalIn[BitAlignOS+i:BitAlignOS+i+19] = -SignalIn[BitAlignOS+i:BitAlignOS+i+19]

    return(SignalIn)
    
def createStepAccelerationScenario(CNO):
    motionList = []
    Jerk = 100
    
    Time = 0
    targetAcceleration = 0
    for i in range(0,10):
        Time+=10
        motionList.append(motion(Time,targetAcceleration,Jerk))
        targetAcceleration+=10
        
    TMin = 0
    TMax = 100
    initialState = [0, 0, 0]

    T,stateHistory = GenerateTrajectory(TMin, TMax, initialState, motionList)
    SignalIn = GenerateSignal(T,stateHistory,CNO)
    return(T,SignalIn,stateHistory)

def createScenarioFromSpirentFile(SVNum,CNO=38,Quantize=False):
    Jerk = 100
    dT = 0.001
    satelliteDict = Scenario.readSimulatorMotionFile()
    motionList = []
   
    Time,losX,losY,losZ = satelliteDict[SVNum]
    Time/=1000
    pseudorange = np.sqrt(losX**2+losY**2+losZ**2)
    velocity = np.diff(pseudorange)
    velocity-=velocity[0]
    acceleration = np.diff(velocity)

    pseudorange = pseudorange[:-2]
    velocity = velocity[:-1]
    T = Time[:-2]


    for i in range(0,acceleration.size):
        time = Time[i]
        targetAcceleration = acceleration[i]
        motionList.append(motion(time,targetAcceleration,Jerk))
        
        
    TMin = Time[0]
    TMax = Time[-1]
    initialState = [0, 0, acceleration[0]]
    T,stateHistory = GenerateTrajectory(TMin, TMax, initialState, motionList,dT)
    if Quantize==True:
        stateHistory = quantizeSpirentStateHistory(T,stateHistory)
    SignalIn = GenerateSignal(T,stateHistory,CNO)
    return(T,SignalIn,stateHistory)

def quantizeSpirentStateHistory(T,stateHistory):
    #Quantizing State history in time
    plt.plot(stateHistory[:,1],label='Before')
    velocity = stateHistory[:,1]
    for i in range(0,T.size):
        if(i%2)==0:
            v = velocity[i]
        else:
            velocity[i] = v
    
    pseudorange = np.cumsum(velocity)
    stateHistory[:,0] = pseudorange
    stateHistory[:,1] = velocity
    plt.plot(stateHistory[:,1],label='After')
    plt.legend() 
    plt.show()
    return(stateHistory)

def constantAcceleration(a,CNO,numSeconds):
    t = np.linspace(0,numSeconds,1000*numSeconds)
    v = a*t
    y = a*t**2/2.0

    a = a*np.ones((1000*numSeconds))
    stateHistory = np.vstack((y,v,a)).T
    SignalIn = GenerateSignal(t,stateHistory,CNO)
    return(t,SignalIn,stateHistory)

def constantJerk(jerk,CNO,numSeconds):
    t = np.linspace(0,numSeconds,1000*numSeconds)
    a = jerk*t
    v = jerk*t**2/2.0
    y = jerk*t**3/6.0    
    stateHistory = np.vstack((y,v,a)).T
    SignalIn = GenerateSignal(t,stateHistory,CNO)
    return(t,SignalIn,stateHistory)

def constantSnap(snap,CNO,numSeconds):
    t = np.linspace(0,numSeconds,1000*numSeconds)
    j = snap*t
    a = snap*t**2/2.0
    v = snap*t**3/6.0
    y = snap*t**4/24.0
    stateHistory = np.vstack((y,v,a)).T
    SignalIn = GenerateSignal(t,stateHistory,CNO)
    return(t,SignalIn,stateHistory)

if __name__ == '__main__':
    pass




