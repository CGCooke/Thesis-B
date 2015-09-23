import math
import numpy as np
import matplotlib.pyplot as plt

from generateCAcode import generateCAcode
import getSamples

class State:
    def __init__(self,settings,channel):
        #Filter coefficient values
        (settings.C1code, settings.C2code) = calcLoopCoef(settings.dllNoiseBandwidth,
        settings.dllDampingRatio,settings.dllLoopGain)
        ##PLL Variables##
        (settings.C1carr,settings.C2carr) = calcLoopCoef(settings.pllNoiseBandwidth,
        settings.pllDampingRatio,settings.pllLoopGain)

        self.sampleIndex=0
        #Initialize phases and frequencies
        self.codeFreq = settings.codeFreqBasis
        self.remCodePhase = 0.0 #residual code phase
        self.carrFreq = channel.acquiredFreq
        self.remCarrPhase = 0.0 #residual carrier phase

        #code tracking loop parameters
        self.oldCodeNco = 0.0
        self.oldCodeError = 0.0
        #carrier/Costas loop parameters
        self.oldCarrNco = 0.0
        self.oldCarrError = 0.0

def calcLoopCoef(LBW, zeta, k):
    #See Spillker for more information
    #LBW           - Loop noise bandwidth 
    #zeta          - Damping ratio 
    #k             - Loop gain 
    #C1, C2   - Loop filter coefficients  

    Wn = LBW*8*zeta / (4*zeta**2 + 1)
    C1 = k / (Wn**2)
    C2 = 2.0* zeta / Wn
    return (C1, C2)

def correlate(state,settings,rawSignal,caCode):
    #Update the code phase rate based on 
    #code freq and sampling freq
    codePhaseStep = round(state.codeFreq/
    settings.samplingFreq,12)
    blksize = int(np.ceil((settings.CACodeLength - state.remCodePhase)/
    codePhaseStep))

    #Read samples for this integration period
    rawSignalSubsection = rawSignal[state.sampleIndex:
    (state.sampleIndex+blksize)]
    state.sampleIndex = state.sampleIndex + blksize

    earlyLateSpc = settings.dllCorrelatorSpacing

    #Define index into early code vector
    tcode = np.arange((state.remCodePhase-earlyLateSpc),
    (blksize*codePhaseStep+state.remCodePhase-earlyLateSpc),codePhaseStep)
    earlyCode = caCode[np.int_(np.ceil(tcode))]
    
    #Define index into late code vector
    tcode = np.arange((state.remCodePhase+earlyLateSpc),
    (blksize*codePhaseStep+state.remCodePhase+earlyLateSpc),codePhaseStep)
    lateCode = caCode[np.int_(np.ceil(tcode))]
    
    #Define index into prompt code vector
    tcode = np.arange(state.remCodePhase,
    (blksize*codePhaseStep+state.remCodePhase),codePhaseStep)
    promptCode = caCode[np.int_(np.ceil(tcode))]   

    state.remCodePhase = (tcode[blksize-1] + codePhaseStep) -
    settings.CACodeLength

    #Generate the carrier frequency to
    #mix the signal to baseband
    Time = np.arange(0,blksize+1) / 
    settings.samplingFreq #(seconds)

    #Get the argument to sin/cos functions
    trigarg = (state.carrFreq * 2.0 * math.pi)*
    Time + state.remCarrPhase

    state.remCarrPhase = np.remainder(trigarg[blksize],
    (2*math.pi))

    #Finally compute the signal to mix 
    #the collected data to baseband
    carrCos = np.cos(trigarg[0:blksize])
    carrSin = np.sin(trigarg[0:blksize])

    #Mix signals to baseband
    qBasebandSignal = carrCos*rawSignalSubsection
    iBasebandSignal = carrSin*rawSignalSubsection

    #Get early, prompt, and late I/Q correlations
    state.I_E = np.sum(earlyCode * iBasebandSignal)
    state.Q_E = np.sum(earlyCode * qBasebandSignal)
    state.I_P = np.sum(promptCode * iBasebandSignal)
    state.Q_P = np.sum(promptCode * qBasebandSignal)
    state.I_L = np.sum(lateCode * iBasebandSignal)
    state.Q_L = np.sum(lateCode * qBasebandSignal)
    return(state)

def computeCarrierState(state,settings,channel):
    #Find PLL error and update carrier NCO
    #Carrier loop discriminator (phase detector)
    #See Spilker
    carrError = math.atan(state.Q_P/state.I_P) / (2.0 * math.pi)
    #Carrier loop filter and NCO
    carrNco = state.oldCarrNco + (settings.C2carr/settings.C1carr)
    * (carrError-state.oldCarrError) + carrError*(settings.PDICarr/settings.C1carr)
    state.oldCarrNco = carrNco
    state.oldCarrError = carrError
    #Modify carrier freq based on NCO
    state.carrFreq = channel.acquiredFreq + carrNco
    return(state)

def computeCodeState(state,settings):
    #See Spilker
    #Find DLL error and update code NCO
    #Code Error discriminator
    codeError = (math.sqrt(state.I_E**2+ state.Q_E**2) - math.sqrt(state.I_L**2 + state.Q_L**2)) /
    (math.sqrt(state.I_E**2 + state.Q_E**2) + math.sqrt(state.I_L**2 + state.Q_L**2))
    codeNco = state.oldCodeNco + (settings.C2code/settings.C1code)*
    (codeError-state.oldCodeError) + codeError*(settings.PDICode/settings.C1code)
    state.oldCodeNco = codeNco
    state.oldCodeError = codeError
    #Code freq based on NCO
    state.codeFreq = settings.codeFreqBasis - codeNco
    return(state)

def track(channel, settings):
    state = State(settings,channel)
    data = np.zeros(settings.msToProcess)
    caCode = np.array(generateCAcode(channel.PRN))
    caCode = np.concatenate(([caCode[1022]],caCode,[caCode[0]]))

    #number of samples to seek ahead in file
    numSamplesToSkip = settings.startTime*
    settings.samplingFreq + channel.codePhase
    rawSignal = getSamples.readInData(int(settings.samplingFreq*settings.msToProcess/1000.0)
    +1000,numSamplesToSkip,settings.signalToTrack,settings)
    
    #Process the specified number of ms
    for loopCnt in range(settings.msToProcess):
        state = correlate(state,settings,rawSignal,caCode)
        state = computeCarrierState(state,settings,channel)
        state = computeCodeState(state,settings)
        
        if state.I_P>0:
            state.I_P=1
        else:
            state.I_P=-1
        data[loopCnt] = state.I_P
        
    plt.plot(data)
    plt.ylim(-2,2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Data Value')
    plt.title('Digital Data')
    plt.savefig(str(channel.PRN+1)+'Digital.png')
    plt.close()

    