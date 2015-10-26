import math
import numpy as np

class InitGpsSignal():    
	# Initialise the global structure GpsSignal
	def __init__(self,PLLBW,FLLBW):
		#This is used to toggle sample skipping.
		self.SkipFreqSample = True

		self.Mod2 = False
		self.CurrentISample = 0
		self.CurrentQSample = 0
		self.PrevISample = 0
		self.PrevQSample = 0

		self.FrequencyEstimate  = 0 #Estimate of frequency \
		#based on the change in phase between two samples, \
		#divided by the time difference. 
		self.FreqHzFiltered = 0
		self.SumFreqHz = 0 #Integral of DeltaFreqHz, \
		#it's an Integral of the estimate of the frequency,\
		#(Phase)
		self.AbsFreqHzFiltered = 0
		
		self.Phase = 0
		self.Phase2Quadrant = 0
		self.DeltaPhase2Quadrant = 0
		self.SumPhase2Quadrant = 0
		
		self.FLLCount = 0
		self.PLLCount = 0
		self.TrackMode = 0
		self.PhaseWithinLimits = 0
		self.PhaseWithinLimits = []
		self.coherentIntegrationTime = 4 #ms


		self.FLLBW = FLLBW
		self.PLLBW = PLLBW

		self.CarrierDcoFreq = 80
		self.DopplerHistHzS = [self.CarrierDcoFreq,\
		self.CarrierDcoFreq,self.CarrierDcoFreq]

def TrackingLoop(Prompt,GpsSignal):
	GpsSignal.CurrentISample = np.real(Prompt)
	GpsSignal.CurrentQSample = np.imag(Prompt)

	#The carrier tracking loop requires pairs
	#of samples in order to allow an FLL assisted
	#PLL to be implemented while using independent
	#frequency and phase estimates.  This is 
	#handled through the use of the 'Mod2' flag
	#that is toggled as each sample is received.
	
	cNoLock = 0
	cCodeLock = 1
	cFreqLock = 2
	
	if (GpsSignal.SkipFreqSample==False):
		GpsSignal.Mod2=True

	if (GpsSignal.Mod2==True):
		Phase4Quadrant,Phase2Quadrant = FLLDiscriminator(GpsSignal)
		
		#Frequency proportional term
		if (GpsSignal.FLLCount>16):
			GpsSignal.FrequencyEstimate = \
			Phase2Quadrant/(GpsSignal.\
			coherentIntegrationTime*0.001)
		else:
			GpsSignal.FrequencyEstimate = \
			Phase4Quadrant/(GpsSignal.\
			coherentIntegrationTime*0.001)
		
		if (GpsSignal.FLLCount>=16):          # Settling time
			#Frequency Integral term
			GpsSignal.SumFreqHz +=  GpsSignal.FrequencyEstimate

		# Update filtered absolute frequency error
		#that is used to determine when the FLL is
		#locked, where the filter has been designed 
		#to avoid a long startup transient.         

		computeAbsFreqHzFiltered(GpsSignal)

		Phase2Quadrant = PLLDiscriminator(GpsSignal)
		
		#Phase derivative term
		DeltaPhase2Quadrant = Phase2Quadrant - GpsSignal.Phase2Quadrant
		DeltaPhase2Quadrant = limitPhaseAngle(DeltaPhase2Quadrant)
		GpsSignal.DeltaPhase2Quadrant = DeltaPhase2Quadrant
		
		#Phase proportional term
		GpsSignal.Phase2Quadrant = Phase2Quadrant
		
		#Phase Integral term
		GpsSignal.SumPhase2Quadrant += Phase2Quadrant
		
		# Zero out various integration quantities if these quantities don't
		# make sense given the current tracking status.

		if (GpsSignal.TrackMode==cNoLock):
			GpsSignal.SumPhase2Quadrant = 0
			GpsSignal.SumFreqHz = 0
		elif (GpsSignal.TrackMode==cCodeLock):
			GpsSignal.SumPhase2Quadrant = 0

		GpsSignal.DcoFreq = GpsSignal.DopplerHistHzS[2]
		
		LoopFilters(GpsSignal)

	if (GpsSignal.Mod2==True):
		GpsSignal.Mod2 = False
	else:
		GpsSignal.Mod2 = True
	
	# Update copies of 'previous samples' for the next iteration.
	GpsSignal.PrevISample = GpsSignal.CurrentISample
	GpsSignal.PrevQSample = GpsSignal.CurrentQSample
	 
	DetermineTrackingMode(GpsSignal)

def FLLDiscriminator(GpsSignal):
	#FLL Discriminator:
	#Calculate 'DeltaPhase' (frequency).
	#To do this employ an 'atan(Cross,Dot)' frequency estimator.
	#See Table 5.4, Page 171 of Kaplan
	
	dot = (GpsSignal.CurrentISample*GpsSignal.PrevISample)\
	+ (GpsSignal.CurrentQSample*GpsSignal.PrevQSample)
	cross = (GpsSignal.CurrentQSample*GpsSignal.PrevISample)\
	- (GpsSignal.CurrentISample*GpsSignal.PrevQSample)
	Phase4Quadrant,Phase2Quadrant = ATanCycles(cross,dot)
	return(Phase4Quadrant,Phase2Quadrant)

def PLLDiscriminator(GpsSignal):
	# Estimate noisy phase for the PLL loop.
	# The phase estimate is calculated every
	# second coherent integration period using
	# the decision directed coherent sum
	# of the two previous samples.
	
	#This is equivilant to Z_k*conj(Z_(k-1)), and tests if 
	#the vectors are less than +/- 90 degrees apart
	
	dot = (GpsSignal.CurrentISample*GpsSignal.PrevISample)\
	+ (GpsSignal.CurrentQSample*GpsSignal.PrevQSample)
	
	if (dot>0):
		ReTerm = GpsSignal.PrevISample + GpsSignal.CurrentISample
		ImTerm = GpsSignal.PrevQSample + GpsSignal.CurrentQSample
	else:
		ReTerm = GpsSignal.PrevISample - GpsSignal.CurrentISample
		ImTerm = GpsSignal.PrevQSample - GpsSignal.CurrentQSample
	
	#PLL/Costas Discriminator, see Table 5.2, page 168 Kaplan
	Phase4Quadrant,Phase2Quadrant = ATanCycles(ImTerm, ReTerm)	
	return(Phase2Quadrant)

def ATanCycles(ValueY,ValueX):
	# Returns the results in units of cycles
	if (ValueX !=0):
		ATan4Quadrant = math.atan2(ValueY,ValueX)/(2*math.pi)
		ATan2Quadrant = math.atan(ValueY/ValueX)/(2*math.pi)
	elif (ValueY>0 ):
		ATan4Quadrant = 0.25
		ATan2Quadrant = 0.25
	elif (ValueY<0):
		ATan4Quadrant = -0.25
		ATan2Quadrant = -0.25
	else:
		ATan4Quadrant = 0
		ATan2Quadrant = 0
	return(ATan4Quadrant,ATan2Quadrant)

def DetermineTrackingMode(GpsSignal):
	# DetermineTrackingMode
	# 'DetermineTrackingMode' is used to select the type of
	# tracking that the receiver should be trying to perform.
	# This is done by examining the various metrics that are
	# maintained by the receiver, including the filtered 'dot'
	# and 'cross' terms, as well the as the filtered value of
	# absolute residual phase.  These can then be used to 
	# determine whether frequency lock and phase lock have 
	# been achieved.

	# One danger with this approach is that if 'CodeLock' 
	# is lost for any reason, the carrier phase will be reset.
	# However, 'CodeLock' is determined by measuring the average
	# absolute tracking error and a micro-jump or activity dip on
	# the TCXO occurs, then a glitch in the tracking error may
	# well occur.  However, this need not be fatal, provided the
	# tracking loop can adapt sufficiently quickly and the 
	# filtering on 'AbsErrorChipsSFiltered' removes the worst of 
	# the tracking error.
	#
	# The lock indicator for the PLL in this code requires
	# that the maximum measured instantaneous phase error be
	# within 30 degrees 20 times within the last 32 measurements.
	# Note that this is quite a loose definition of phase lock
	# compared to the lock indicator described by Van Dierendonck.
	# The Van Dierendonck PLL lock-indicator requires that 
	# (I*I-Q*Q)/(I*I + Q*Q)>0.4, which can be shown to be 
	# equivalent to requiring that the instantaneous phase 
	# angle always be within 33 degrees.

	# Defines
	cCodeLock = 1
	cFreqLock = 2
	cPhaseLock = 3
 
	FreqErrorThreshold = 30 #Hz
	MinNumberMeasurements = 32 #counts
	NumSamplesWithinTolerence = 20 #At least this many 
	#samples must be within tolorence

	FreqLock = (GpsSignal.AbsFreqHzFiltered < FreqErrorThreshold)\
	 and (GpsSignal.FLLCount > MinNumberMeasurements)
	
	if abs(GpsSignal.Phase2Quadrant)<(1.0/12.0): #+/- 30 degrees
		GpsSignal.PhaseWithinLimits.append(1)
	else:
		GpsSignal.PhaseWithinLimits.append(0)

	if (len(GpsSignal.PhaseWithinLimits)>MinNumberMeasurements):
		GpsSignal.PhaseWithinLimits.pop(0)
		
	PhaseLt30Deg=(GpsSignal.PhaseWithinLimits.count(1)\
	 > NumSamplesWithinTolerence)
	
	PhaseLock = FreqLock and PhaseLt30Deg and \
	GpsSignal.PLLCount > MinNumberMeasurements

	GpsSignal.FLLCount += 1
	
	if (FreqLock==True):
		GpsSignal.PLLCount += 1
	else:
		GpsSignal.PLLCount = 0

	if(FreqLock==False):
		GpsSignal.TrackMode = cCodeLock
	elif (PhaseLock==False):
		GpsSignal.TrackMode = cFreqLock
	else:
		GpsSignal.TrackMode = cPhaseLock
		
def GetLoopParameters(LoopType,GpsSignal):
	# FLL and PLL coefficients for 2nd order FLL
	# and 3rd order PLL, both normalised for 1 Hz
	# noise bandwidth and a sample period of 1 ms. 
	# These values come straight out of Chapter 5 
	# of  'Understanding GPS: Principles and 
	# Applications:2nd Edition', although the 
	# coefficients have been normalised for unit 
	# noise bandwidth and unit predetection integration 
	# (assuming use of 'paired' samples, so consecutive p
	# airs are used to estimate frequency and the 
	# decision directed sum of those pairs the phase).
	# This explains the factor of 2 in (2*0.001), where 
	# the 0.001 indicates 1 ms.

	PLLBW = GpsSignal.PLLBW
	FLLBW = GpsSignal.FLLBW
	LoopCoef = np.zeros(3)

	# Defines
	cCodeLock = 1
	cFreqLock = 2
	cPhaseLock = 3

	a2 = 1.41421
	a3 = 1.1
	b3 = 2.4

	T = (2*0.001)

	FLL1Bn = 5
	FLL1C1 = T/0.25

	PLL2Bn = 10
	PLL2C0 = a2/0.53
	PLL2C1 = T/(0.53)**2

	FLL2C1 = T*a2
	FLL2C2 = T**2/(0.53**2)

	PLL3C0 = b3/0.7845
	PLL3C1 = (a3*T)/(0.7845**2)
	PLL3C2 = T**2/(0.7845**3)

	PLL3WithFLL2Assist = False
	PLL2WithFLL1Assist = False
	PureFLL = False   

	# Select tracking loop parameters
	if (GpsSignal.TrackMode==cCodeLock):
		PureFLL = True     
	elif (GpsSignal.TrackMode==cFreqLock):
		PLL2WithFLL1Assist = True
	elif(GpsSignal.TrackMode==cPhaseLock):
		if (GpsSignal.PLLCount<192):
			PLL2WithFLL1Assist = True
		else:
			PLL3WithFLL2Assist = True
		if (GpsSignal.PLLCount==192):
			GpsSignal.SumFreqHz = 0
			GpsSignal.SumPhase2Quadrant = 0

	# Select loop coefficients   
	if (PLL3WithFLL2Assist==True):
		if (LoopType==1): # FLL
			LoopCoef[0] = 0
			LoopCoef[1] = (FLLBW*FLL2C1)* \
			GpsSignal.coherentIntegrationTime
			LoopCoef[2] = (FLLBW**2 *FLL2C2)* \
			GpsSignal.coherentIntegrationTime**2
		elif (LoopType==2): # PLL
			LoopCoef[0] = (PLLBW*PLL3C0)
			LoopCoef[1] = (PLLBW**2 *PLL3C1)* \
			GpsSignal.coherentIntegrationTime
			LoopCoef[2] = (PLLBW**3 *PLL3C2)* \ 
			GpsSignal.coherentIntegrationTime**2
	elif (PLL2WithFLL1Assist==True):
		if (LoopType==1): # FLL
			LoopCoef[0] = 0
			LoopCoef[1] = (FLL1Bn*FLL1C1)
			LoopCoef[2] = 0
		elif (LoopType==2): # PLL
			LoopCoef[0] = (PLL2Bn*PLL2C0)
			LoopCoef[1] = (PLL2Bn**2 *PLL2C1)* \
			GpsSignal.coherentIntegrationTime
			LoopCoef[2] = 0
	elif (PureFLL==True):
		if (LoopType==1):
			# FLL parameters are: 0, 1/8, 0
			LoopCoef[0] = 0
			LoopCoef[1] = (0.125)
			LoopCoef[2] = 0
		elif (LoopType==2): 
			LoopCoef[0] = 0
			LoopCoef[1] = 0
			LoopCoef[2] = 0
	return(LoopCoef)

def limitPhaseAngle(phase):
	#Limiting phase to +/- 90 degrees
	if (phase > 0.25):
		phase -= 0.5
	elif (phase < -0.25):
		phase +=  0.5
	return(phase)

def computeAbsFreqHzFiltered(GpsSignal):
	#AbsFreqHzFiltered is used for determining code lock 
	AbsDeltaFreqHz = abs(GpsSignal.FrequencyEstimate)\
	-GpsSignal.AbsFreqHzFiltered

	if (GpsSignal.FLLCount<2):
		weightingFactor=1
	elif (GpsSignal.FLLCount<4):
		weightingFactor= 0.5
	elif (GpsSignal.FLLCount<16):
		weightingFactor= 0.25
	elif (GpsSignal.FLLCount<64):
		weightingFactor= 0.125
	else:
		weightingFactor= 0.0625
	GpsSignal.AbsFreqHzFiltered += weightingFactor* \
	(abs(GpsSignal.FrequencyEstimate)- \
	GpsSignal.AbsFreqHzFiltered)

def LoopFilters(GpsSignal):
	DcoFreq = GpsSignal.DcoFreq

	#FLL
	LoopCoef = GetLoopParameters(1, GpsSignal)
	ProportionalTermFLL  = (LoopCoef[1] \ 
	*GpsSignal.FrequencyEstimate)#Proportional
	IntegralTermFLL = (LoopCoef[2] \
	*GpsSignal.SumFreqHz) #Integral

	#PLL
	LoopCoef = GetLoopParameters(2,GpsSignal)
	DifferentialTermPLL = (LoopCoef[0] \
	*GpsSignal.DeltaPhase2Quadrant) #Differential
	ProportionalTermPLL = (LoopCoef[1] \ 
	*GpsSignal.Phase2Quadrant) #Proportional 
	IntegralTermPLL = (LoopCoef[2] \ 
	*GpsSignal.SumPhase2Quadrant) #Integral

	#Integrating up
	DcoFreq +=  ProportionalTermFLL  \
	+ IntegralTermFLL \
	+  DifferentialTermPLL \
	+ ProportionalTermPLL \
	+ IntegralTermPLL
	
	GpsSignal.DopplerHistHzS[0] = DcoFreq

