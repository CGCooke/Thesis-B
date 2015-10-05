ReTerm = CurrentISample*PrevISample + CurrentQSample*PrevQSample
ImTerm = CurrentQSample*PrevISample - CurrentISample*PrevQSample

#Delta phase is possibly Phase2Quad
Phase4Quad,DeltaPhase = ATan2Cycles(ImTerm, ReTerm)

#Limiting phase to +/- 90 degrees
DeltaPhase = limitPhaseAngle(DeltaPhase)
