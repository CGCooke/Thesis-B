def ATanCycles(ValueY,ValueX):
	# returning the results in units of cycles
	if (ValueX !=0):
		ATan4Quadrent = math.atan2(ValueY,ValueX)/(2*math.pi)
		ATan2Quadrent = math.atan(ValueY/ValueX)/(2*math.pi)
	elif (ValueY>0 ):
		ATan4Quadrent = 0.25
		ATan2Quadrent = 0.25
	elif (ValueY<0):
		ATan4Quadrent = -0.25
		ATan2Quadrent = -0.25
	else:
		ATan4Quadrent = 0
		ATan2Quadrent = 0
	return(ATan4Quadrent,ATan2Quadrent)