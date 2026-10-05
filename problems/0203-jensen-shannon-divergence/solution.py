import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
	"""
	Compute the Jensen-Shannon Divergence between two probability distributions.
	
	Args:
		P: First probability distribution
		Q: Second probability distribution
	
	Returns:
		Jensen-Shannon Divergence value
	"""
	# Your code here
	P,Q = np.asarray(P, float ), np.asarray(Q,float)
	M = 0.5 * (P+Q)
	DKLP =0 
	DKLQ =0
	for i,j in zip(P,M):
		if i>0:
			DKLP = DKLP + i*(np.log(i/j))
	for i,j in zip(Q,M):
		if i>0:
			DKLQ = DKLQ + i*(np.log(i/j))
	JSD = 0.5*(DKLP + DKLQ)
	return JSD
	pass