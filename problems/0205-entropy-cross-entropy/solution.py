import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
	m = P > 0 # boolean mask that is True wherever P is nonzero. Indexing with P[m] keeps only those entries, which applies the 0·log 0 = 0 convention.
	entropy = -np.sum(P[m] * np.log(P[m]))
	cross_entropy = -np.sum(P[m] * np.log(Q[m]))
	return float(entropy), float(cross_entropy)
	pass