package core

import "math"

// SquaredEuclideanDistance computes ||a - b||² between two vectors.
// Returns +Inf if dimensions do not match.
func SquaredEuclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// ComputeEmbeddingScore returns log(price) - distance²/σ².
// If bidEmbedding is nil/empty or sigma is 0, returns log(price) (pure price ranking).
func ComputeEmbeddingScore(price float64, bidEmbedding []float64, sigma float64, queryEmbedding []float64) float64 {
	logPrice := math.Log(price)
	if len(bidEmbedding) == 0 || len(queryEmbedding) == 0 || sigma == 0 {
		return logPrice
	}
	dist2 := SquaredEuclideanDistance(bidEmbedding, queryEmbedding)
	return logPrice - dist2/(sigma*sigma)
}

// HasEmbedding returns true if the bid carries a non-empty embedding vector.
func HasEmbedding(bid *CoreBid) bool {
	return len(bid.Embedding) > 0
}
