package core

import (
	"math"
	"testing"

	"github.com/peterldowns/testy/check"
)

func TestSquaredEuclideanDistance_IdenticalVectors(t *testing.T) {
	a := []float64{1.0, 2.0, 3.0}
	check.Equal(t, 0.0, SquaredEuclideanDistance(a, a))
}

func TestSquaredEuclideanDistance_KnownVectors(t *testing.T) {
	a := []float64{1.0, 0.0}
	b := []float64{0.0, 1.0}
	// (1-0)² + (0-1)² = 2.0
	check.Equal(t, 2.0, SquaredEuclideanDistance(a, b))
}

func TestSquaredEuclideanDistance_DimensionMismatch(t *testing.T) {
	a := []float64{1.0, 2.0}
	b := []float64{1.0, 2.0, 3.0}
	check.True(t, math.IsInf(SquaredEuclideanDistance(a, b), 1))
}

func TestComputeEmbeddingScore_NoEmbedding(t *testing.T) {
	// No embedding → pure log(price)
	score := ComputeEmbeddingScore(10.0, nil, 1.0, []float64{0.0, 0.0})
	check.Equal(t, math.Log(10.0), score)
}

func TestComputeEmbeddingScore_NoQueryEmbedding(t *testing.T) {
	// No query embedding → pure log(price)
	score := ComputeEmbeddingScore(10.0, []float64{1.0, 0.0}, 1.0, nil)
	check.Equal(t, math.Log(10.0), score)
}

func TestComputeEmbeddingScore_SigmaZero(t *testing.T) {
	// σ=0 → pure log(price)
	score := ComputeEmbeddingScore(10.0, []float64{1.0, 0.0}, 0.0, []float64{0.0, 0.0})
	check.Equal(t, math.Log(10.0), score)
}

func TestComputeEmbeddingScore_WithEmbedding(t *testing.T) {
	// bid at [1,0], query at [0,0], σ=1
	// distance² = 1, score = log(10) - 1/1 = log(10) - 1
	price := 10.0
	score := ComputeEmbeddingScore(price, []float64{1.0, 0.0}, 1.0, []float64{0.0, 0.0})
	expected := math.Log(10.0) - 1.0
	check.True(t, math.Abs(score-expected) < 1e-12)
}

func TestComputeEmbeddingScore_BlogExample_NikeVsPeloton(t *testing.T) {
	// From the blog series: Nike bid $4 at [0.8, 0.2], Peloton bid $3 at [0.3, 0.7]
	// Query point at [0.5, 0.5], σ = 0.5
	query := []float64{0.5, 0.5}
	sigma := 0.5

	// Nike: distance² = (0.8-0.5)² + (0.2-0.5)² = 0.09 + 0.09 = 0.18
	// score = log(4) - 0.18/0.25 = 1.3863 - 0.72 = 0.6663
	nikeScore := ComputeEmbeddingScore(4.0, []float64{0.8, 0.2}, sigma, query)

	// Peloton: distance² = (0.3-0.5)² + (0.7-0.5)² = 0.04 + 0.04 = 0.08
	// score = log(3) - 0.08/0.25 = 1.0986 - 0.32 = 0.7786
	pelotonScore := ComputeEmbeddingScore(3.0, []float64{0.3, 0.7}, sigma, query)

	// Peloton wins despite lower price — closer to the query
	check.True(t, pelotonScore > nikeScore)

	// Verify against hand-computed values
	expectedNike := math.Log(4.0) - 0.18/0.25
	expectedPeloton := math.Log(3.0) - 0.08/0.25
	check.True(t, math.Abs(nikeScore-expectedNike) < 1e-12)
	check.True(t, math.Abs(pelotonScore-expectedPeloton) < 1e-12)
}

func TestHasEmbedding(t *testing.T) {
	check.True(t, HasEmbedding(&CoreBid{Embedding: []float64{1.0}}))
	check.False(t, HasEmbedding(&CoreBid{}))
	check.False(t, HasEmbedding(&CoreBid{Embedding: []float64{}}))
}
