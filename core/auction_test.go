package core

import (
	"testing"

	"github.com/peterldowns/testy/check"
)

func TestRunAuction_BasicFlow(t *testing.T) {
	// Test the complete auction flow with adjustment, floor enforcement, and ranking
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0},
		{ID: "bid2", Bidder: "bidder_b", Price: 1.5},
		{ID: "bid3", Bidder: "bidder_c", Price: 1.0},
	}

	adjustmentFactors := map[string]float64{
		"bidder_a": 1.0,
		"bidder_b": 1.2, // Boost bidder_b by 20%
		"bidder_c": 1.0,
	}

	bidFloor := 1.5 // bidder_c bid should fail floor

	result := RunAuction(bids, adjustmentFactors, bidFloor)

	// After adjustment: bidder_a=2.0, bidder_b=1.8, bidder_c=1.0
	// After floor enforcement: bidder_a=2.0, bidder_b=1.8 (bidder_c rejected)
	// Ranking: 1=bidder_a, 2=bidder_b

	check.NotNil(t, result)
	check.NotNil(t, result.Winner)
	check.NotNil(t, result.RunnerUp)

	// Verify winner (highest bid after adjustment)
	check.Equal(t, "bidder_a", result.Winner.Bidder)
	check.Equal(t, 2.0, result.Winner.Price)

	// Verify runner-up
	check.Equal(t, "bidder_b", result.RunnerUp.Bidder)
	check.Equal(t, 1.8, result.RunnerUp.Price)

	// Verify eligible bids (only bidder_a and bidder_b passed floor)
	check.Equal(t, 2, len(result.EligibleBids))

	// Verify rejected bids (bidder_c failed floor)
	check.Equal(t, 1, len(result.FloorRejectedBidIDs))
	check.Equal(t, "bid3", result.FloorRejectedBidIDs[0])
}

func TestRunAuction_NoBids(t *testing.T) {
	result := RunAuction([]CoreBid{}, nil, 0.0)

	check.NotNil(t, result)
	check.Nil(t, result.Winner)
	check.Nil(t, result.RunnerUp)
	check.Equal(t, 0, len(result.EligibleBids))
	check.Equal(t, 0, len(result.FloorRejectedBidIDs))
}

func TestRunAuction_SingleBid(t *testing.T) {
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0},
	}

	result := RunAuction(bids, nil, 0.0)

	check.NotNil(t, result)
	check.NotNil(t, result.Winner)
	check.Nil(t, result.RunnerUp) // Only one bid, no runner-up

	check.Equal(t, "bidder_a", result.Winner.Bidder)
	check.Equal(t, 2.0, result.Winner.Price)
}

func TestRunAuction_AllBidsRejectedByFloor(t *testing.T) {
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 1.0},
		{ID: "bid2", Bidder: "bidder_b", Price: 0.5},
	}

	bidFloor := 2.0 // Both bids below floor

	result := RunAuction(bids, nil, bidFloor)

	check.NotNil(t, result)
	check.Nil(t, result.Winner)
	check.Nil(t, result.RunnerUp)
	check.Equal(t, 0, len(result.EligibleBids))
	check.Equal(t, 2, len(result.FloorRejectedBidIDs))
}

func TestRunAuction_NoAdjustmentFactors(t *testing.T) {
	// Test that auction works without adjustment factors
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0},
		{ID: "bid2", Bidder: "bidder_b", Price: 1.5},
	}

	result := RunAuction(bids, nil, 0.0)

	check.NotNil(t, result)
	check.NotNil(t, result.Winner)

	// Without adjustments, original ranking is preserved
	check.Equal(t, "bidder_a", result.Winner.Bidder)
	check.Equal(t, 2.0, result.Winner.Price)

	// Verify runner-up
	check.NotNil(t, result.RunnerUp)
	check.Equal(t, "bidder_b", result.RunnerUp.Bidder)
	check.Equal(t, 1.5, result.RunnerUp.Price)
}

func TestRunAuction_NoFloors(t *testing.T) {
	// Test that auction works without floor enforcement
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0},
		{ID: "bid2", Bidder: "bidder_b", Price: 0.01}, // Very low bid
	}

	result := RunAuction(bids, nil, 0.0)

	check.NotNil(t, result)

	// Without floors, all bids are eligible
	check.Equal(t, 2, len(result.EligibleBids))
	check.Equal(t, 0, len(result.FloorRejectedBidIDs))
}

func TestRunAuction_AdjustmentChangesWinner(t *testing.T) {
	// Test that adjustment factors can change the auction winner
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0},
		{ID: "bid2", Bidder: "bidder_b", Price: 1.5},
	}

	adjustmentFactors := map[string]float64{
		"bidder_a": 1.0,
		"bidder_b": 1.5, // Boost bidder_b to 2.25
	}

	result := RunAuction(bids, adjustmentFactors, 0.0)

	check.NotNil(t, result)
	check.NotNil(t, result.Winner)

	// After adjustment, bidder_b should win (1.5 * 1.5 = 2.25 > 2.0)
	check.Equal(t, "bidder_b", result.Winner.Bidder)
	check.True(t, result.Winner.Price > 2.24 && result.Winner.Price < 2.26)

	check.Equal(t, "bidder_a", result.RunnerUp.Bidder)
	check.Equal(t, 2.0, result.RunnerUp.Price)
}

func TestRunAuction_PreservesOriginalBids(t *testing.T) {
	// Test that original bid slice is not modified
	originalBids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0},
	}

	adjustmentFactors := map[string]float64{
		"bidder_a": 2.0,
	}

	result := RunAuction(originalBids, adjustmentFactors, 0.0)

	check.NotNil(t, result)

	// Original bid should be unchanged
	check.Equal(t, 2.0, originalBids[0].Price)

	// Result should have adjusted price
	check.Equal(t, 4.0, result.Winner.Price)
}

func TestRunAuction_RejectsNegativePrices(t *testing.T) {
	// Test that negative prices are rejected during price validation
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0},
		{ID: "bid2", Bidder: "bidder_b", Price: -1.5},
	}

	result := RunAuction(bids, nil, 0.0)

	check.NotNil(t, result)
	check.NotNil(t, result.Winner)

	// Check eligible bids
	eligibleIDs := make(map[string]bool)
	for _, bid := range result.EligibleBids {
		eligibleIDs[bid.ID] = true
	}
	check.True(t, eligibleIDs["bid1"])
	check.False(t, eligibleIDs["bid2"])

	// Check rejected bids
	check.Equal(t, "bid2", result.PriceRejectedBidIDs[0])

	check.Equal(t, "bidder_a", result.Winner.Bidder)
	check.Nil(t, result.RunnerUp)
}

func TestRunAuction_RejectsZeroPrices(t *testing.T) {
	// Test that zero prices are rejected
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0},
		{ID: "bid2", Bidder: "bidder_b", Price: 0.0},
	}

	result := RunAuction(bids, nil, 0.0)

	check.NotNil(t, result)

	// Check eligible bids
	eligibleIDs := map[string]bool{}
	for _, bid := range result.EligibleBids {
		eligibleIDs[bid.ID] = true
	}
	check.True(t, eligibleIDs["bid1"])
	check.False(t, eligibleIDs["bid2"])

	// Check rejected bids
	check.Equal(t, "bid2", result.PriceRejectedBidIDs[0])

	check.Equal(t, "bidder_a", result.Winner.Bidder)
	check.Nil(t, result.RunnerUp)
}

func TestRunAuction_MixedPriceValidation(t *testing.T) {
	// Test combination of valid, negative, and zero price bids
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0},
		{ID: "bid2", Bidder: "bidder_b", Price: -0.5},
		{ID: "bid3", Bidder: "bidder_c", Price: 0.0},
		{ID: "bid4", Bidder: "bidder_d", Price: 0.0},
		{ID: "bid5", Bidder: "bidder_e", Price: 1.5},
	}

	result := RunAuction(bids, nil, 0.0)

	check.NotNil(t, result)

	// Check eligible bids
	eligibleIDs := map[string]bool{}
	for _, bid := range result.EligibleBids {
		eligibleIDs[bid.ID] = true
	}
	check.True(t, eligibleIDs["bid1"])
	check.True(t, eligibleIDs["bid5"])
	check.False(t, eligibleIDs["bid2"])
	check.False(t, eligibleIDs["bid3"])
	check.False(t, eligibleIDs["bid4"])

	// Check rejected bids
	rejectedIDs := map[string]bool{}
	for _, id := range result.PriceRejectedBidIDs {
		rejectedIDs[id] = true
	}
	check.True(t, rejectedIDs["bid2"])
	check.True(t, rejectedIDs["bid3"])
	check.True(t, rejectedIDs["bid4"])
}

// --- Embedding auction tests ---

func TestRunAuction_BackwardCompat_NilQuery(t *testing.T) {
	// No query embedding → identical to current behavior (pure price ranking)
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0},
		{ID: "bid2", Bidder: "bidder_b", Price: 3.0},
	}

	result := RunAuction(bids, nil, 0.0)

	check.NotNil(t, result)
	check.Equal(t, "bidder_b", result.Winner.Bidder)
	check.Equal(t, 3.0, result.Winner.Price)
	check.Equal(t, "bidder_a", result.RunnerUp.Bidder)
}

func TestRunAuction_CloserBidWins(t *testing.T) {
	// Same price, different distances → closer bid wins
	query := []float64{0.0, 0.0}
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_far", Price: 2.0, Embedding: []float64{3.0, 0.0}, Sigma: 1.0},
		{ID: "bid2", Bidder: "bidder_close", Price: 2.0, Embedding: []float64{0.1, 0.0}, Sigma: 1.0},
	}

	result := RunAuction(bids, nil, 0.0, query)

	check.NotNil(t, result)
	check.Equal(t, "bidder_close", result.Winner.Bidder)
	check.Equal(t, "bidder_far", result.RunnerUp.Bidder)
}

func TestRunAuction_PriceVsProximityTradeoff(t *testing.T) {
	// Expensive-far vs cheap-close: proximity wins when σ is small enough
	query := []float64{0.0, 0.0}
	bids := []CoreBid{
		{ID: "bid1", Bidder: "expensive_far", Price: 5.0, Embedding: []float64{2.0, 0.0}, Sigma: 0.5},
		{ID: "bid2", Bidder: "cheap_close", Price: 2.0, Embedding: []float64{0.1, 0.0}, Sigma: 0.5},
	}

	result := RunAuction(bids, nil, 0.0, query)

	check.NotNil(t, result)
	// With σ=0.5: expensive_far score = log(5) - 4/0.25 = 1.609 - 16 = -14.39
	//              cheap_close score = log(2) - 0.01/0.25 = 0.693 - 0.04 = 0.653
	check.Equal(t, "cheap_close", result.Winner.Bidder)
}

func TestRunAuction_SigmaZeroDegeneracy(t *testing.T) {
	// σ=0 with embeddings → falls back to pure price ranking
	query := []float64{0.0, 0.0}
	bids := []CoreBid{
		{ID: "bid1", Bidder: "bidder_a", Price: 2.0, Embedding: []float64{10.0, 10.0}, Sigma: 0.0},
		{ID: "bid2", Bidder: "bidder_b", Price: 3.0, Embedding: []float64{0.1, 0.1}, Sigma: 0.0},
	}

	result := RunAuction(bids, nil, 0.0, query)

	check.NotNil(t, result)
	// σ=0 means score = log(price), so higher price wins
	check.Equal(t, "bidder_b", result.Winner.Bidder)
	check.Equal(t, 3.0, result.Winner.Price)
}

func TestRunAuction_MixedEmbeddings(t *testing.T) {
	// Some bids with embeddings, some without → all participate
	query := []float64{0.0, 0.0}
	bids := []CoreBid{
		{ID: "bid1", Bidder: "with_embedding", Price: 2.0, Embedding: []float64{0.1, 0.0}, Sigma: 1.0},
		{ID: "bid2", Bidder: "without_embedding", Price: 3.0}, // No embedding → score = log(3)
	}

	result := RunAuction(bids, nil, 0.0, query)

	check.NotNil(t, result)
	// with_embedding score = log(2) - 0.01 = 0.683
	// without_embedding score = log(3) = 1.099
	// without_embedding wins on price alone since it has no distance penalty
	check.Equal(t, "without_embedding", result.Winner.Bidder)
}

func TestRunAuction_FloorAppliesToPrice_NotScore(t *testing.T) {
	// Floor enforcement is on price, not score
	query := []float64{0.0, 0.0}
	bids := []CoreBid{
		{ID: "bid1", Bidder: "above_floor", Price: 2.0, Embedding: []float64{0.0, 0.0}, Sigma: 1.0},
		{ID: "bid2", Bidder: "below_floor", Price: 0.5, Embedding: []float64{0.0, 0.0}, Sigma: 1.0},
	}

	result := RunAuction(bids, nil, 1.0, query) // floor = 1.0

	check.NotNil(t, result)
	check.Equal(t, 1, len(result.EligibleBids))
	check.Equal(t, "above_floor", result.Winner.Bidder)
	check.Equal(t, 1, len(result.FloorRejectedBidIDs))
	check.Equal(t, "bid2", result.FloorRejectedBidIDs[0])
}
