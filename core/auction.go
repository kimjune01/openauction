package core

// validateBidPrices filters bids with invalid (non-positive) prices.
func validateBidPrices(bids []CoreBid) (valid []CoreBid, rejectedBidIDs []string) {
	validBids := make([]CoreBid, 0, len(bids))
	rejectedIDs := make([]string, 0)

	for _, bid := range bids {
		if bid.Price > 0.0 {
			validBids = append(validBids, bid)
		} else {
			rejectedIDs = append(rejectedIDs, bid.ID)
		}
	}

	return validBids, rejectedIDs
}

// RunAuction executes the core auction logic: price validation → adjustment → floor enforcement → ranking.
// This function provides a unified auction implementation used by both TEE and local processing.
//
// Parameters:
//   - bids: Input bids (should already be decrypted if from TEE)
//   - adjustmentFactors: Per-bidder adjustment multipliers
//   - bidFloor: Floor price (applies to price, not score)
//   - queryEmbedding: (optional) If provided, enables embedding-space scoring.
//     Bids with embeddings are scored as log(price) - distance²/σ² and ranked by score.
//     Bids without embeddings fall back to log(price).
//
// Returns:
//   - AuctionResult containing winner, runner-up, eligible bids, rejected bids, and full ranking
//
// Processing flow:
//  1. Validate bid prices (reject non-positive prices)
//  2. Apply bid adjustment factors (multipliers per bidder)
//  3. Enforce floor price (on price, not score)
//  4. If queryEmbedding provided: compute scores, rank by score; otherwise: rank by price
//  5. Extract winner and runner-up from ranking
func RunAuction(
	bids []CoreBid,
	adjustmentFactors map[string]float64,
	bidFloor float64,
	queryEmbedding ...[]float64,
) *AuctionResult {
	// Step 1: Validate bid prices
	validBids, priceRejectedBids := validateBidPrices(bids)

	// Step 2: Apply bid adjustment factors
	adjustedBids := validBids
	if len(adjustmentFactors) > 0 {
		adjustedBids = ApplyBidAdjustmentFactors(validBids, adjustmentFactors)
	}

	// Step 3: Enforce floor price (on price, not score)
	eligibleBids, floorRejectedBids := EnforceBidFloor(adjustedBids, bidFloor)

	// Step 4: Rank eligible bids
	var ranking *CoreRankingResult
	if len(queryEmbedding) > 0 && len(queryEmbedding[0]) > 0 {
		// Embedding-space scoring path
		qe := queryEmbedding[0]
		scoredBids := make([]ScoredBid, len(eligibleBids))
		for i, bid := range eligibleBids {
			scoredBids[i] = ScoredBid{
				CoreBid: bid,
				Score:   ComputeEmbeddingScore(bid.Price, bid.Embedding, bid.Sigma, qe),
			}
		}
		ranking = RankScoredBids(scoredBids, defaultRandSource)
	} else {
		// Pure price ranking path (unchanged)
		ranking = RankCoreBids(eligibleBids, defaultRandSource)
	}

	// Step 5: Extract winner and runner-up from ranking
	var winner, runnerUp *CoreBid
	if len(ranking.SortedBidders) > 0 {
		winner = ranking.HighestBids[ranking.SortedBidders[0]]
	}
	if len(ranking.SortedBidders) > 1 {
		runnerUp = ranking.HighestBids[ranking.SortedBidders[1]]
	}

	return &AuctionResult{
		Winner:              winner,
		RunnerUp:            runnerUp,
		EligibleBids:        eligibleBids,
		PriceRejectedBidIDs: priceRejectedBids,
		FloorRejectedBidIDs: floorRejectedBids,
	}
}
