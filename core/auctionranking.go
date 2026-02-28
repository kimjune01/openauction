package core

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"sort"
)

// RandSource provides random number generation for tie-breaking.
// This interface enables dependency injection for deterministic testing.
type RandSource interface {
	// Intn returns a random integer in [0, n). Panics if n <= 0.
	Intn(n int) int
}

// cryptoRandSource wraps crypto/rand for production use
type cryptoRandSource struct{}

// Intn returns a cryptographically secure random integer in [0, n).
// Panics if n <= 0 (programmer error).
func (cryptoRandSource) Intn(n int) int {
	if n <= 0 {
		panic(fmt.Sprintf("cryptoRandSource.Intn: n must be positive, got %d", n))
	}
	// rand.Int does not error when using rand.Reader
	// https://pkg.go.dev/crypto/rand#Int
	nBig, _ := rand.Int(rand.Reader, big.NewInt(int64(n)))
	return int(nBig.Int64())
}

// defaultRandSource provides a cryptographically secure random source for production
var defaultRandSource RandSource = cryptoRandSource{}

// ScoredBid pairs a CoreBid with a pre-computed embedding score.
type ScoredBid struct {
	CoreBid
	Score float64
}

// RankScoredBids ranks bids by Score (descending) instead of Price.
// Per-bidder highest is chosen by Score. Tie-breaking uses random shuffle.
func RankScoredBids(bids []ScoredBid, randSource RandSource) *CoreRankingResult {
	if len(bids) == 0 {
		return &CoreRankingResult{
			Ranks:         make(map[string]int),
			HighestBids:   make(map[string]*CoreBid),
			SortedBidders: make([]string, 0),
		}
	}

	type ScoredEntry struct {
		bidder string
		bid    *CoreBid
		score  float64
	}

	// Find highest-score bid per bidder
	bidderMap := make(map[string]*ScoredEntry)
	bidderOrder := make([]string, 0, len(bids))
	seenBidders := make(map[string]bool)

	for i := range bids {
		sb := &bids[i]
		if !seenBidders[sb.Bidder] {
			bidderOrder = append(bidderOrder, sb.Bidder)
			seenBidders[sb.Bidder] = true
		}
		existing, exists := bidderMap[sb.Bidder]
		if !exists || sb.Score > existing.score {
			bid := sb.CoreBid // copy
			bidderMap[sb.Bidder] = &ScoredEntry{bidder: sb.Bidder, bid: &bid, score: sb.Score}
		}
	}

	entries := make([]ScoredEntry, 0, len(bidderOrder))
	for _, bidder := range bidderOrder {
		entries = append(entries, *bidderMap[bidder])
	}

	// Sort by score descending
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].score > entries[j].score
	})

	if randSource == nil {
		randSource = defaultRandSource
	}

	// Break ties randomly (same Fisher-Yates as RankCoreBids)
	i := 0
	for i < len(entries) {
		score := entries[i].score
		j := i + 1
		for j < len(entries) && entries[j].score == score {
			j++
		}
		if j-i > 1 {
			for k := j - 1; k > i; k-- {
				randIdx := i + randSource.Intn(k-i+1)
				entries[k], entries[randIdx] = entries[randIdx], entries[k]
			}
		}
		i = j
	}

	result := &CoreRankingResult{
		Ranks:         make(map[string]int, len(entries)),
		HighestBids:   make(map[string]*CoreBid, len(entries)),
		SortedBidders: make([]string, len(entries)),
	}

	for rank, entry := range entries {
		rankValue := rank + 1
		result.Ranks[entry.bidder] = rankValue
		result.HighestBids[entry.bidder] = entry.bid
		result.SortedBidders[rank] = entry.bidder
	}

	return result
}

func RankCoreBids(bids []CoreBid, randSource RandSource) *CoreRankingResult {
	if len(bids) == 0 {
		return &CoreRankingResult{
			Ranks:         make(map[string]int),
			HighestBids:   make(map[string]*CoreBid),
			SortedBidders: make([]string, 0),
		}
	}

	type BidEntry struct {
		bidder string
		bid    *CoreBid
	}

	// Find highest bid per bidder while preserving order of first occurrence
	bidderMap := make(map[string]*CoreBid)
	bidderOrder := make([]string, 0, len(bids))
	seenBidders := make(map[string]bool)

	for i := range bids {
		bid := &bids[i]

		// Track first occurrence order
		if !seenBidders[bid.Bidder] {
			bidderOrder = append(bidderOrder, bid.Bidder)
			seenBidders[bid.Bidder] = true
		}

		// Keep highest bid per bidder
		existing, exists := bidderMap[bid.Bidder]
		if !exists || bid.Price > existing.Price {
			bidderMap[bid.Bidder] = bid
		}
	}

	// Build entries in order of first occurrence
	entries := make([]BidEntry, 0, len(bidderOrder))
	for _, bidder := range bidderOrder {
		entries = append(entries, BidEntry{
			bidder: bidder,
			bid:    bidderMap[bidder],
		})
	}

	// Sort by price descending
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].bid.Price > entries[j].bid.Price
	})

	// Use default crypto/rand source if none provided
	if randSource == nil {
		randSource = defaultRandSource
	}

	// Break ties randomly: shuffle groups of bids with the same price using Fisher-Yates
	i := 0
	for i < len(entries) {
		// Find the range of entries with the same price
		price := entries[i].bid.Price
		j := i + 1
		for j < len(entries) && entries[j].bid.Price == price {
			j++
		}

		// If there are ties (j-i > 1), shuffle this group
		if j-i > 1 {
			for k := j - 1; k > i; k-- {
				// Pick a random index from i to k (inclusive)
				randIdx := i + randSource.Intn(k-i+1)
				entries[k], entries[randIdx] = entries[randIdx], entries[k]
			}
		}

		i = j
	}

	result := &CoreRankingResult{
		Ranks:         make(map[string]int, len(entries)),
		HighestBids:   make(map[string]*CoreBid, len(entries)),
		SortedBidders: make([]string, len(entries)),
	}

	for rank, entry := range entries {
		rankValue := rank + 1
		result.Ranks[entry.bidder] = rankValue
		result.HighestBids[entry.bidder] = entry.bid
		result.SortedBidders[rank] = entry.bidder
	}

	return result
}
