package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/cloudx-io/openauction/core"
)

// --- Vector utilities ---

func squaredDist(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func vecCopy(v []float64) []float64 {
	r := make([]float64, len(v))
	copy(r, v)
	return r
}

func vecNorm(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// --- Advertiser ---

type queryResult struct {
	Query   []float64
	Value   float64
	Won     bool
	Payment float64
}

type Advertiser struct {
	Name            string
	IdealCenter     []float64
	CommittedCenter []float64
	Center          []float64
	Price           float64
	Sigma           float64
	Budget          float64
	MaxValue        float64
	ValueDecay      float64

	// Cumulative tracking
	TotalValueWon float64
	TotalSpend    float64
	TotalFees     float64
	TotalWins     int

	// Per-round tracking (reset each round)
	RoundWins     int
	RoundSpend    float64
	RoundValueWon float64
	RoundQueries  []queryResult
}

const bidThresholdFrac = 0.05

func (a *Advertiser) ValueAt(query []float64) float64 {
	dist2 := squaredDist(a.IdealCenter, query)
	return a.MaxValue * math.Exp(-dist2/a.ValueDecay)
}

func (a *Advertiser) ShouldBid(query []float64) bool {
	return a.ValueAt(query) > bidThresholdFrac*a.MaxValue
}

func (a *Advertiser) Drift() float64 {
	return math.Sqrt(squaredDist(a.Center, a.CommittedCenter))
}

func (a *Advertiser) MakeBid() core.CoreBid {
	return core.CoreBid{
		ID:        "bid-" + a.Name,
		Bidder:    a.Name,
		Price:     a.Price,
		Currency:  "USD",
		Embedding: vecCopy(a.Center),
		Sigma:     a.Sigma,
	}
}

func (a *Advertiser) ResetRound() {
	a.RoundWins = 0
	a.RoundSpend = 0
	a.RoundValueWon = 0
	a.RoundQueries = a.RoundQueries[:0]
}

func (a *Advertiser) Surplus() float64 {
	return a.TotalValueWon - a.TotalSpend - a.TotalFees
}

// Adapt uses gradient-based optimization to update center, σ, and price.
func (a *Advertiser) Adapt(lambda float64) {
	const lr = 0.02

	if len(a.RoundQueries) == 0 {
		return
	}
	nq := float64(len(a.RoundQueries))

	// --- Position gradient ---
	grad := make([]float64, len(a.Center))
	for _, qr := range a.RoundQueries {
		for d := range grad {
			grad[d] += qr.Value * (qr.Query[d] - a.Center[d])
		}
	}
	gradMag := vecNorm(grad)

	// --- Sigma gradient ---
	avgPayment := a.RoundSpend / math.Max(float64(a.RoundWins), 1)
	sigmaGrad := 0.0
	for _, qr := range a.RoundQueries {
		dist2 := squaredDist(qr.Query, a.Center)
		dScore := 2 * dist2 / (a.Sigma * a.Sigma * a.Sigma)
		marginalProfit := qr.Value - avgPayment
		sigmaGrad += dScore * marginalProfit
	}
	a.Sigma += lr * sigmaGrad / nq
	a.Sigma = math.Max(a.Sigma, 0.01)

	// --- Price gradient ---
	targetPrice := 0.0
	for _, qr := range a.RoundQueries {
		targetPrice += qr.Value
	}
	targetPrice /= nq
	a.Price += lr * (targetPrice - a.Price)
	a.Price = math.Max(a.Price, 0.01)

	// --- Position update with relocation cost ---
	if gradMag < 1e-10 {
		return
	}
	stepSize := 0.03
	newCenter := vecCopy(a.Center)
	for d := range newCenter {
		newCenter[d] += (grad[d] / gradMag) * stepSize
	}

	newDist2 := squaredDist(newCenter, a.CommittedCenter)
	curDist2 := squaredDist(a.Center, a.CommittedCenter)
	cost := lambda * (newDist2 - curDist2)
	if cost < 0 {
		cost = 0
	}

	// Expected gain: average value * estimated extra wins
	totalWeight := 0.0
	for _, qr := range a.RoundQueries {
		totalWeight += qr.Value
	}
	avgValue := totalWeight / nq
	expectedGain := avgValue * math.Max(1, nq/5-float64(a.RoundWins))

	if cost <= a.Budget && (cost == 0 || expectedGain > cost) {
		a.Center = newCenter
		a.Budget -= cost
		a.TotalFees += cost
	}
}

// --- Publisher ---

type Publisher struct{ rng *rand.Rand }

func NewPublisher(seed int64) *Publisher {
	return &Publisher{rng: rand.New(rand.NewSource(seed))}
}

func (p *Publisher) SampleImpressions(n int) [][]float64 {
	pts := make([][]float64, n)
	for i := range pts {
		r := p.rng.Float64()
		cum := 0.0
		var cluster *queryCluster
		for j := range impressionClusters {
			cum += impressionClusters[j].Weight
			if r < cum {
				cluster = &impressionClusters[j]
				break
			}
		}
		if cluster == nil {
			cluster = &impressionClusters[len(impressionClusters)-1]
		}
		q := cluster.Queries[p.rng.Intn(len(cluster.Queries))]
		pts[i] = vecCopy(q)
	}
	return pts
}

// --- VCG Auction Round ---

func runAuctionRound(advs []*Advertiser, queries [][]float64) float64 {
	revenue := 0.0
	for _, q := range queries {
		var bids []core.CoreBid
		bidderMap := make(map[string]*Advertiser)
		for _, adv := range advs {
			if adv.Budget > adv.Price && adv.ShouldBid(q) {
				bids = append(bids, adv.MakeBid())
				bidderMap[adv.Name] = adv
			}
		}
		if len(bids) == 0 {
			continue
		}

		result := core.RunAuction(bids, nil, 0, q)
		if result.Winner == nil {
			continue
		}
		winner := bidderMap[result.Winner.Bidder]
		if winner == nil {
			continue
		}

		value := winner.ValueAt(q)

		// VCG settlement for scoring auctions
		var payment float64
		if result.RunnerUp != nil {
			ru := bidderMap[result.RunnerUp.Bidder]
			if ru != nil {
				distW2 := squaredDist(result.Winner.Embedding, q)
				distR2 := squaredDist(result.RunnerUp.Embedding, q)
				sigmaW := result.Winner.Sigma
				sigmaR := result.RunnerUp.Sigma
				payment = result.RunnerUp.Price * math.Exp(distW2/(sigmaW*sigmaW)-distR2/(sigmaR*sigmaR))
			} else {
				payment = result.RunnerUp.Price
			}
		} else {
			payment = winner.Price
		}

		winner.RoundWins++
		winner.RoundSpend += payment
		winner.RoundValueWon += value
		winner.TotalWins++
		winner.TotalSpend += payment
		winner.TotalValueWon += value
		winner.Budget -= payment
		revenue += payment

		// Record query result for winner
		winner.RoundQueries = append(winner.RoundQueries, queryResult{
			Query: q, Value: value, Won: true, Payment: payment,
		})

		// Record for all other bidders who bid
		for name, adv := range bidderMap {
			if name != winner.Name {
				adv.RoundQueries = append(adv.RoundQueries, queryResult{
					Query: q, Value: adv.ValueAt(q), Won: false, Payment: 0,
				})
			}
		}
	}
	return revenue
}

// --- Metrics ---

func winDiversity(advs []*Advertiser) float64 {
	totalWins := 0
	for _, a := range advs {
		totalWins += a.RoundWins
	}
	if totalWins == 0 {
		return 0
	}
	hhi := 0.0
	for _, a := range advs {
		share := float64(a.RoundWins) / float64(totalWins)
		hhi += share * share
	}
	n := float64(len(advs))
	minHHI := 1.0 / n
	if hhi <= minHHI {
		return 1.0
	}
	return (1.0 - hhi) / (1.0 - minHHI)
}

func centroid(pts [][]float64) []float64 {
	if len(pts) == 0 {
		return nil
	}
	dim := len(pts[0])
	c := make([]float64, dim)
	for _, p := range pts {
		for d := range c {
			c[d] += p[d]
		}
	}
	n := float64(len(pts))
	for d := range c {
		c[d] /= n
	}
	return c
}

func computePositionVariance(advs []*Advertiser) float64 {
	if len(advs) == 0 {
		return 0
	}
	centers := make([][]float64, len(advs))
	for i, a := range advs {
		centers[i] = a.Center
	}
	cx := centroid(centers)
	v := 0.0
	for _, a := range advs {
		v += squaredDist(a.Center, cx)
	}
	return v / float64(len(advs))
}

func avgDrift(advs []*Advertiser) float64 {
	sum := 0.0
	for _, a := range advs {
		sum += a.Drift()
	}
	return sum / float64(len(advs))
}

// --- Convergence detection ---

type advSnapshot struct {
	Center []float64
	Sigma  float64
	Price  float64
}

func snapshotAdvs(advs []*Advertiser) []advSnapshot {
	s := make([]advSnapshot, len(advs))
	for i, a := range advs {
		s[i] = advSnapshot{Center: vecCopy(a.Center), Sigma: a.Sigma, Price: a.Price}
	}
	return s
}

func hasConverged(prev []advSnapshot, curr []*Advertiser, epsilon float64) bool {
	maxChange := 0.0
	for i := range curr {
		centerDelta := math.Sqrt(squaredDist(prev[i].Center, curr[i].Center))
		sigmaDelta := math.Abs(curr[i].Sigma-prev[i].Sigma) / math.Max(prev[i].Sigma, 0.01)
		priceDelta := math.Abs(curr[i].Price-prev[i].Price) / math.Max(prev[i].Price, 0.01)
		change := centerDelta + sigmaDelta + priceDelta
		if change > maxChange {
			maxChange = change
		}
	}
	return maxChange < epsilon
}

// --- Trial / Experiment ---

type TrialResult struct {
	AdvSurplus   float64
	PubRevenue   float64
	TotalSurplus float64
	FeeRevenue   float64
	WinDiversity float64
	PosVariance  float64
	Rounds       int
	AvgDrift     float64
}

const (
	maxRounds          = 500
	convergenceEpsilon = 0.001
	convergenceWindow  = 5
	impressionsPerRound = 50
)

func runTrial(lambda float64, valueDecay float64, makeAdvs func(rng *rand.Rand, valueDecay float64) []*Advertiser, seed int64) TrialResult {
	rng := rand.New(rand.NewSource(seed))
	pub := NewPublisher(seed)
	advs := makeAdvs(rng, valueDecay)

	stableCount := 0
	roundsRun := 0

	for round := 1; round <= maxRounds; round++ {
		prev := snapshotAdvs(advs)

		for _, adv := range advs {
			adv.ResetRound()
		}

		queries := pub.SampleImpressions(impressionsPerRound)
		runAuctionRound(advs, queries)

		for _, adv := range advs {
			adv.Adapt(lambda)
		}

		roundsRun = round

		if hasConverged(prev, advs, convergenceEpsilon) {
			stableCount++
			if stableCount >= convergenceWindow {
				break
			}
		} else {
			stableCount = 0
		}
	}

	// Compute metrics
	totalAdvSurplus := 0.0
	totalFees := 0.0
	for _, a := range advs {
		totalAdvSurplus += a.Surplus()
		totalFees += a.TotalFees
	}
	avgSurplus := totalAdvSurplus / float64(len(advs))

	pubRevenue := 0.0
	for _, a := range advs {
		pubRevenue += a.TotalSpend
	}

	return TrialResult{
		AdvSurplus:   avgSurplus,
		PubRevenue:   pubRevenue,
		TotalSurplus: totalAdvSurplus + pubRevenue,
		FeeRevenue:   totalFees,
		WinDiversity: winDiversity(advs),
		PosVariance:  computePositionVariance(advs),
		Rounds:       roundsRun,
		AvgDrift:     avgDrift(advs),
	}
}

// --- Stat helpers ---

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func meanStd(vals []float64) (float64, float64) {
	if len(vals) == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	m := sum / float64(len(vals))
	v := 0.0
	for _, x := range vals {
		d := x - m
		v += d * d
	}
	return m, math.Sqrt(v / float64(len(vals)))
}

func pct(vals []float64, p float64) float64 {
	s := make([]float64, len(vals))
	copy(s, vals)
	sort.Float64s(s)
	return s[int(p/100.0*float64(len(s)-1))]
}

func fmtStats(vals []float64) string {
	mean, std := meanStd(vals)
	return fmt.Sprintf("mean=%.4f  std=%.4f  [p5=%.4f, p95=%.4f]",
		mean, std, pct(vals, 5), pct(vals, 95))
}

func jitter(rng *rand.Rand, base, spread float64) float64 {
	v := base + rng.NormFloat64()*spread
	if v < 0.01 {
		return 0.01
	}
	return v
}

// --- Factories ---

func makeAdvertisers(rng *rand.Rand, valueDecay float64) []*Advertiser {
	advs := make([]*Advertiser, len(advertiserData))
	noiseScale := 0.02 / math.Sqrt(float64(embeddingDim))
	for i, d := range advertiserData {
		ideal := vecCopy(d.Embedding)
		center := vecCopy(d.Embedding)
		for j := range center {
			center[j] += rng.NormFloat64() * noiseScale
		}
		advs[i] = &Advertiser{
			Name:            d.Name,
			IdealCenter:     ideal,
			CommittedCenter: vecCopy(center),
			Center:          center,
			Price:           jitter(rng, d.BaseBid, d.BaseBid*0.10),
			Sigma:           clamp(jitter(rng, d.BaseSigma, 0.03), 0.10, 0.75),
			Budget:          jitter(rng, d.BaseBudget, d.BaseBudget*0.10),
			MaxValue:        jitter(rng, d.MaxValue, 1.0),
			ValueDecay:      valueDecay,
		}
	}
	return advs
}

func makeKeywordEmbedding(rng *rand.Rand, valueDecay float64) []*Advertiser {
	advs := makeAdvertisers(rng, valueDecay)
	for i := 0; i < 5 && i < len(advs); i++ {
		advs[i].Sigma = clamp(jitter(rng, 0.12, 0.02), 0.08, 0.18)
		advs[i].Price *= 1.5
	}
	return advs
}

func cloneAdvertiser(a *Advertiser) *Advertiser {
	c := *a
	c.IdealCenter = vecCopy(a.IdealCenter)
	c.CommittedCenter = vecCopy(a.CommittedCenter)
	c.Center = vecCopy(a.Center)
	c.RoundQueries = nil
	return &c
}

// --- Experiment 0: Value Decay Calibration ---

func runCalibration() float64 {
	fmt.Println("\nExperiment 0: Value Decay Calibration")
	fmt.Println("=====================================")

	// For each advertiser, find closest query (proxy for "own niche")
	// and use percentile-based classification
	var allDists []float64
	var closestDists []float64 // each advertiser's closest query dist²

	for _, ad := range advertiserData {
		minDist := math.Inf(1)
		for _, c := range impressionClusters {
			for _, q := range c.Queries {
				dist2 := squaredDist(ad.Embedding, q)
				allDists = append(allDists, dist2)
				if dist2 < minDist {
					minDist = dist2
				}
			}
		}
		closestDists = append(closestDists, minDist)
	}

	sort.Float64s(allDists)
	sort.Float64s(closestDists)

	// Use p25 of all distances as "nearby" threshold
	nearbyThreshold := pct(allDists, 25)
	var nearDists, farDists []float64
	for _, d := range allDists {
		if d <= nearbyThreshold {
			nearDists = append(nearDists, d)
		} else {
			farDists = append(farDists, d)
		}
	}

	fmt.Printf("\n  All %d advertiser-query pairs:\n", len(allDists))
	fmt.Printf("    dist² range: [%.4f, %.4f]\n", allDists[0], allDists[len(allDists)-1])
	fmt.Printf("    p10=%.4f  p25=%.4f  p50=%.4f  p75=%.4f  p90=%.4f\n",
		pct(allDists, 10), pct(allDists, 25), pct(allDists, 50),
		pct(allDists, 75), pct(allDists, 90))

	fmt.Printf("\n  Closest-query dist² per advertiser:\n")
	closestMean, closestStd := meanStd(closestDists)
	fmt.Printf("    mean=%.4f  std=%.4f  range=[%.4f, %.4f]\n",
		closestMean, closestStd, closestDists[0], closestDists[len(closestDists)-1])

	fmt.Printf("\n  Near pairs (%d, dist²≤%.4f):\n", len(nearDists), nearbyThreshold)
	nearMean, nearStd := meanStd(nearDists)
	fmt.Printf("    mean=%.4f  std=%.4f\n", nearMean, nearStd)

	fmt.Printf("  Far pairs (%d, dist²>%.4f):\n", len(farDists), nearbyThreshold)
	farMean, farStd := meanStd(farDists)
	fmt.Printf("    mean=%.4f  std=%.4f\n", farMean, farStd)

	// Calibrate so that nearby queries get ~65% of max value
	// and far queries get ~15% of max value
	// exp(-nearMedian / decay) = 0.65  →  decay = -nearMedian / ln(0.65)
	nearMedian := pct(nearDists, 50)
	calibrated := -nearMedian / math.Log(0.65)
	calibrated = math.Max(calibrated, 0.1)

	fmt.Printf("\n  Calibrated valueDecay = %.4f\n", calibrated)
	fmt.Printf("  Verification: near median (%.4f) → value = %.1f%% of max\n",
		nearMedian, 100*math.Exp(-nearMedian/calibrated))
	farMedian := pct(farDists, 50)
	fmt.Printf("  Verification: far median (%.4f) → value = %.1f%% of max\n",
		farMedian, 100*math.Exp(-farMedian/calibrated))

	return calibrated
}

// --- Experiment 1: Lambda Sweep ---

type sweepResult struct {
	Lambda  float64
	Results []TrialResult
}

func runLambdaSweep(valueDecay float64) []sweepResult {
	name := "Experiment 1: Lambda Sweep"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	lambdas := []float64{0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000}
	const trials = 50

	results := make([]sweepResult, len(lambdas))

	for li, lambda := range lambdas {
		trialResults := make([]TrialResult, trials)
		for i := range trialResults {
			trialResults[i] = runTrial(lambda, valueDecay, makeAdvertisers, int64(i*7919+42))
		}
		results[li] = sweepResult{Lambda: lambda, Results: trialResults}
	}

	// Print table
	fmt.Printf("\n  %-8s  %-12s  %-12s  %-12s  %-12s  %-12s  %-8s\n",
		"Lambda", "AdvSurplus", "PubRevenue", "TotalSurp", "WinDiv", "PosVar", "Rounds")
	fmt.Println("  " + strings.Repeat("-", 84))

	for _, sr := range results {
		surps := make([]float64, len(sr.Results))
		revs := make([]float64, len(sr.Results))
		totals := make([]float64, len(sr.Results))
		divs := make([]float64, len(sr.Results))
		vars := make([]float64, len(sr.Results))
		rounds := make([]float64, len(sr.Results))
		for i, r := range sr.Results {
			surps[i] = r.AdvSurplus
			revs[i] = r.PubRevenue
			totals[i] = r.TotalSurplus
			divs[i] = r.WinDiversity
			vars[i] = r.PosVariance
			rounds[i] = float64(r.Rounds)
		}
		ms, _ := meanStd(surps)
		mr, _ := meanStd(revs)
		mt, _ := meanStd(totals)
		md, _ := meanStd(divs)
		mv, _ := meanStd(vars)
		mrd, _ := meanStd(rounds)
		fmt.Printf("  %-8.0f  %-12.2f  %-12.2f  %-12.2f  %-12.4f  %-12.6f  %-8.0f\n",
			sr.Lambda, ms, mr, mt, md, mv, mrd)
	}

	return results
}

func findOptimalLambda(results []sweepResult) float64 {
	// Select optimal λ: highest total surplus among configurations
	// with meaningful win diversity (>0.05). If none qualify, pick
	// the one with highest diversity as a fallback.
	const minDiv = 0.05

	bestLambda := 0.0
	bestSurplus := math.Inf(-1)
	fallbackLambda := 0.0
	fallbackDiv := -1.0

	for _, sr := range results {
		totals := make([]float64, len(sr.Results))
		divs := make([]float64, len(sr.Results))
		for i, r := range sr.Results {
			totals[i] = r.TotalSurplus
			divs[i] = r.WinDiversity
		}
		meanSurplus, _ := meanStd(totals)
		meanDiv, _ := meanStd(divs)

		if meanDiv > fallbackDiv {
			fallbackDiv = meanDiv
			fallbackLambda = sr.Lambda
		}

		if meanDiv >= minDiv && meanSurplus > bestSurplus {
			bestSurplus = meanSurplus
			bestLambda = sr.Lambda
		}
	}

	if bestSurplus == math.Inf(-1) {
		bestLambda = fallbackLambda
		fmt.Printf("\n  No λ achieved diversity ≥ %.2f; using highest-diversity λ = %.0f\n", minDiv, bestLambda)
	} else {
		fmt.Printf("\n  Optimal λ = %.0f (total surplus = %.2f, diversity ≥ %.2f)\n", bestLambda, bestSurplus, minDiv)
	}
	return bestLambda
}

// --- Experiment 2: Switching Cost ---

func runSwitchingCost(valueDecay float64, optimalLambda float64) {
	name := "Experiment 2: Switching Cost Recovery"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	const trials = 50

	type switchResult struct {
		PreCollapsePosVar  float64
		PostRecoveryPosVar float64
		PreCollapseDiv     float64
		PostRecoveryDiv    float64
		CollapseRounds     int
		RecoveryRounds     int
	}

	sResults := make([]switchResult, trials)

	for t := 0; t < trials; t++ {
		seed := int64(t*7919 + 42)
		rng := rand.New(rand.NewSource(seed))
		pub := NewPublisher(seed)
		advs := makeAdvertisers(rng, valueDecay)

		// Phase 1: λ=0 until convergence (collapse)
		stableCount := 0
		collapseRounds := 0
		for round := 1; round <= maxRounds; round++ {
			prev := snapshotAdvs(advs)
			for _, adv := range advs {
				adv.ResetRound()
			}
			queries := pub.SampleImpressions(impressionsPerRound)
			runAuctionRound(advs, queries)
			for _, adv := range advs {
				adv.Adapt(0)
			}
			collapseRounds = round
			if hasConverged(prev, advs, convergenceEpsilon) {
				stableCount++
				if stableCount >= convergenceWindow {
					break
				}
			} else {
				stableCount = 0
			}
		}

		preVar := computePositionVariance(advs)
		preDiv := winDiversity(advs)

		// Recommit: anchor positions at current (collapsed) state
		// so λ constrains further drift, not drift from original
		for _, adv := range advs {
			adv.CommittedCenter = vecCopy(adv.Center)
		}

		// Phase 2: switch to optimal λ, continue until convergence
		stableCount = 0
		recoveryRounds := 0
		for round := 1; round <= maxRounds; round++ {
			prev := snapshotAdvs(advs)
			for _, adv := range advs {
				adv.ResetRound()
			}
			queries := pub.SampleImpressions(impressionsPerRound)
			runAuctionRound(advs, queries)
			for _, adv := range advs {
				adv.Adapt(optimalLambda)
			}
			recoveryRounds = round
			if hasConverged(prev, advs, convergenceEpsilon) {
				stableCount++
				if stableCount >= convergenceWindow {
					break
				}
			} else {
				stableCount = 0
			}
		}

		postVar := computePositionVariance(advs)
		postDiv := winDiversity(advs)

		sResults[t] = switchResult{
			PreCollapsePosVar:  preVar,
			PostRecoveryPosVar: postVar,
			PreCollapseDiv:     preDiv,
			PostRecoveryDiv:    postDiv,
			CollapseRounds:     collapseRounds,
			RecoveryRounds:     recoveryRounds,
		}
	}

	preVars := make([]float64, trials)
	postVars := make([]float64, trials)
	preDivs := make([]float64, trials)
	postDivs := make([]float64, trials)
	colRounds := make([]float64, trials)
	recRounds := make([]float64, trials)
	for i, r := range sResults {
		preVars[i] = r.PreCollapsePosVar
		postVars[i] = r.PostRecoveryPosVar
		preDivs[i] = r.PreCollapseDiv
		postDivs[i] = r.PostRecoveryDiv
		colRounds[i] = float64(r.CollapseRounds)
		recRounds[i] = float64(r.RecoveryRounds)
	}

	fmt.Printf("\n  Trials: %d  |  Phase 1: λ=0 → collapse  |  Phase 2: λ=%.0f → recovery\n", trials, optimalLambda)
	fmt.Printf("\n  Phase 1 (collapse with λ=0):\n")
	fmt.Printf("    Rounds to converge:   %s\n", fmtStats(colRounds))
	fmt.Printf("    Position variance:    %s\n", fmtStats(preVars))
	fmt.Printf("    Win diversity:        %s\n", fmtStats(preDivs))
	fmt.Printf("\n  Phase 2 (recovery with λ=%.0f):\n", optimalLambda)
	fmt.Printf("    Rounds to converge:   %s\n", fmtStats(recRounds))
	fmt.Printf("    Position variance:    %s\n", fmtStats(postVars))
	fmt.Printf("    Win diversity:        %s\n", fmtStats(postDivs))
}

// --- Experiment 3: Keyword/Embedding Coexistence ---

func runKeywordCoexistence(valueDecay float64, optimalLambda float64) {
	name := "Experiment 3: Keyword/Embedding Coexistence"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	const trials = 50

	results := make([]TrialResult, trials)
	for i := range results {
		results[i] = runTrial(optimalLambda, valueDecay, makeKeywordEmbedding, int64(i*7919+42))
	}

	surps := make([]float64, trials)
	revs := make([]float64, trials)
	divs := make([]float64, trials)
	vars := make([]float64, trials)
	rounds := make([]float64, trials)
	drifts := make([]float64, trials)
	for i, r := range results {
		surps[i] = r.AdvSurplus
		revs[i] = r.PubRevenue
		divs[i] = r.WinDiversity
		vars[i] = r.PosVariance
		rounds[i] = float64(r.Rounds)
		drifts[i] = r.AvgDrift
	}

	fmt.Printf("\n  Trials: %d  |  5 keyword (σ=0.10-0.15) + 10 embedding  |  λ=%.0f\n", trials, optimalLambda)
	fmt.Printf("  Avg surplus:      %s\n", fmtStats(surps))
	fmt.Printf("  Pub revenue:      %s\n", fmtStats(revs))
	fmt.Printf("  Win diversity:    %s\n", fmtStats(divs))
	fmt.Printf("  Position var:     %s\n", fmtStats(vars))
	fmt.Printf("  Avg drift:        %s\n", fmtStats(drifts))
	fmt.Printf("  Rounds:           %s\n", fmtStats(rounds))
}

// --- Experiment 4: Dual Market ---

type dualTrialResult struct {
	SurplusA, SurplusB   float64
	RevenueA, RevenueB   float64
	DivA, DivB           float64
	VarA, VarB           float64
	DriftA, DriftB       float64
	RoundsA, RoundsB     int
}

func runDualMarket(valueDecay float64, optimalLambda float64) {
	name := "Experiment 4: Dual Market Comparison"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	const trials = 50

	results := make([]dualTrialResult, trials)

	for t := 0; t < trials; t++ {
		seed := int64(t*7919 + 42)
		rng := rand.New(rand.NewSource(seed))
		pub := NewPublisher(seed)

		baseAdvs := makeAdvertisers(rng, valueDecay)
		advsA := make([]*Advertiser, len(baseAdvs))
		advsB := make([]*Advertiser, len(baseAdvs))
		for i, a := range baseAdvs {
			advsA[i] = cloneAdvertiser(a)
			advsB[i] = cloneAdvertiser(a)
		}

		stableCountA, stableCountB := 0, 0
		roundsA, roundsB := maxRounds, maxRounds
		doneA, doneB := false, false

		for round := 1; round <= maxRounds; round++ {
			queries := pub.SampleImpressions(impressionsPerRound)

			// Run both exchanges
			if !doneA {
				prevA := snapshotAdvs(advsA)
				for _, adv := range advsA {
					adv.ResetRound()
				}
				runAuctionRound(advsA, queries)
				for _, adv := range advsA {
					adv.Adapt(optimalLambda)
				}
				if hasConverged(prevA, advsA, convergenceEpsilon) {
					stableCountA++
					if stableCountA >= convergenceWindow {
						roundsA = round
						doneA = true
					}
				} else {
					stableCountA = 0
				}
			}

			if !doneB {
				prevB := snapshotAdvs(advsB)
				for _, adv := range advsB {
					adv.ResetRound()
				}
				runAuctionRound(advsB, queries)
				for _, adv := range advsB {
					adv.Adapt(0)
				}
				if hasConverged(prevB, advsB, convergenceEpsilon) {
					stableCountB++
					if stableCountB >= convergenceWindow {
						roundsB = round
						doneB = true
					}
				} else {
					stableCountB = 0
				}
			}

			// Shared budget: deduct from base
			for i := range baseAdvs {
				spentA := advsA[i].TotalSpend + advsA[i].TotalFees
				spentB := advsB[i].TotalSpend + advsB[i].TotalFees
				remaining := baseAdvs[i].Budget - spentA - spentB
				if remaining < 0 {
					remaining = 0
				}
				advsA[i].Budget = remaining
				advsB[i].Budget = remaining
			}

			if doneA && doneB {
				break
			}
		}

		// Compute metrics for each exchange
		surpA, surpB := 0.0, 0.0
		for _, a := range advsA {
			surpA += a.Surplus()
		}
		for _, a := range advsB {
			surpB += a.Surplus()
		}

		revA, revB := 0.0, 0.0
		for _, a := range advsA {
			revA += a.TotalSpend
		}
		for _, a := range advsB {
			revB += a.TotalSpend
		}

		results[t] = dualTrialResult{
			SurplusA: surpA / float64(len(advsA)),
			SurplusB: surpB / float64(len(advsB)),
			RevenueA: revA,
			RevenueB: revB,
			DivA:     winDiversity(advsA),
			DivB:     winDiversity(advsB),
			VarA:     computePositionVariance(advsA),
			VarB:     computePositionVariance(advsB),
			DriftA:   avgDrift(advsA),
			DriftB:   avgDrift(advsB),
			RoundsA:  roundsA,
			RoundsB:  roundsB,
		}
	}

	// Print comparison
	surpAs := make([]float64, trials)
	surpBs := make([]float64, trials)
	revAs := make([]float64, trials)
	revBs := make([]float64, trials)
	divAs := make([]float64, trials)
	divBs := make([]float64, trials)
	varAs := make([]float64, trials)
	varBs := make([]float64, trials)
	driftAs := make([]float64, trials)
	driftBs := make([]float64, trials)
	for i, r := range results {
		surpAs[i] = r.SurplusA
		surpBs[i] = r.SurplusB
		revAs[i] = r.RevenueA
		revBs[i] = r.RevenueB
		divAs[i] = r.DivA
		divBs[i] = r.DivB
		varAs[i] = r.VarA
		varBs[i] = r.VarB
		driftAs[i] = r.DriftA
		driftBs[i] = r.DriftB
	}

	fmt.Printf("\n  Trials: %d  |  Shared budget, same impressions\n", trials)
	fmt.Printf("\n  Exchange A (λ=%.0f):\n", optimalLambda)
	fmt.Printf("    Avg surplus:      %s\n", fmtStats(surpAs))
	fmt.Printf("    Revenue:          %s\n", fmtStats(revAs))
	fmt.Printf("    Win diversity:    %s\n", fmtStats(divAs))
	fmt.Printf("    Position var:     %s\n", fmtStats(varAs))
	fmt.Printf("    Avg drift:        %s\n", fmtStats(driftAs))
	fmt.Printf("\n  Exchange B (λ=0):\n")
	fmt.Printf("    Avg surplus:      %s\n", fmtStats(surpBs))
	fmt.Printf("    Revenue:          %s\n", fmtStats(revBs))
	fmt.Printf("    Win diversity:    %s\n", fmtStats(divBs))
	fmt.Printf("    Position var:     %s\n", fmtStats(varBs))
	fmt.Printf("    Avg drift:        %s\n", fmtStats(driftBs))
}

// --- Main ---

func main() {
	// Exp 0: calibrate value decay
	valueDecay := runCalibration()

	// Exp 1: λ sweep
	sweepResults := runLambdaSweep(valueDecay)
	optimalLambda := findOptimalLambda(sweepResults)

	// Exp 2: switching cost recovery
	runSwitchingCost(valueDecay, optimalLambda)

	// Exp 3: keyword/embedding coexistence
	runKeywordCoexistence(valueDecay, optimalLambda)

	// Exp 4: dual market
	runDualMarket(valueDecay, optimalLambda)
}
