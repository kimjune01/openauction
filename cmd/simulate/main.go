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

func cosineSim(a, b []float64) float64 {
	dot := 0.0
	for i := range a {
		dot += a[i] * b[i]
	}
	na := vecNorm(a)
	nb := vecNorm(b)
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (na * nb)
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
	Cluster         int
	FreezePosition  bool // keyword advertisers: frozen center + sigma
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

	// Keyword advertisers (FreezePosition=true): only adapt price, not position/sigma
	if !a.FreezePosition {
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

		// --- Position update with relocation cost ---
		if gradMag >= 1e-10 {
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
	}

	// --- Price gradient (always applies) ---
	targetPrice := 0.0
	for _, qr := range a.RoundQueries {
		targetPrice += qr.Value
	}
	targetPrice /= nq
	a.Price += lr * (targetPrice - a.Price)
	a.Price = math.Max(a.Price, 0.01)
}

// --- Publisher ---

type Publisher struct{ rng *rand.Rand }

func NewPublisher(seed int64) *Publisher {
	return &Publisher{rng: rand.New(rand.NewSource(seed))}
}

// SampledQuery carries embedding + metadata through the auction.
type SampledQuery struct {
	Embedding  []float64
	ClusterIdx int
	QueryIdx   int
}

func (p *Publisher) SampleImpressions(n int) []SampledQuery {
	pts := make([]SampledQuery, n)
	for i := range pts {
		r := p.rng.Float64()
		cum := 0.0
		ci := len(impressionClusters) - 1
		for j := range impressionClusters {
			cum += impressionClusters[j].Weight
			if r < cum {
				ci = j
				break
			}
		}
		qi := p.rng.Intn(len(impressionClusters[ci].Queries))
		pts[i] = SampledQuery{
			Embedding:  vecCopy(impressionClusters[ci].Queries[qi]),
			ClusterIdx: ci,
			QueryIdx:   qi,
		}
	}
	return pts
}

// --- Per-query auction result ---

type perQueryResult struct {
	WinnerValue float64
	MaxValue    float64
	WinnerName  string
	QueryType   int // QuerySpecialist/QueryBoundary/QueryGeneral
}

// --- Runtime query type classification ---

// computeQueryTypes classifies each query as specialist/boundary/general
// based on the actual advertiser value landscape.
func computeQueryTypes(valueDecay float64) [][]int {
	types := make([][]int, len(impressionClusters))
	for ci, c := range impressionClusters {
		types[ci] = make([]int, len(c.Queries))
		for qi, q := range c.Queries {
			// Compute value for each advertiser
			vals := make([]float64, len(advertiserData))
			for ai, ad := range advertiserData {
				dist2 := squaredDist(ad.Embedding, q)
				vals[ai] = ad.MaxValue * math.Exp(-dist2/valueDecay)
			}
			// Sort descending
			sorted := make([]float64, len(vals))
			copy(sorted, vals)
			sort.Sort(sort.Reverse(sort.Float64Slice(sorted)))

			top := sorted[0]
			second := sorted[1]

			if top < 0.5 {
				types[ci][qi] = QueryGeneral
			} else if second > 0 && top/second > 1.5 {
				types[ci][qi] = QuerySpecialist
			} else {
				types[ci][qi] = QueryBoundary
			}
		}
	}
	return types
}

// Global query types, set once in main after calibration.
var globalQueryTypes [][]int

// --- VCG Auction Round ---

func runAuctionRound(advs []*Advertiser, queries []SampledQuery) (float64, []perQueryResult) {
	revenue := 0.0
	var pqResults []perQueryResult

	for _, sq := range queries {
		q := sq.Embedding
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

		// Find max value across all bidders for this query
		maxVal := 0.0
		for _, adv := range bidderMap {
			v := adv.ValueAt(q)
			if v > maxVal {
				maxVal = v
			}
		}

		// VCG settlement — same formula for all cells (no σ=0 special case)
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

		// Cap payment to prevent numerical explosion
		if payment > winner.Price*10 {
			payment = winner.Price * 10
		}

		winner.RoundWins++
		winner.RoundSpend += payment
		winner.RoundValueWon += value
		winner.TotalWins++
		winner.TotalSpend += payment
		winner.TotalValueWon += value
		winner.Budget -= payment
		revenue += payment

		// Determine query type from global cache
		qType := QueryGeneral
		if globalQueryTypes != nil && sq.ClusterIdx < len(globalQueryTypes) && sq.QueryIdx < len(globalQueryTypes[sq.ClusterIdx]) {
			qType = globalQueryTypes[sq.ClusterIdx][sq.QueryIdx]
		}

		pqResults = append(pqResults, perQueryResult{
			WinnerValue: value,
			MaxValue:    maxVal,
			WinnerName:  winner.Name,
			QueryType:   qType,
		})

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
	return revenue, pqResults
}

// --- Metrics ---

func winDiversityFromWins(winCounts map[string]int, nAdvs int) float64 {
	total := 0
	for _, c := range winCounts {
		total += c
	}
	if total == 0 {
		return 0
	}
	hhi := 0.0
	for _, c := range winCounts {
		share := float64(c) / float64(total)
		hhi += share * share
	}
	n := float64(nAdvs)
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

func valueEfficiencyFromResults(pqResults []perQueryResult) float64 {
	if len(pqResults) == 0 {
		return 0
	}
	sum := 0.0
	for _, pq := range pqResults {
		if pq.MaxValue > 0 {
			sum += pq.WinnerValue / pq.MaxValue
		}
	}
	return sum / float64(len(pqResults))
}

// boundaryDiversityFromResults computes HHI on boundary query winners.
func boundaryDiversityFromResults(pqResults []perQueryResult, nAdvs int) float64 {
	winCounts := make(map[string]int)
	total := 0
	for _, pq := range pqResults {
		if pq.QueryType == QueryBoundary {
			winCounts[pq.WinnerName]++
			total++
		}
	}
	if total == 0 {
		return 0
	}
	hhi := 0.0
	for _, c := range winCounts {
		share := float64(c) / float64(total)
		hhi += share * share
	}
	n := float64(nAdvs)
	minHHI := 1.0 / n
	if hhi <= minHHI {
		return 1.0
	}
	return (1.0 - hhi) / (1.0 - minHHI)
}

// intraClusterPosVariance computes weighted mean position variance within clusters.
func intraClusterPosVariance(advs []*Advertiser) float64 {
	clusterAdvs := make(map[int][]*Advertiser)
	for _, a := range advs {
		clusterAdvs[a.Cluster] = append(clusterAdvs[a.Cluster], a)
	}
	totalVar := 0.0
	totalWeight := 0.0
	for _, cas := range clusterAdvs {
		if len(cas) < 2 {
			continue
		}
		v := computePositionVariance(cas)
		w := float64(len(cas))
		totalVar += v * w
		totalWeight += w
	}
	if totalWeight == 0 {
		return 0
	}
	return totalVar / totalWeight
}

// --- Convergence detection (per-dimension, diagnostic only) ---

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
	for i := range curr {
		// Per-dimension max center change (fair across different dimensionalities)
		maxCenterDelta := 0.0
		for d := range prev[i].Center {
			delta := math.Abs(curr[i].Center[d] - prev[i].Center[d])
			if delta > maxCenterDelta {
				maxCenterDelta = delta
			}
		}
		sigmaDelta := math.Abs(curr[i].Sigma-prev[i].Sigma) / math.Max(prev[i].Sigma, 0.01)
		priceDelta := math.Abs(curr[i].Price-prev[i].Price) / math.Max(prev[i].Price, 0.01)

		change := math.Max(maxCenterDelta, math.Max(sigmaDelta, priceDelta))
		if change > epsilon {
			return false
		}
	}
	return true
}

// --- Trial / Experiment ---

type TrialResult struct {
	AdvSurplus        float64 // per-round, per-advertiser
	PubRevenue        float64 // per-round
	FeeRevenue        float64
	WinDiversity      float64 // eval window
	ValueEfficiency   float64 // eval window
	BoundaryDiversity float64 // eval window
	PosVariance       float64 // snapshot at end
	IntraClusterVar   float64 // snapshot at end
	AvgDrift          float64 // snapshot at end
	ConvergedAtRound  int     // 0 = did not converge (diagnostic only)
}

const (
	maxRounds           = 200 // fixed for all cells — no early stopping
	evalWindow          = 20  // last 20 rounds for metrics
	impressionsPerRound = 50
	convergenceEpsilon  = 0.01
	convergenceWindow   = 5
	keywordSigma        = 0.20 // tight σ for keyword baseline (same VCG formula)
)

func runTrial(lambda float64, valueDecay float64, makeAdvs func(rng *rand.Rand, valueDecay float64) []*Advertiser, seed int64) TrialResult {
	rng := rand.New(rand.NewSource(seed))
	pub := NewPublisher(seed)
	advs := makeAdvs(rng, valueDecay)

	// Eval window accumulators
	var evalPQResults []perQueryResult
	evalWins := make(map[string]int)

	// Convergence diagnostic
	convergedAt := 0
	stableCount := 0

	for round := 1; round <= maxRounds; round++ {
		prev := snapshotAdvs(advs)

		for _, adv := range advs {
			adv.ResetRound()
		}

		queries := pub.SampleImpressions(impressionsPerRound)
		_, pqResults := runAuctionRound(advs, queries)

		for _, adv := range advs {
			adv.Adapt(lambda)
		}

		// Accumulate eval window data (last evalWindow rounds)
		if round > maxRounds-evalWindow {
			evalPQResults = append(evalPQResults, pqResults...)
			for _, adv := range advs {
				if adv.RoundWins > 0 {
					evalWins[adv.Name] += adv.RoundWins
				}
			}
		}

		// Track convergence as diagnostic (does not stop the trial)
		if convergedAt == 0 {
			if hasConverged(prev, advs, convergenceEpsilon) {
				stableCount++
				if stableCount >= convergenceWindow {
					convergedAt = round
				}
			} else {
				stableCount = 0
			}
		}
	}

	// Compute metrics
	totalAdvSurplus := 0.0
	totalFees := 0.0
	for _, a := range advs {
		totalAdvSurplus += a.Surplus()
		totalFees += a.TotalFees
	}
	avgSurplusPerRound := totalAdvSurplus / float64(len(advs)) / float64(maxRounds)

	pubRevenue := 0.0
	for _, a := range advs {
		pubRevenue += a.TotalSpend
	}
	pubRevenuePerRound := pubRevenue / float64(maxRounds)

	return TrialResult{
		AdvSurplus:        avgSurplusPerRound,
		PubRevenue:        pubRevenuePerRound,
		FeeRevenue:        totalFees,
		WinDiversity:      winDiversityFromWins(evalWins, len(advs)),
		ValueEfficiency:   valueEfficiencyFromResults(evalPQResults),
		BoundaryDiversity: boundaryDiversityFromResults(evalPQResults, len(advs)),
		PosVariance:       computePositionVariance(advs),
		IntraClusterVar:   intraClusterPosVariance(advs),
		AvgDrift:          avgDrift(advs),
		ConvergedAtRound:  convergedAt,
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

// sampleMeanStd computes mean and sample standard deviation (N-1).
func sampleMeanStd(vals []float64) (float64, float64) {
	n := len(vals)
	if n == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	m := sum / float64(n)
	if n < 2 {
		return m, 0
	}
	v := 0.0
	for _, x := range vals {
		d := x - m
		v += d * d
	}
	return m, math.Sqrt(v / float64(n-1))
}

func pct(vals []float64, p float64) float64 {
	s := make([]float64, len(vals))
	copy(s, vals)
	sort.Float64s(s)
	return s[int(p/100.0*float64(len(s)-1))]
}

func fmtStats(vals []float64) string {
	mean, std := sampleMeanStd(vals)
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

// Welch's t-test for two independent samples.
func welchT(a, b []float64) (t float64, p float64) {
	ma, sa := sampleMeanStd(a)
	mb, sb := sampleMeanStd(b)
	na, nb := float64(len(a)), float64(len(b))

	se := math.Sqrt(sa*sa/na + sb*sb/nb)
	if se == 0 {
		return 0, 1
	}
	t = (ma - mb) / se

	// Welch-Satterthwaite degrees of freedom
	v1 := sa * sa / na
	v2 := sb * sb / nb
	num := (v1 + v2) * (v1 + v2)
	den := v1*v1/(na-1) + v2*v2/(nb-1)
	if den == 0 {
		return t, 0
	}
	df := num / den

	// Two-tailed p-value: for df>30 t≈z, use normal CDF approximation
	x := t * (1 - 1/(4*df)) / math.Sqrt(1+t*t/(2*df))
	p = 2 * (1 - normalCDF(math.Abs(x)))
	return t, p
}

func normalCDF(x float64) float64 {
	return 0.5 * math.Erfc(-x/math.Sqrt2)
}

func sigStars(p float64) string {
	if p < 0.001 {
		return "***"
	}
	if p < 0.01 {
		return "**"
	}
	if p < 0.05 {
		return "*"
	}
	return "ns"
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
			Cluster:         d.Cluster,
			FreezePosition:  false,
			IdealCenter:     ideal,
			CommittedCenter: vecCopy(center),
			Center:          center,
			Price:           jitter(rng, d.BaseBid, d.BaseBid*0.10),
			Sigma:           clamp(jitter(rng, d.BaseSigma, 0.03), 0.10, 0.75),
			Budget:          1e9, // effectively infinite — remove budget depletion confound
			MaxValue:        jitter(rng, d.MaxValue, 1.0),
			ValueDecay:      valueDecay,
		}
	}
	return advs
}

// makeKeywordAdvertisers creates advertisers with tight σ and frozen positions.
// Same VCG scoring formula as embedding cells — just very selective + can't move.
func makeKeywordAdvertisers(rng *rand.Rand, valueDecay float64) []*Advertiser {
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
			Cluster:         d.Cluster,
			FreezePosition:  true,
			IdealCenter:     ideal,
			CommittedCenter: vecCopy(center),
			Center:          center,
			Price:           jitter(rng, d.BaseBid, d.BaseBid*0.10),
			Sigma:           keywordSigma, // tight σ — same VCG formula, just very selective
			Budget:          1e9,
			MaxValue:        jitter(rng, d.MaxValue, 1.0),
			ValueDecay:      valueDecay,
		}
	}
	return advs
}

// --- Experiment 0: Distance Validation ---

func runDistanceValidation() {
	name := "Experiment 0: Distance Validation"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	// Cluster membership
	clusterMembers := make(map[int][]string)
	for _, ad := range advertiserData {
		clusterMembers[ad.Cluster] = append(clusterMembers[ad.Cluster], ad.Name)
	}
	fmt.Printf("\n  K-means clusters:\n")
	for c := 0; c < 4; c++ {
		if members, ok := clusterMembers[c]; ok {
			fmt.Printf("    Cluster %d: %s\n", c, strings.Join(members, ", "))
		}
	}

	// Intra-cluster distances
	fmt.Printf("\n  Intra-cluster advertiser distances:\n")
	fmt.Printf("  %-20s %-20s  %-8s  %s\n", "Advertiser A", "Advertiser B", "cos", "cluster")
	fmt.Println("  " + strings.Repeat("-", 60))

	for i := range advertiserData {
		for j := i + 1; j < len(advertiserData); j++ {
			if advertiserData[i].Cluster == advertiserData[j].Cluster {
				cos := cosineSim(advertiserData[i].Embedding, advertiserData[j].Embedding)
				fmt.Printf("  %-20s %-20s  %-8.4f  c%d\n",
					advertiserData[i].Name, advertiserData[j].Name, cos, advertiserData[i].Cluster)
			}
		}
	}

	// Cross-cluster nearest pairs
	fmt.Printf("\n  Cross-cluster nearest pairs:\n")
	for ci := 0; ci < 4; ci++ {
		for cj := ci + 1; cj < 4; cj++ {
			bestCos := -1.0
			bestA, bestB := "", ""
			for i := range advertiserData {
				for j := range advertiserData {
					if advertiserData[i].Cluster == ci && advertiserData[j].Cluster == cj {
						cos := cosineSim(advertiserData[i].Embedding, advertiserData[j].Embedding)
						if cos > bestCos {
							bestCos = cos
							bestA = advertiserData[i].Name
							bestB = advertiserData[j].Name
						}
					}
				}
			}
			if bestCos > 0 {
				fmt.Printf("    c%d↔c%d: %-15s ↔ %-15s  cos=%.4f\n", ci, cj, bestA, bestB, bestCos)
			}
		}
	}

	// Each advertiser's closest query
	fmt.Printf("\n  Each advertiser's closest query:\n")
	fmt.Printf("  %-20s  %-8s  %s\n", "Advertiser", "cos", "Query")
	fmt.Println("  " + strings.Repeat("-", 65))

	for _, ad := range advertiserData {
		bestCos := -1.0
		bestLabel := ""
		for _, c := range impressionClusters {
			for qi, q := range c.Queries {
				cos := cosineSim(ad.Embedding, q)
				if cos > bestCos {
					bestCos = cos
					if qi < len(c.Labels) {
						bestLabel = c.Labels[qi]
					}
				}
			}
		}
		fmt.Printf("  %-20s  %-8.4f  %s\n", ad.Name, bestCos, bestLabel)
	}
}

// --- Experiment 1: Value Decay Calibration ---

func runCalibration() float64 {
	fmt.Println("\nExperiment 1: Value Decay Calibration")
	fmt.Println("=====================================")

	type distPair struct {
		adv   string
		query string
		dist2 float64
	}

	var allPairs []distPair
	var closestPerAdv []distPair

	for _, ad := range advertiserData {
		best := distPair{dist2: math.Inf(1)}
		for _, c := range impressionClusters {
			for qi, q := range c.Queries {
				dist2 := squaredDist(ad.Embedding, q)
				label := ""
				if qi < len(c.Labels) {
					label = c.Labels[qi]
				}
				p := distPair{adv: ad.Name, query: label, dist2: dist2}
				allPairs = append(allPairs, p)
				if dist2 < best.dist2 {
					best = p
				}
			}
		}
		closestPerAdv = append(closestPerAdv, best)
	}

	sort.Slice(allPairs, func(i, j int) bool { return allPairs[i].dist2 < allPairs[j].dist2 })

	fmt.Printf("\n  5 closest advertiser-query pairs:\n")
	for i := 0; i < 5 && i < len(allPairs); i++ {
		p := allPairs[i]
		fmt.Printf("    dist²=%.4f cos=%.3f  %-20s ↔ %s\n", p.dist2, 1-p.dist2/2, p.adv, p.query)
	}
	fmt.Printf("\n  5 farthest advertiser-query pairs:\n")
	for i := len(allPairs) - 5; i < len(allPairs); i++ {
		p := allPairs[i]
		fmt.Printf("    dist²=%.4f cos=%.3f  %-20s ↔ %s\n", p.dist2, 1-p.dist2/2, p.adv, p.query)
	}

	fmt.Printf("\n  Each advertiser's closest query:\n")
	for _, p := range closestPerAdv {
		fmt.Printf("    dist²=%.4f cos=%.3f  %-20s ↔ %s\n", p.dist2, 1-p.dist2/2, p.adv, p.query)
	}

	// Value at different decay values
	decays := []float64{0.2, 0.3, 0.4, 0.5}
	examples := []distPair{
		allPairs[0],
		closestPerAdv[len(closestPerAdv)/2],
		allPairs[len(allPairs)/2],
		allPairs[len(allPairs)-1],
	}

	fmt.Printf("\n  Value %% at different decays:\n")
	fmt.Printf("  %-22s %-30s  dist²   ", "Advertiser", "Query")
	for _, d := range decays {
		fmt.Printf("d=%.1f   ", d)
	}
	fmt.Println()
	fmt.Printf("  %s\n", strings.Repeat("-", 110))

	for _, p := range examples {
		fmt.Printf("  %-22s %-30s  %.4f  ", p.adv, p.query, p.dist2)
		for _, d := range decays {
			fmt.Printf("%-8.1f", 100*math.Exp(-p.dist2/d))
		}
		fmt.Println()
	}

	chosen := 0.3
	fmt.Printf("\n  Chosen valueDecay = %.2f\n", chosen)
	fmt.Printf("  Rationale: closest pair → %.0f%%, farthest → %.1f%% (ratio %.0f:1)\n",
		100*math.Exp(-allPairs[0].dist2/chosen),
		100*math.Exp(-allPairs[len(allPairs)-1].dist2/chosen),
		math.Exp(-allPairs[0].dist2/chosen)/math.Exp(-allPairs[len(allPairs)-1].dist2/chosen))

	// Print query type distribution
	globalQueryTypes = computeQueryTypes(chosen)
	counts := [3]int{}
	for _, types := range globalQueryTypes {
		for _, t := range types {
			counts[t]++
		}
	}
	fmt.Printf("\n  Query types (at decay=%.2f): specialist=%d, boundary=%d, general=%d\n",
		chosen, counts[0], counts[1], counts[2])

	return chosen
}

// --- Cell result collection ---

type CellResult struct {
	ValEff    []float64
	BoundDiv  []float64
	WinDiv    []float64
	AdvSurp   []float64
	PubRev    []float64
	Drift     []float64
	Converged []float64 // round at which converged, 0=never
}

func collectCellResult(results []TrialResult) CellResult {
	n := len(results)
	cr := CellResult{
		ValEff:    make([]float64, n),
		BoundDiv:  make([]float64, n),
		WinDiv:    make([]float64, n),
		AdvSurp:   make([]float64, n),
		PubRev:    make([]float64, n),
		Drift:     make([]float64, n),
		Converged: make([]float64, n),
	}
	for i, r := range results {
		cr.ValEff[i] = r.ValueEfficiency
		cr.BoundDiv[i] = r.BoundaryDiversity
		cr.WinDiv[i] = r.WinDiversity
		cr.AdvSurp[i] = r.AdvSurplus
		cr.PubRev[i] = r.PubRevenue
		cr.Drift[i] = r.AvgDrift
		cr.Converged[i] = float64(r.ConvergedAtRound)
	}
	return cr
}

func printCellResult(cr CellResult) {
	fmt.Printf("    Value efficiency:    %s\n", fmtStats(cr.ValEff))
	fmt.Printf("    Boundary diversity:  %s\n", fmtStats(cr.BoundDiv))
	fmt.Printf("    Win diversity:       %s\n", fmtStats(cr.WinDiv))
	fmt.Printf("    Avg surplus/round:   %s\n", fmtStats(cr.AdvSurp))
	fmt.Printf("    Pub revenue/round:   %s\n", fmtStats(cr.PubRev))
	fmt.Printf("    Avg drift:           %s\n", fmtStats(cr.Drift))

	// Convergence summary
	converged := 0
	for _, c := range cr.Converged {
		if c > 0 {
			converged++
		}
	}
	fmt.Printf("    Converged:           %d/%d trials\n", converged, len(cr.Converged))
}

// --- Experiment 2: Cell A — Keyword Baseline ---

func runKeywordBaseline(valueDecay float64) CellResult {
	name := fmt.Sprintf("Experiment 2: Cell A — Keyword Baseline (σ=%.2f, frozen)", keywordSigma)
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	const trials = 50
	results := make([]TrialResult, trials)
	for i := range results {
		results[i] = runTrial(0, valueDecay, makeKeywordAdvertisers, int64(i*7919+42))
	}

	cr := collectCellResult(results)
	fmt.Printf("\n  Trials: %d  |  σ=%.2f, frozen positions  |  λ=0  |  %d rounds\n",
		trials, keywordSigma, maxRounds)
	printCellResult(cr)
	return cr
}

// --- Experiment 3: Cell C — Embeddings, No Fees ---

func runEmbeddingsNoFees(valueDecay float64) CellResult {
	name := "Experiment 3: Cell C — Embeddings, No Fees (λ=0)"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	const trials = 50
	results := make([]TrialResult, trials)
	for i := range results {
		results[i] = runTrial(0, valueDecay, makeAdvertisers, int64(i*7919+42))
	}

	cr := collectCellResult(results)
	fmt.Printf("\n  Trials: %d  |  Normal σ, free positions  |  λ=0  |  %d rounds\n",
		trials, maxRounds)
	printCellResult(cr)
	return cr
}

// --- Experiment 4: Cell D — Embeddings, With Fees ---

func runEmbeddingsWithFees(valueDecay float64) (CellResult, float64) {
	name := "Experiment 4: Cell D — Embeddings, With Fees (λ sweep)"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	lambdas := []float64{500, 1000, 2500, 5000, 10000}
	const trials = 50

	type sweepEntry struct {
		Lambda float64
		Cell   CellResult
	}

	entries := make([]sweepEntry, len(lambdas))
	for li, lambda := range lambdas {
		trialResults := make([]TrialResult, trials)
		for i := range trialResults {
			trialResults[i] = runTrial(lambda, valueDecay, makeAdvertisers, int64(i*7919+42))
		}
		entries[li] = sweepEntry{Lambda: lambda, Cell: collectCellResult(trialResults)}
	}

	// Print sweep table
	fmt.Printf("\n  %-8s  %-12s  %-12s  %-12s  %-12s  %-12s\n",
		"Lambda", "ValEff", "BoundDiv", "WinDiv", "Surplus/rnd", "PubRev/rnd")
	fmt.Println("  " + strings.Repeat("-", 72))

	for _, e := range entries {
		mve, _ := sampleMeanStd(e.Cell.ValEff)
		mbd, _ := sampleMeanStd(e.Cell.BoundDiv)
		mwd, _ := sampleMeanStd(e.Cell.WinDiv)
		ms, _ := sampleMeanStd(e.Cell.AdvSurp)
		mr, _ := sampleMeanStd(e.Cell.PubRev)
		fmt.Printf("  %-8.0f  %-12.4f  %-12.4f  %-12.4f  %-12.4f  %-12.2f\n",
			e.Lambda, mve, mbd, mwd, ms, mr)
	}

	// Select optimal λ: highest value efficiency
	bestIdx := 0
	bestValEff := -1.0
	for i, e := range entries {
		mve, _ := sampleMeanStd(e.Cell.ValEff)
		if mve > bestValEff {
			bestValEff = mve
			bestIdx = i
		}
	}

	optLambda := entries[bestIdx].Lambda
	fmt.Printf("\n  Optimal λ = %.0f (value efficiency = %.4f)\n", optLambda, bestValEff)

	cr := entries[bestIdx].Cell
	fmt.Printf("\n  Cell D results at λ=%.0f:\n", optLambda)
	printCellResult(cr)

	return cr, optLambda
}

// --- Experiment 5: Comparison with statistical tests ---

func runComparison(cellA, cellC, cellD CellResult, optLambda float64) {
	name := "Experiment 5: Comparison Table"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	type metricRow struct {
		name   string
		a, c, d []float64
		fmt    string
	}

	rows := []metricRow{
		{"Value efficiency", cellA.ValEff, cellC.ValEff, cellD.ValEff, "%.4f"},
		{"Boundary diversity", cellA.BoundDiv, cellC.BoundDiv, cellD.BoundDiv, "%.4f"},
		{"Win diversity", cellA.WinDiv, cellC.WinDiv, cellD.WinDiv, "%.4f"},
		{"Avg surplus/round", cellA.AdvSurp, cellC.AdvSurp, cellD.AdvSurp, "%.4f"},
		{"Pub revenue/round", cellA.PubRev, cellC.PubRev, cellD.PubRev, "%.2f"},
		{"Avg drift", cellA.Drift, cellC.Drift, cellD.Drift, "%.4f"},
	}

	fmt.Printf("\n  %-22s  %-16s  %-16s  %-16s  %-6s  %-6s\n",
		"Metric", "Cell A (kw)", "Cell C (emb)", fmt.Sprintf("Cell D (λ=%.0f)", optLambda), "A↔C", "C↔D")
	fmt.Println("  " + strings.Repeat("-", 90))

	for _, r := range rows {
		ma, sa := sampleMeanStd(r.a)
		mc, sc := sampleMeanStd(r.c)
		md, sd := sampleMeanStd(r.d)
		_, pac := welchT(r.a, r.c)
		_, pcd := welchT(r.c, r.d)

		aStr := fmt.Sprintf(r.fmt+"±"+r.fmt, ma, sa)
		cStr := fmt.Sprintf(r.fmt+"±"+r.fmt, mc, sc)
		dStr := fmt.Sprintf(r.fmt+"±"+r.fmt, md, sd)
		fmt.Printf("  %-22s  %-16s  %-16s  %-16s  %-6s  %-6s\n",
			r.name, aStr, cStr, dStr, sigStars(pac), sigStars(pcd))
	}

	// Keyword regression analysis
	mveA, _ := sampleMeanStd(cellA.ValEff)
	mveC, _ := sampleMeanStd(cellC.ValEff)
	mveD, _ := sampleMeanStd(cellD.ValEff)

	fmt.Printf("\n  Keyword regression analysis:\n")
	if mveD != mveA {
		ratio := (mveC - mveA) / (mveD - mveA)
		fmt.Printf("    Value efficiency: C captures %.0f%% of D's improvement over A\n", ratio*100)
	}

	mbdA, _ := sampleMeanStd(cellA.BoundDiv)
	mbdC, _ := sampleMeanStd(cellC.BoundDiv)
	mbdD, _ := sampleMeanStd(cellD.BoundDiv)
	if mbdD != mbdA {
		ratio := (mbdC - mbdA) / (mbdD - mbdA)
		fmt.Printf("    Boundary diversity: C captures %.0f%% of D's improvement over A\n", ratio*100)
	}

	fmt.Printf("\n  Key insight: ")
	_, pAC := welchT(cellA.ValEff, cellC.ValEff)
	_, pCD := welchT(cellC.ValEff, cellD.ValEff)

	if mveD > mveC && pCD < 0.05 {
		fmt.Println("Relocation fees significantly improve value efficiency over unpenalized embeddings.")
	} else if mveC > mveA && pAC < 0.05 {
		if pCD >= 0.05 {
			fmt.Println("Embeddings significantly improve over keywords. Fees do not provide additional significant benefit.")
		} else {
			fmt.Println("Embeddings improve over keywords, and fees further improve over unpenalized embeddings.")
		}
	} else {
		fmt.Println("No statistically significant differences detected between cells.")
	}
}

// --- Main ---

func main() {
	// Exp 0: Distance validation
	runDistanceValidation()

	// Exp 1: Calibrate value decay + compute query types
	valueDecay := runCalibration()

	// Exp 2: Cell A — keyword baseline (tight σ, frozen positions)
	cellA := runKeywordBaseline(valueDecay)

	// Exp 3: Cell C — embeddings, no fees
	cellC := runEmbeddingsNoFees(valueDecay)

	// Exp 4: Cell D — embeddings, with fees (lambda sweep)
	cellD, optLambda := runEmbeddingsWithFees(valueDecay)

	// Exp 5: Comparison with statistical tests
	runComparison(cellA, cellC, cellD, optLambda)
}
