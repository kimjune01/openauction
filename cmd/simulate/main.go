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

func vecSub(a, b []float64) []float64 {
	r := make([]float64, len(a))
	for i := range a {
		r[i] = a[i] - b[i]
	}
	return r
}

func vecDot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
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
	IsSpecialist    bool
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

	// --- Price gradient ---
	targetPrice := 0.0
	for _, qr := range a.RoundQueries {
		targetPrice += qr.Value
	}
	targetPrice /= nq
	a.Price += lr * (targetPrice - a.Price)
	a.Price = math.Max(a.Price, 0.01)
}

// AdaptKeyword uses price-only adaptation (no positioning or sigma changes).
func (a *Advertiser) AdaptKeyword() {
	const lr = 0.02

	if len(a.RoundQueries) == 0 {
		return
	}
	nq := float64(len(a.RoundQueries))

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

func computeQueryTypes(valueDecay float64) [][]int {
	types := make([][]int, len(impressionClusters))
	for ci, c := range impressionClusters {
		types[ci] = make([]int, len(c.Queries))
		for qi, q := range c.Queries {
			vals := make([]float64, len(advertiserData))
			for ai, ad := range advertiserData {
				dist2 := squaredDist(ad.Embedding, q)
				vals[ai] = ad.MaxValue * math.Exp(-dist2/valueDecay)
			}
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

		maxVal := 0.0
		for _, adv := range advs {
			v := adv.ValueAt(q)
			if v > maxVal {
				maxVal = v
			}
		}

		// VCG settlement
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

		// Individual rationality: never pay more than your value
		if payment > value {
			payment = value
		}
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

		winner.RoundQueries = append(winner.RoundQueries, queryResult{
			Query: q, Value: value, Won: true, Payment: payment,
		})

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
	for i := range curr {
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

// --- Hotelling Trial ---

// HotellingTrialResult captures per-cluster metrics from a single trial.
type HotellingTrialResult struct {
	// Per-cluster metrics (indexed by cluster)
	CentripetalFraction []float64 // cos(drift_vector, direction_to_centroid)
	ClusterDrift        []float64 // mean drift magnitude
	ClusterPosVariance  []float64 // position variance within cluster
	ClusterSurplus      []float64 // mean surplus per advertiser per round

	// Aggregate
	ValueEfficiency  float64
	AvgDrift         float64
	ConvergedAtRound int
}

const (
	maxRounds           = 300
	evalWindow          = 30
	impressionsPerRound = 80
	convergenceEpsilon  = 0.01
	convergenceWindow   = 5
)

func runHotellingTrial(lambdas []float64, valueDecay float64, seed int64) HotellingTrialResult {
	nClusters := len(clusterTightnessLabels)
	rng := rand.New(rand.NewSource(seed))
	pub := NewPublisher(seed)
	advs := makeAdvertisers(rng, valueDecay)

	// Eval window accumulators
	var evalPQResults []perQueryResult

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
			adv.Adapt(lambdas[adv.Cluster])
		}

		if round > maxRounds-evalWindow {
			evalPQResults = append(evalPQResults, pqResults...)
		}

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

	return computeClusterMetrics(advs, evalPQResults, convergedAt, nClusters)
}

// computeClusterMetrics extracts per-cluster and aggregate metrics from a completed trial.
func computeClusterMetrics(advs []*Advertiser, evalPQResults []perQueryResult, convergedAt int, nClusters int) HotellingTrialResult {
	var result HotellingTrialResult
	result.CentripetalFraction = make([]float64, nClusters)
	result.ClusterDrift = make([]float64, nClusters)
	result.ClusterPosVariance = make([]float64, nClusters)
	result.ClusterSurplus = make([]float64, nClusters)
	result.ValueEfficiency = valueEfficiencyFromResults(evalPQResults)
	result.AvgDrift = avgDrift(advs)
	result.ConvergedAtRound = convergedAt

	// Group advertisers by cluster
	clusterAdvs := make([][]*Advertiser, nClusters)
	for _, a := range advs {
		if a.Cluster >= 0 && a.Cluster < nClusters {
			clusterAdvs[a.Cluster] = append(clusterAdvs[a.Cluster], a)
		}
	}

	for c := 0; c < nClusters; c++ {
		cas := clusterAdvs[c]
		if len(cas) == 0 {
			continue
		}

		// Cluster centroid (of committed centers — the "home" position)
		committedCenters := make([][]float64, len(cas))
		for i, a := range cas {
			committedCenters[i] = a.CommittedCenter
		}
		clusterCentroid := centroid(committedCenters)

		// Centripetal fraction: for each advertiser, cos(drift_vector, direction_to_centroid)
		cpSum := 0.0
		cpCount := 0
		driftSum := 0.0
		surplusSum := 0.0

		for _, a := range cas {
			drift := a.Drift()
			driftSum += drift
			surplusSum += a.Surplus()

			if drift < 1e-10 {
				continue
			}
			// drift vector = current - committed
			driftVec := vecSub(a.Center, a.CommittedCenter)
			// direction to centroid = centroid - committed
			toCentroid := vecSub(clusterCentroid, a.CommittedCenter)

			driftNorm := vecNorm(driftVec)
			centroidNorm := vecNorm(toCentroid)
			if driftNorm > 1e-10 && centroidNorm > 1e-10 {
				cpSum += vecDot(driftVec, toCentroid) / (driftNorm * centroidNorm)
				cpCount++
			}
		}

		if cpCount > 0 {
			result.CentripetalFraction[c] = cpSum / float64(cpCount)
		}
		result.ClusterDrift[c] = driftSum / float64(len(cas))
		result.ClusterPosVariance[c] = computePositionVariance(cas)
		result.ClusterSurplus[c] = surplusSum / float64(len(cas)) / float64(maxRounds)
	}

	return result
}

// --- Cluster density ---

// computeClusterDensity returns the mean pairwise cosine similarity within each cluster.
func computeClusterDensity() []float64 {
	nClusters := len(clusterTightnessLabels)
	density := make([]float64, nClusters)
	for c := 0; c < nClusters; c++ {
		var embs [][]float64
		for _, ad := range advertiserData {
			if ad.Cluster == c {
				embs = append(embs, ad.Embedding)
			}
		}
		if len(embs) < 2 {
			continue
		}
		sum := 0.0
		count := 0
		for i := 0; i < len(embs); i++ {
			for j := i + 1; j < len(embs); j++ {
				sum += cosineSim(embs[i], embs[j])
				count++
			}
		}
		density[c] = sum / float64(count)
	}
	return density
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

	v1 := sa * sa / na
	v2 := sb * sb / nb
	num := (v1 + v2) * (v1 + v2)
	den := v1*v1/(na-1) + v2*v2/(nb-1)
	if den == 0 {
		return t, 0
	}
	df := num / den

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

// pearsonR computes Pearson correlation coefficient.
func pearsonR(x, y []float64) float64 {
	n := len(x)
	if n < 2 || n != len(y) {
		return 0
	}
	mx, _ := sampleMeanStd(x)
	my, _ := sampleMeanStd(y)
	num := 0.0
	dx2 := 0.0
	dy2 := 0.0
	for i := range x {
		dx := x[i] - mx
		dy := y[i] - my
		num += dx * dy
		dx2 += dx * dx
		dy2 += dy * dy
	}
	den := math.Sqrt(dx2 * dy2)
	if den == 0 {
		return 0
	}
	return num / den
}

// --- Factory ---

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
			IsSpecialist:    d.IsSpecialist,
			IdealCenter:     ideal,
			CommittedCenter: vecCopy(center),
			Center:          center,
			Price:           jitter(rng, d.BaseBid, d.BaseBid*0.10),
			Sigma:           clamp(jitter(rng, d.BaseSigma, 0.03), 0.10, 0.75),
			Budget:          1e9,
			MaxValue:        jitter(rng, d.MaxValue, 1.0),
			ValueDecay:      valueDecay,
		}
	}
	return advs
}

// --- Keyword Auction (Cell A baseline) ---

// keywordBins maps each query (by cluster index, query index) to the nearest advertiser's cluster.
// This simulates keyword-based matching: each query goes to the cluster whose centroid is closest.
var keywordBins [][]int // [clusterIdx][queryIdx] → advertiser cluster index

func computeKeywordBins() {
	nClusters := len(clusterTightnessLabels)
	// Compute per-cluster centroid of advertiser embeddings
	clusterCentroids := make([][]float64, nClusters)
	for c := 0; c < nClusters; c++ {
		var embs [][]float64
		for _, ad := range advertiserData {
			if ad.Cluster == c {
				embs = append(embs, ad.Embedding)
			}
		}
		clusterCentroids[c] = centroid(embs)
	}

	keywordBins = make([][]int, len(impressionClusters))
	for ci, ic := range impressionClusters {
		keywordBins[ci] = make([]int, len(ic.Queries))
		for qi, q := range ic.Queries {
			bestCluster := 0
			bestCos := -1.0
			for c := 0; c < nClusters; c++ {
				if clusterCentroids[c] == nil {
					continue
				}
				cos := cosineSim(q, clusterCentroids[c])
				if cos > bestCos {
					bestCos = cos
					bestCluster = c
				}
			}
			keywordBins[ci][qi] = bestCluster
		}
	}
}

// runKeywordAuctionRound runs a keyword-based (cluster-filtered) second-price auction.
func runKeywordAuctionRound(advs []*Advertiser, queries []SampledQuery) (float64, []perQueryResult) {
	revenue := 0.0
	var pqResults []perQueryResult

	for _, sq := range queries {
		q := sq.Embedding
		// Determine which cluster this query belongs to via keyword bin
		targetCluster := 0
		if sq.ClusterIdx < len(keywordBins) && sq.QueryIdx < len(keywordBins[sq.ClusterIdx]) {
			targetCluster = keywordBins[sq.ClusterIdx][sq.QueryIdx]
		}

		// Only advertisers in the target cluster may bid
		var bids []core.CoreBid
		bidderMap := make(map[string]*Advertiser)
		for _, adv := range advs {
			if adv.Cluster == targetCluster && adv.Budget > adv.Price {
				bids = append(bids, adv.MakeBid())
				bidderMap[adv.Name] = adv
			}
		}
		if len(bids) == 0 {
			continue
		}

		// Price-only second-price auction (no embedding scoring)
		result := core.RunAuction(bids, nil, 0)
		if result.Winner == nil {
			continue
		}
		winner := bidderMap[result.Winner.Bidder]
		if winner == nil {
			continue
		}

		value := winner.ValueAt(q)
		maxVal := 0.0
		for _, adv := range advs {
			v := adv.ValueAt(q)
			if v > maxVal {
				maxVal = v
			}
		}

		// Second-price payment
		payment := winner.Price
		if result.RunnerUp != nil {
			payment = result.RunnerUp.Price
		}
		// Individual rationality cap
		if payment > value {
			payment = value
		}

		winner.RoundWins++
		winner.RoundSpend += payment
		winner.RoundValueWon += value
		winner.TotalWins++
		winner.TotalSpend += payment
		winner.TotalValueWon += value
		winner.Budget -= payment
		revenue += payment

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

		winner.RoundQueries = append(winner.RoundQueries, queryResult{
			Query: q, Value: value, Won: true, Payment: payment,
		})

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

// runKeywordHotellingTrial runs a keyword-based trial (Cell A).
func runKeywordHotellingTrial(valueDecay float64, seed int64) HotellingTrialResult {
	nClusters := len(clusterTightnessLabels)
	rng := rand.New(rand.NewSource(seed))
	pub := NewPublisher(seed)
	advs := makeAdvertisers(rng, valueDecay)

	var evalPQResults []perQueryResult
	convergedAt := 0
	stableCount := 0

	for round := 1; round <= maxRounds; round++ {
		prev := snapshotAdvs(advs)

		for _, adv := range advs {
			adv.ResetRound()
		}

		queries := pub.SampleImpressions(impressionsPerRound)
		_, pqResults := runKeywordAuctionRound(advs, queries)

		for _, adv := range advs {
			adv.AdaptKeyword()
		}

		if round > maxRounds-evalWindow {
			evalPQResults = append(evalPQResults, pqResults...)
		}

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

	return computeClusterMetrics(advs, evalPQResults, convergedAt, nClusters)
}

// uniformLambdas creates a slice of identical λ values for all clusters.
func uniformLambdas(lambda float64) []float64 {
	n := len(clusterTightnessLabels)
	ls := make([]float64, n)
	for i := range ls {
		ls[i] = lambda
	}
	return ls
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
	nClusters := len(clusterTightnessLabels)
	fmt.Printf("\n  Cluster assignments:\n")
	for c := 0; c < nClusters; c++ {
		if members, ok := clusterMembers[c]; ok {
			fmt.Printf("    Cluster %d (%s): %s\n", c, clusterTightnessLabels[c], strings.Join(members, ", "))
		}
	}

	// Intra-cluster cosine similarities with per-cluster summary
	fmt.Printf("\n  Intra-cluster cosine similarities:\n")
	for c := 0; c < nClusters; c++ {
		var cosVals []float64
		for i := range advertiserData {
			for j := i + 1; j < len(advertiserData); j++ {
				if advertiserData[i].Cluster == c && advertiserData[j].Cluster == c {
					cos := cosineSim(advertiserData[i].Embedding, advertiserData[j].Embedding)
					cosVals = append(cosVals, cos)
					fmt.Printf("    [c%d %s] %-15s ↔ %-15s  cos=%.4f\n",
						c, clusterTightnessLabels[c], advertiserData[i].Name, advertiserData[j].Name, cos)
				}
			}
		}
		if len(cosVals) > 0 {
			mean, _ := sampleMeanStd(cosVals)
			min, max := cosVals[0], cosVals[0]
			for _, v := range cosVals {
				if v < min {
					min = v
				}
				if v > max {
					max = v
				}
			}
			fmt.Printf("    → Cluster %d (%s) mean=%.4f  min=%.4f  max=%.4f\n\n",
				c, clusterTightnessLabels[c], mean, min, max)
		}
	}

	// Cross-cluster nearest pairs
	fmt.Printf("  Cross-cluster nearest pairs:\n")
	for ci := 0; ci < nClusters; ci++ {
		for cj := ci + 1; cj < nClusters; cj++ {
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

// --- Hotelling Analysis ---

func runHotellingAnalysis(valueDecay float64) {
	nClusters := len(clusterTightnessLabels)
	name := "Experiment 2: Hotelling Drift × Cluster Tightness"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	// 1. Cluster density table
	density := computeClusterDensity()
	fmt.Printf("\n  Cluster density (mean pairwise cosine):\n")
	for c := 0; c < nClusters; c++ {
		fmt.Printf("    Cluster %d (%s): %.4f\n", c, clusterTightnessLabels[c], density[c])
	}

	// 2. Lambda sweep (uniform) → optimal λ
	lambdaValues := []float64{500, 1000, 2500, 5000, 10000}
	const trials = 50

	fmt.Printf("\n  Lambda sweep (%d trials each, %d rounds):\n", trials, maxRounds)
	fmt.Printf("  %-8s  %-12s  %-12s  %-12s\n", "Lambda", "ValEff", "AvgDrift", "Converged")
	fmt.Println("  " + strings.Repeat("-", 50))

	type sweepEntry struct {
		lambda  float64
		results []HotellingTrialResult
	}
	var sweepResults []sweepEntry

	for _, lambda := range lambdaValues {
		trialResults := make([]HotellingTrialResult, trials)
		for i := range trialResults {
			trialResults[i] = runHotellingTrial(uniformLambdas(lambda), valueDecay, int64(i*7919+42))
		}
		sweepResults = append(sweepResults, sweepEntry{lambda, trialResults})

		veVals := make([]float64, trials)
		driftVals := make([]float64, trials)
		nConverged := 0
		for i, r := range trialResults {
			veVals[i] = r.ValueEfficiency
			driftVals[i] = r.AvgDrift
			if r.ConvergedAtRound > 0 {
				nConverged++
			}
		}
		mve, _ := sampleMeanStd(veVals)
		mdr, _ := sampleMeanStd(driftVals)
		fmt.Printf("  %-8.0f  %-12.4f  %-12.4f  %d/%d\n", lambda, mve, mdr, nConverged, trials)
	}

	// Select optimal λ by highest value efficiency
	bestIdx := 0
	bestValEff := -1.0
	for i, se := range sweepResults {
		veVals := make([]float64, trials)
		for j, r := range se.results {
			veVals[j] = r.ValueEfficiency
		}
		mve, _ := sampleMeanStd(veVals)
		if mve > bestValEff {
			bestValEff = mve
			bestIdx = i
		}
	}
	optLambda := sweepResults[bestIdx].lambda
	fmt.Printf("\n  Optimal λ = %.0f (value efficiency = %.4f)\n", optLambda, bestValEff)

	// 3. Run Cells A, C, D, E — all 50 trials, same seeds
	fmt.Printf("\n  Running Cell A (keyword), Cell C (λ=0), Cell D (λ=%.0f), Cell E (adaptive λ), %d trials each...\n", optLambda, trials)

	// Compute adaptive lambdas: λ_c = optLambda * density_c / meanDensity
	meanDensity := 0.0
	for _, d := range density {
		meanDensity += d
	}
	meanDensity /= float64(nClusters)

	adaptiveLambdas := make([]float64, nClusters)
	for c := 0; c < nClusters; c++ {
		adaptiveLambdas[c] = optLambda * density[c] / meanDensity
	}
	fmt.Printf("\n  Adaptive λ schedule (optλ=%.0f, mean density=%.4f):\n", optLambda, meanDensity)
	for c := 0; c < nClusters; c++ {
		fmt.Printf("    Cluster %d (%s): λ=%.0f (density=%.4f)\n", c, clusterTightnessLabels[c], adaptiveLambdas[c], density[c])
	}

	cellAResults := make([]HotellingTrialResult, trials)
	cellCResults := make([]HotellingTrialResult, trials)
	cellDResults := make([]HotellingTrialResult, trials)
	cellEResults := make([]HotellingTrialResult, trials)
	for i := 0; i < trials; i++ {
		seed := int64(i*7919 + 42)
		cellAResults[i] = runKeywordHotellingTrial(valueDecay, seed)
		cellCResults[i] = runHotellingTrial(uniformLambdas(0), valueDecay, seed)
		cellDResults[i] = runHotellingTrial(uniformLambdas(optLambda), valueDecay, seed)
		cellEResults[i] = runHotellingTrial(adaptiveLambdas, valueDecay, seed)
	}

	// Helper to extract per-cluster metric from trial results
	extractCluster := func(results []HotellingTrialResult, c int, getter func(HotellingTrialResult) float64) []float64 {
		vals := make([]float64, len(results))
		for i, r := range results {
			vals[i] = getter(r)
		}
		return vals
	}

	// 4. Per-cluster comparison table: A vs C vs D vs E
	for c := 0; c < nClusters; c++ {
		fmt.Printf("\n  ── Cluster %d (%s, density=%.4f) ──\n", c, clusterTightnessLabels[c], density[c])

		surpA := extractCluster(cellAResults, c, func(r HotellingTrialResult) float64 { return r.ClusterSurplus[c] })
		surpC := extractCluster(cellCResults, c, func(r HotellingTrialResult) float64 { return r.ClusterSurplus[c] })
		surpD := extractCluster(cellDResults, c, func(r HotellingTrialResult) float64 { return r.ClusterSurplus[c] })
		surpE := extractCluster(cellEResults, c, func(r HotellingTrialResult) float64 { return r.ClusterSurplus[c] })

		driftC := extractCluster(cellCResults, c, func(r HotellingTrialResult) float64 { return r.ClusterDrift[c] })
		driftD := extractCluster(cellDResults, c, func(r HotellingTrialResult) float64 { return r.ClusterDrift[c] })

		cpC := extractCluster(cellCResults, c, func(r HotellingTrialResult) float64 { return r.CentripetalFraction[c] })
		cpD := extractCluster(cellDResults, c, func(r HotellingTrialResult) float64 { return r.CentripetalFraction[c] })

		mSA, _ := sampleMeanStd(surpA)
		mSC, _ := sampleMeanStd(surpC)
		mSD, _ := sampleMeanStd(surpD)
		mSE, _ := sampleMeanStd(surpE)

		mDrC, _ := sampleMeanStd(driftC)
		mDrD, _ := sampleMeanStd(driftD)
		_, pDr := welchT(driftC, driftD)

		mCpC, _ := sampleMeanStd(cpC)
		mCpD, _ := sampleMeanStd(cpD)
		_, pCp := welchT(cpC, cpD)

		_, pSurpAC := welchT(surpA, surpC)
		_, pSurpAD := welchT(surpA, surpD)
		_, pSurpDE := welchT(surpD, surpE)

		fmt.Printf("    %-24s  %-10s  %-10s  %-10s  %-10s\n", "Metric", "Cell A", "Cell C", "Cell D", "Cell E")
		fmt.Println("    " + strings.Repeat("-", 68))
		fmt.Printf("    %-24s  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n", "Surplus/round/adv", mSA, mSC, mSD, mSE)
		fmt.Printf("    %-24s  %-10s  %-10.4f  %-10.4f  %-10s\n", "Drift magnitude", "n/a", mDrC, mDrD, "n/a")
		fmt.Printf("    %-24s  %-10s  %-10.4f  %-10.4f  %-10s\n", "Centripetal fraction", "n/a", mCpC, mCpD, "n/a")
		fmt.Printf("    Surplus A↔C: p=%.4f %s  A↔D: p=%.4f %s  D↔E: p=%.4f %s\n",
			pSurpAC, sigStars(pSurpAC), pSurpAD, sigStars(pSurpAD), pSurpDE, sigStars(pSurpDE))
		fmt.Printf("    Drift C↔D: p=%.4f %s  Centripetal C↔D: p=%.4f %s\n", pDr, sigStars(pDr), pCp, sigStars(pCp))
	}

	// 5. Aggregate comparison: A vs C vs D vs E
	fmt.Printf("\n  ── Aggregate Comparison ──\n")
	veA := make([]float64, trials)
	veC := make([]float64, trials)
	veD := make([]float64, trials)
	veE := make([]float64, trials)
	for i := 0; i < trials; i++ {
		veA[i] = cellAResults[i].ValueEfficiency
		veC[i] = cellCResults[i].ValueEfficiency
		veD[i] = cellDResults[i].ValueEfficiency
		veE[i] = cellEResults[i].ValueEfficiency
	}
	mVeA, _ := sampleMeanStd(veA)
	mVeC, _ := sampleMeanStd(veC)
	mVeD, _ := sampleMeanStd(veD)
	mVeE, _ := sampleMeanStd(veE)

	fmt.Printf("    %-24s  %-10s  %-10s  %-10s  %-10s\n", "Metric", "Cell A", "Cell C", "Cell D", "Cell E")
	fmt.Println("    " + strings.Repeat("-", 68))
	fmt.Printf("    %-24s  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n", "Value efficiency", mVeA, mVeC, mVeD, mVeE)

	// Per-cell aggregate surplus
	aggSurpA := make([]float64, trials)
	aggSurpC := make([]float64, trials)
	aggSurpD := make([]float64, trials)
	aggSurpE := make([]float64, trials)
	for i := 0; i < trials; i++ {
		for c := 0; c < nClusters; c++ {
			aggSurpA[i] += cellAResults[i].ClusterSurplus[c]
			aggSurpC[i] += cellCResults[i].ClusterSurplus[c]
			aggSurpD[i] += cellDResults[i].ClusterSurplus[c]
			aggSurpE[i] += cellEResults[i].ClusterSurplus[c]
		}
		aggSurpA[i] /= float64(nClusters)
		aggSurpC[i] /= float64(nClusters)
		aggSurpD[i] /= float64(nClusters)
		aggSurpE[i] /= float64(nClusters)
	}
	mSA, _ := sampleMeanStd(aggSurpA)
	mSC, _ := sampleMeanStd(aggSurpC)
	mSD, _ := sampleMeanStd(aggSurpD)
	mSE, _ := sampleMeanStd(aggSurpE)
	fmt.Printf("    %-24s  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n", "Avg surplus/round/adv", mSA, mSC, mSD, mSE)

	_, pAC := welchT(aggSurpA, aggSurpC)
	_, pAD := welchT(aggSurpA, aggSurpD)
	_, pCD := welchT(aggSurpC, aggSurpD)
	_, pDE := welchT(aggSurpD, aggSurpE)
	fmt.Printf("    Surplus: A↔C p=%.4f %s  A↔D p=%.4f %s  C↔D p=%.4f %s  D↔E p=%.4f %s\n",
		pAC, sigStars(pAC), pAD, sigStars(pAD), pCD, sigStars(pCD), pDE, sigStars(pDE))

	// 6. Tightness × fee effect correlation (n=5)
	fmt.Printf("\n  ── Tightness × Fee Effect Correlation (n=%d) ──\n", nClusters)

	tightnessVals := make([]float64, nClusters)
	surplusImprovement := make([]float64, nClusters)
	driftReduction := make([]float64, nClusters)
	cpChange := make([]float64, nClusters)

	for c := 0; c < nClusters; c++ {
		tightnessVals[c] = density[c]

		surpC := extractCluster(cellCResults, c, func(r HotellingTrialResult) float64 { return r.ClusterSurplus[c] })
		surpD := extractCluster(cellDResults, c, func(r HotellingTrialResult) float64 { return r.ClusterSurplus[c] })
		driftC := extractCluster(cellCResults, c, func(r HotellingTrialResult) float64 { return r.ClusterDrift[c] })
		driftD := extractCluster(cellDResults, c, func(r HotellingTrialResult) float64 { return r.ClusterDrift[c] })
		cpCv := extractCluster(cellCResults, c, func(r HotellingTrialResult) float64 { return r.CentripetalFraction[c] })
		cpDv := extractCluster(cellDResults, c, func(r HotellingTrialResult) float64 { return r.CentripetalFraction[c] })

		mSCv, _ := sampleMeanStd(surpC)
		mSDv, _ := sampleMeanStd(surpD)
		surplusImprovement[c] = mSDv - mSCv

		mDCv, _ := sampleMeanStd(driftC)
		mDDv, _ := sampleMeanStd(driftD)
		driftReduction[c] = mDCv - mDDv

		mCCv, _ := sampleMeanStd(cpCv)
		mCDv, _ := sampleMeanStd(cpDv)
		cpChange[c] = mCDv - mCCv
	}

	fmt.Printf("    %-12s  %-10s  %-14s  %-14s  %-14s\n",
		"Cluster", "Density", "ΔSurplus(D-C)", "ΔDrift(C-D)", "ΔCentripetal")
	fmt.Println("    " + strings.Repeat("-", 70))
	for c := 0; c < nClusters; c++ {
		fmt.Printf("    %-12s  %-10.4f  %-14.4f  %-14.4f  %-14.4f\n",
			clusterTightnessLabels[c], tightnessVals[c], surplusImprovement[c], driftReduction[c], cpChange[c])
	}

	rSurplus := pearsonR(tightnessVals, surplusImprovement)
	rDrift := pearsonR(tightnessVals, driftReduction)
	rCp := pearsonR(tightnessVals, cpChange)

	fmt.Printf("\n    Pearson r (density × ΔSurplus):      %.4f  (n=%d)\n", rSurplus, nClusters)
	fmt.Printf("    Pearson r (density × ΔDrift):        %.4f  (n=%d)\n", rDrift, nClusters)
	fmt.Printf("    Pearson r (density × ΔCentripetal):  %.4f  (n=%d)\n", rCp, nClusters)

	// 7. Cell D vs Cell E comparison (uniform vs adaptive)
	fmt.Printf("\n  ── Cell D (uniform λ) vs Cell E (adaptive λ) ──\n")
	for c := 0; c < nClusters; c++ {
		surpD := extractCluster(cellDResults, c, func(r HotellingTrialResult) float64 { return r.ClusterSurplus[c] })
		surpE := extractCluster(cellEResults, c, func(r HotellingTrialResult) float64 { return r.ClusterSurplus[c] })
		mDv, _ := sampleMeanStd(surpD)
		mEv, _ := sampleMeanStd(surpE)
		_, p := welchT(surpD, surpE)
		fmt.Printf("    Cluster %d (%s): D=%.4f  E=%.4f  Δ=%.4f  p=%.4f %s\n",
			c, clusterTightnessLabels[c], mDv, mEv, mEv-mDv, p, sigStars(p))
	}

	// 8. Key findings
	fmt.Printf("\n  ── Key Findings ──\n")

	allCpPositive := true
	for c := 0; c < nClusters; c++ {
		cpCv := extractCluster(cellCResults, c, func(r HotellingTrialResult) float64 { return r.CentripetalFraction[c] })
		mCp, _ := sampleMeanStd(cpCv)
		if mCp <= 0 {
			allCpPositive = false
			fmt.Printf("    [!] Cluster %d (%s): centripetal fraction = %.4f (NOT positive)\n",
				c, clusterTightnessLabels[c], mCp)
		} else {
			fmt.Printf("    [+] Cluster %d (%s): centripetal fraction = %.4f (positive → Hotelling drift)\n",
				c, clusterTightnessLabels[c], mCp)
		}
	}

	if allCpPositive {
		fmt.Println("    → Hotelling drift detected in all clusters under free positioning (Cell C)")
	}

	if rSurplus > 0.5 {
		fmt.Printf("    → Fee effect scales with tightness (r=%.2f, n=%d): tighter clusters benefit more from fees\n", rSurplus, nClusters)
	} else if rSurplus < -0.5 {
		fmt.Printf("    → Fee effect inversely scales with tightness (r=%.2f, n=%d): looser clusters benefit more\n", rSurplus, nClusters)
	} else {
		fmt.Printf("    → No strong correlation between tightness and fee effect (r=%.2f, n=%d)\n", rSurplus, nClusters)
	}

	// Adaptive λ finding
	_, pDEagg := welchT(aggSurpD, aggSurpE)
	if pDEagg < 0.05 {
		fmt.Printf("    → Adaptive λ significantly different from uniform (p=%.4f)\n", pDEagg)
	} else {
		fmt.Printf("    → Adaptive λ not significantly different from uniform (p=%.4f)\n", pDEagg)
	}
}

// --- Main ---

func main() {
	// Exp 0: Distance validation
	runDistanceValidation()

	// Exp 1: Calibrate value decay + compute query types
	valueDecay := runCalibration()

	// Compute keyword bins for Cell A
	computeKeywordBins()

	// Exp 2: Hotelling drift × cluster tightness analysis
	runHotellingAnalysis(valueDecay)
}
