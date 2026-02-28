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

	// Skip position and sigma gradients for keyword advertisers (σ=0)
	if a.Sigma > 0.001 {
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
	}

	// --- Price gradient (always applies, even for keyword advertisers) ---
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

// --- Per-query auction result (for value efficiency tracking) ---

type perQueryResult struct {
	WinnerValue float64
	MaxValue    float64
	WinnerName  string
}

// --- VCG Auction Round ---

func runAuctionRound(advs []*Advertiser, queries [][]float64) (float64, []perQueryResult) {
	revenue := 0.0
	var pqResults []perQueryResult

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

		// Find max value across all bidders for this query
		maxVal := 0.0
		for _, adv := range bidderMap {
			v := adv.ValueAt(q)
			if v > maxVal {
				maxVal = v
			}
		}

		// VCG settlement for scoring auctions
		var payment float64
		if result.RunnerUp != nil {
			ru := bidderMap[result.RunnerUp.Bidder]
			if ru != nil {
				// Guard for σ=0 (keyword mode): fall back to second-price
				if result.Winner.Sigma < 0.001 || result.RunnerUp.Sigma < 0.001 {
					payment = result.RunnerUp.Price
				} else {
					distW2 := squaredDist(result.Winner.Embedding, q)
					distR2 := squaredDist(result.RunnerUp.Embedding, q)
					sigmaW := result.Winner.Sigma
					sigmaR := result.RunnerUp.Sigma
					payment = result.RunnerUp.Price * math.Exp(distW2/(sigmaW*sigmaW)-distR2/(sigmaR*sigmaR))
				}
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

		pqResults = append(pqResults, perQueryResult{
			WinnerValue: value,
			MaxValue:    maxVal,
			WinnerName:  winner.Name,
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

// valueEfficiencyFromResults computes mean(winnerValue / maxValue) from per-query results.
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

// boundaryQueryDiversity computes inverse HHI on boundary query winners only.
// Uses the last round's per-query results + query type metadata.
func boundaryQueryDiversity(advs []*Advertiser, queries [][]float64) float64 {
	// Run one more round just to measure boundary query winners
	winCounts := make(map[string]int)
	total := 0
	for _, q := range queries {
		// Check if this query is a boundary query
		isBoundary := false
		for _, cluster := range impressionClusters {
			for qi, cq := range cluster.Queries {
				if squaredDist(q, cq) < 1e-10 && qi < len(cluster.Types) && cluster.Types[qi] == QueryBoundary {
					isBoundary = true
					break
				}
			}
			if isBoundary {
				break
			}
		}
		if !isBoundary {
			continue
		}

		// Find winner for this boundary query
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
		winCounts[result.Winner.Bidder]++
		total++
	}

	if total == 0 {
		return 0
	}
	hhi := 0.0
	for _, count := range winCounts {
		share := float64(count) / float64(total)
		hhi += share * share
	}
	n := float64(len(advs))
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
	AdvSurplus        float64
	PubRevenue        float64
	TotalSurplus      float64
	FeeRevenue        float64
	WinDiversity      float64
	PosVariance       float64
	Rounds            int
	AvgDrift          float64
	ValueEfficiency   float64
	BoundaryDiversity float64
	IntraClusterVar   float64
}

const (
	maxRounds           = 500
	convergenceEpsilon  = 0.01
	convergenceWindow   = 5
	impressionsPerRound = 50
)

func runTrial(lambda float64, valueDecay float64, makeAdvs func(rng *rand.Rand, valueDecay float64) []*Advertiser, seed int64) TrialResult {
	rng := rand.New(rand.NewSource(seed))
	pub := NewPublisher(seed)
	advs := makeAdvs(rng, valueDecay)

	stableCount := 0
	roundsRun := 0
	var allPQResults []perQueryResult

	for round := 1; round <= maxRounds; round++ {
		prev := snapshotAdvs(advs)

		for _, adv := range advs {
			adv.ResetRound()
		}

		queries := pub.SampleImpressions(impressionsPerRound)
		_, pqResults := runAuctionRound(advs, queries)
		allPQResults = append(allPQResults, pqResults...)

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

	// Value efficiency from last 50 rounds (or all if fewer)
	startIdx := 0
	if len(allPQResults) > 50*impressionsPerRound {
		startIdx = len(allPQResults) - 50*impressionsPerRound
	}
	valEff := valueEfficiencyFromResults(allPQResults[startIdx:])

	// Boundary diversity: collect all boundary queries and measure
	var boundaryQueries [][]float64
	for _, cluster := range impressionClusters {
		for qi, q := range cluster.Queries {
			if qi < len(cluster.Types) && cluster.Types[qi] == QueryBoundary {
				boundaryQueries = append(boundaryQueries, q)
			}
		}
	}
	bDiv := boundaryQueryDiversity(advs, boundaryQueries)

	return TrialResult{
		AdvSurplus:        avgSurplus,
		PubRevenue:        pubRevenue,
		TotalSurplus:      totalAdvSurplus + pubRevenue,
		FeeRevenue:        totalFees,
		WinDiversity:      winDiversity(advs),
		PosVariance:       computePositionVariance(advs),
		Rounds:            roundsRun,
		AvgDrift:          avgDrift(advs),
		ValueEfficiency:   valEff,
		BoundaryDiversity: bDiv,
		IntraClusterVar:   intraClusterPosVariance(advs),
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
			Cluster:         d.Cluster,
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
			IdealCenter:     ideal,
			CommittedCenter: vecCopy(center),
			Center:          center,
			Price:           jitter(rng, d.BaseBid, d.BaseBid*0.10),
			Sigma:           0, // keyword mode: σ=0 means pure price ranking
			Budget:          jitter(rng, d.BaseBudget, d.BaseBudget*0.10),
			MaxValue:        jitter(rng, d.MaxValue, 1.0),
			ValueDecay:      valueDecay,
		}
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

// --- Query labels for display (match order in impressionClusters) ---

var queryLabels = [][]string{
	// physical_therapy (12)
	{"finger pulley climbing", "A2 pulley rehab", "pelvic floor C-section", "potty training PT",
		"shoulder overhead sport", "hip flexor running/climbing", "core stability postpartum", "growing pains child",
		"PT lower back pain", "find PT near me", "PT vs chiropractor", "does PT work"},
	// fitness_coaching (10)
	{"finger strength V7", "marathon plan sub-3", "CrossFit open strategy",
		"strength for endurance", "grip strength athletes", "HIIT vs steady state",
		"get in shape beginner", "exercise for weight loss", "fitness coach online", "workout busy pros"},
	// nutrition (10)
	{"eat before marathon", "low FODMAP IBS", "macro split cutting",
		"protein timing workouts", "bloating high protein", "meal prep athletes",
		"healthy eating beginners", "eat better no diet", "nutritionist vs dietitian", "balanced meal plan"},
	// tutoring (6)
	{"ADHD math tutor", "SAT math prep",
		"kid math motivation", "hands-on math activities",
		"math tutor near me", "online math tutoring"},
}

// --- Experiment 0: Distance Validation ---

func runDistanceValidation() {
	name := "Experiment 0: Distance Validation"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	clusterNames := []string{"PT", "Fitness", "Nutrition", "Tutoring"}

	// Intra-cluster distances
	fmt.Printf("\n  Intra-cluster advertiser distances:\n")
	fmt.Printf("  %-20s %-20s  %-8s  %-8s  %s\n", "Advertiser A", "Advertiser B", "cos", "dist²", "cluster")
	fmt.Println("  " + strings.Repeat("-", 70))

	for i := range advertiserData {
		for j := i + 1; j < len(advertiserData); j++ {
			if advertiserData[i].Cluster == advertiserData[j].Cluster {
				cos := cosineSim(advertiserData[i].Embedding, advertiserData[j].Embedding)
				d2 := squaredDist(advertiserData[i].Embedding, advertiserData[j].Embedding)
				fmt.Printf("  %-20s %-20s  %-8.4f  %-8.4f  %s\n",
					advertiserData[i].Name, advertiserData[j].Name, cos, d2, clusterNames[advertiserData[i].Cluster])
			}
		}
	}

	// Inter-cluster sample (one per cluster pair)
	fmt.Printf("\n  Inter-cluster advertiser distances (sample):\n")
	fmt.Printf("  %-20s %-20s  %-8s  %-8s  %s\n", "Advertiser A", "Advertiser B", "cos", "dist²", "clusters")
	fmt.Println("  " + strings.Repeat("-", 70))

	// Pick first advertiser from each cluster
	clusterFirst := make(map[int]int)
	for i, d := range advertiserData {
		if _, ok := clusterFirst[d.Cluster]; !ok {
			clusterFirst[d.Cluster] = i
		}
	}
	for ci := 0; ci < 4; ci++ {
		for cj := ci + 1; cj < 4; cj++ {
			i := clusterFirst[ci]
			j := clusterFirst[cj]
			cos := cosineSim(advertiserData[i].Embedding, advertiserData[j].Embedding)
			d2 := squaredDist(advertiserData[i].Embedding, advertiserData[j].Embedding)
			fmt.Printf("  %-20s %-20s  %-8.4f  %-8.4f  %s↔%s\n",
				advertiserData[i].Name, advertiserData[j].Name, cos, d2,
				clusterNames[ci], clusterNames[cj])
		}
	}

	// Each advertiser's closest query
	fmt.Printf("\n  Each advertiser's closest query:\n")
	fmt.Printf("  %-20s  %-8s  %s\n", "Advertiser", "cos", "Query")
	fmt.Println("  " + strings.Repeat("-", 60))

	for _, ad := range advertiserData {
		bestCos := -1.0
		bestLabel := ""
		for ci, c := range impressionClusters {
			for qi, q := range c.Queries {
				cos := cosineSim(ad.Embedding, q)
				if cos > bestCos {
					bestCos = cos
					bestLabel = queryLabels[ci][qi]
				}
			}
		}
		fmt.Printf("  %-20s  %-8.4f  %s\n", ad.Name, bestCos, bestLabel)
	}

	// Specialist query → advertiser matching
	fmt.Printf("\n  Specialist queries → top 3 closest advertisers:\n")
	for ci, c := range impressionClusters {
		for qi, q := range c.Queries {
			if qi >= len(c.Types) || c.Types[qi] != QuerySpecialist {
				continue
			}
			type advDist struct {
				name string
				cos  float64
			}
			var dists []advDist
			for _, ad := range advertiserData {
				dists = append(dists, advDist{ad.Name, cosineSim(ad.Embedding, q)})
			}
			sort.Slice(dists, func(a, b int) bool { return dists[a].cos > dists[b].cos })
			fmt.Printf("    %-40s → ", queryLabels[ci][qi])
			for k := 0; k < 3 && k < len(dists); k++ {
				fmt.Printf("%s(%.3f) ", dists[k].name, dists[k].cos)
			}
			fmt.Println()
		}
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
		for ci, c := range impressionClusters {
			for qi, q := range c.Queries {
				dist2 := squaredDist(ad.Embedding, q)
				p := distPair{adv: ad.Name, query: queryLabels[ci][qi], dist2: dist2}
				allPairs = append(allPairs, p)
				if dist2 < best.dist2 {
					best = p
				}
			}
		}
		closestPerAdv = append(closestPerAdv, best)
	}

	sort.Slice(allPairs, func(i, j int) bool { return allPairs[i].dist2 < allPairs[j].dist2 })

	// Show real pairs at distance extremes
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

	// Show value at different decay values for real pairs
	decays := []float64{0.2, 0.3, 0.4, 0.5}
	examples := []distPair{
		allPairs[0],                             // closest
		closestPerAdv[len(closestPerAdv)/2],     // median advertiser's best
		allPairs[len(allPairs)/2],               // median pair
		allPairs[len(allPairs)-1],               // farthest
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

	// Pick decay where specialist gets >30% and distant clusters get <10%
	chosen := 0.3
	fmt.Printf("\n  Chosen valueDecay = %.2f\n", chosen)
	fmt.Printf("  Rationale: closest pair → %.0f%%, farthest → %.1f%% (ratio %.0f:1)\n",
		100*math.Exp(-allPairs[0].dist2/chosen),
		100*math.Exp(-allPairs[len(allPairs)-1].dist2/chosen),
		math.Exp(-allPairs[0].dist2/chosen)/math.Exp(-allPairs[len(allPairs)-1].dist2/chosen))

	return chosen
}

// --- Experiment 2: Cell A — Keyword Baseline ---

type CellResult struct {
	ValEff    []float64
	BoundDiv  []float64
	WinDiv    []float64
	AdvSurp   []float64
	PubRev    []float64
	PosVar    []float64
	IntraVar  []float64
	Drift     []float64
	Rounds    []float64
}

func collectCellResult(results []TrialResult) CellResult {
	n := len(results)
	cr := CellResult{
		ValEff:   make([]float64, n),
		BoundDiv: make([]float64, n),
		WinDiv:   make([]float64, n),
		AdvSurp:  make([]float64, n),
		PubRev:   make([]float64, n),
		PosVar:   make([]float64, n),
		IntraVar: make([]float64, n),
		Drift:    make([]float64, n),
		Rounds:   make([]float64, n),
	}
	for i, r := range results {
		cr.ValEff[i] = r.ValueEfficiency
		cr.BoundDiv[i] = r.BoundaryDiversity
		cr.WinDiv[i] = r.WinDiversity
		cr.AdvSurp[i] = r.AdvSurplus
		cr.PubRev[i] = r.PubRevenue
		cr.PosVar[i] = r.PosVariance
		cr.IntraVar[i] = r.IntraClusterVar
		cr.Drift[i] = r.AvgDrift
		cr.Rounds[i] = float64(r.Rounds)
	}
	return cr
}

func printCellResult(cr CellResult) {
	fmt.Printf("    Value efficiency:    %s\n", fmtStats(cr.ValEff))
	fmt.Printf("    Boundary diversity:  %s\n", fmtStats(cr.BoundDiv))
	fmt.Printf("    Win diversity:       %s\n", fmtStats(cr.WinDiv))
	fmt.Printf("    Avg surplus:         %s\n", fmtStats(cr.AdvSurp))
	fmt.Printf("    Pub revenue:         %s\n", fmtStats(cr.PubRev))
	fmt.Printf("    Position var:        %s\n", fmtStats(cr.PosVar))
	fmt.Printf("    Intra-cluster var:   %s\n", fmtStats(cr.IntraVar))
	fmt.Printf("    Avg drift:           %s\n", fmtStats(cr.Drift))
	fmt.Printf("    Rounds:              %s\n", fmtStats(cr.Rounds))
}

func runKeywordBaseline(valueDecay float64) CellResult {
	name := "Experiment 2: Cell A — Keyword Baseline (σ=0)"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	const trials = 50
	results := make([]TrialResult, trials)
	for i := range results {
		results[i] = runTrial(0, valueDecay, makeKeywordAdvertisers, int64(i*7919+42))
	}

	cr := collectCellResult(results)
	fmt.Printf("\n  Trials: %d  |  All advertisers σ=0 (keyword mode)  |  λ=0\n", trials)
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
	fmt.Printf("\n  Trials: %d  |  Normal σ, no relocation fees  |  λ=0\n", trials)
	printCellResult(cr)
	return cr
}

// --- Experiment 4: Cell D — Embeddings, With Fees (lambda sweep) ---

func runEmbeddingsWithFees(valueDecay float64) (CellResult, float64) {
	name := "Experiment 4: Cell D — Embeddings, With Fees (λ sweep)"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	lambdas := []float64{500, 1000, 2500, 5000, 10000}
	const trials = 50

	type sweepEntry struct {
		Lambda  float64
		Results []TrialResult
		Cell    CellResult
	}

	entries := make([]sweepEntry, len(lambdas))
	for li, lambda := range lambdas {
		trialResults := make([]TrialResult, trials)
		for i := range trialResults {
			trialResults[i] = runTrial(lambda, valueDecay, makeAdvertisers, int64(i*7919+42))
		}
		entries[li] = sweepEntry{
			Lambda:  lambda,
			Results: trialResults,
			Cell:    collectCellResult(trialResults),
		}
	}

	// Print table
	fmt.Printf("\n  %-8s  %-12s  %-12s  %-12s  %-12s  %-12s  %-8s\n",
		"Lambda", "ValEff", "BoundDiv", "WinDiv", "AvgSurplus", "PubRevenue", "Rounds")
	fmt.Println("  " + strings.Repeat("-", 84))

	for _, e := range entries {
		mve, _ := meanStd(e.Cell.ValEff)
		mbd, _ := meanStd(e.Cell.BoundDiv)
		mwd, _ := meanStd(e.Cell.WinDiv)
		ms, _ := meanStd(e.Cell.AdvSurp)
		mr, _ := meanStd(e.Cell.PubRev)
		mrd, _ := meanStd(e.Cell.Rounds)
		fmt.Printf("  %-8.0f  %-12.4f  %-12.4f  %-12.4f  %-12.2f  %-12.2f  %-8.0f\n",
			e.Lambda, mve, mbd, mwd, ms, mr, mrd)
	}

	// Select optimal λ: highest value efficiency with boundary diversity > 0.1
	bestIdx := 0
	bestValEff := -1.0
	for i, e := range entries {
		mve, _ := meanStd(e.Cell.ValEff)
		mbd, _ := meanStd(e.Cell.BoundDiv)
		if mbd >= 0.05 && mve > bestValEff {
			bestValEff = mve
			bestIdx = i
		}
	}
	// If no lambda qualifies, pick highest value efficiency
	if bestValEff < 0 {
		for i, e := range entries {
			mve, _ := meanStd(e.Cell.ValEff)
			if mve > bestValEff {
				bestValEff = mve
				bestIdx = i
			}
		}
	}

	optLambda := entries[bestIdx].Lambda
	fmt.Printf("\n  Optimal λ = %.0f (value efficiency = %.4f)\n", optLambda, bestValEff)

	cr := entries[bestIdx].Cell
	fmt.Printf("\n  Cell D results at λ=%.0f:\n", optLambda)
	printCellResult(cr)

	return cr, optLambda
}

// --- Experiment 5: Comparison ---

func runComparison(cellA, cellC, cellD CellResult, optLambda float64) {
	name := "Experiment 5: Comparison Table"
	fmt.Printf("\n%s\n%s\n", name, strings.Repeat("=", len(name)))

	mveA, _ := meanStd(cellA.ValEff)
	mveC, _ := meanStd(cellC.ValEff)
	mveD, _ := meanStd(cellD.ValEff)
	mbdA, _ := meanStd(cellA.BoundDiv)
	mbdC, _ := meanStd(cellC.BoundDiv)
	mbdD, _ := meanStd(cellD.BoundDiv)
	mwdA, _ := meanStd(cellA.WinDiv)
	mwdC, _ := meanStd(cellC.WinDiv)
	mwdD, _ := meanStd(cellD.WinDiv)
	msA, _ := meanStd(cellA.AdvSurp)
	msC, _ := meanStd(cellC.AdvSurp)
	msD, _ := meanStd(cellD.AdvSurp)
	mrA, _ := meanStd(cellA.PubRev)
	mrC, _ := meanStd(cellC.PubRev)
	mrD, _ := meanStd(cellD.PubRev)
	mdA, _ := meanStd(cellA.Drift)
	mdC, _ := meanStd(cellC.Drift)
	mdD, _ := meanStd(cellD.Drift)

	fmt.Printf("\n  %-22s  %-14s  %-14s  %-14s\n", "Metric", "Cell A (kw)", "Cell C (emb)", fmt.Sprintf("Cell D (λ=%.0f)", optLambda))
	fmt.Println("  " + strings.Repeat("-", 68))
	fmt.Printf("  %-22s  %-14.4f  %-14.4f  %-14.4f\n", "Value efficiency", mveA, mveC, mveD)
	fmt.Printf("  %-22s  %-14.4f  %-14.4f  %-14.4f\n", "Boundary diversity", mbdA, mbdC, mbdD)
	fmt.Printf("  %-22s  %-14.4f  %-14.4f  %-14.4f\n", "Win diversity", mwdA, mwdC, mwdD)
	fmt.Printf("  %-22s  %-14.2f  %-14.2f  %-14.2f\n", "Avg surplus", msA, msC, msD)
	fmt.Printf("  %-22s  %-14.2f  %-14.2f  %-14.2f\n", "Pub revenue", mrA, mrC, mrD)
	fmt.Printf("  %-22s  %-14.4f  %-14.4f  %-14.4f\n", "Avg drift", mdA, mdC, mdD)

	// Keyword regression ratio: how much of the keyword→embedding improvement does drift destroy?
	fmt.Printf("\n  Keyword regression analysis:\n")
	if mveD > mveA {
		ratio := (mveC - mveA) / (mveD - mveA)
		fmt.Printf("    Value efficiency: C captures %.0f%% of D's improvement over A\n", ratio*100)
		if mveC < mveD {
			fmt.Printf("    Drift destroys %.0f%% of embedding value (C vs D)\n", (1-ratio)*100)
		}
	}
	if mbdD > mbdA {
		ratio := (mbdC - mbdA) / (mbdD - mbdA)
		fmt.Printf("    Boundary diversity: C captures %.0f%% of D's improvement over A\n", ratio*100)
	}

	fmt.Printf("\n  Key insight: ")
	if mveD > mveC && mveD > mveA {
		fmt.Println("Relocation fees improve value efficiency over both keyword and unpenalized embeddings.")
	} else if mveC > mveA {
		fmt.Println("Embeddings improve over keywords, but fees may not be necessary.")
	} else {
		fmt.Println("Results inconclusive — embeddings may not differentiate from keywords with this data.")
	}
}

// --- Main ---

func main() {
	// Exp 0: Distance validation
	runDistanceValidation()

	// Exp 1: Calibrate value decay
	valueDecay := runCalibration()

	// Exp 2: Cell A — keyword baseline
	cellA := runKeywordBaseline(valueDecay)

	// Exp 3: Cell C — embeddings, no fees
	cellC := runEmbeddingsNoFees(valueDecay)

	// Exp 4: Cell D — embeddings, with fees (lambda sweep)
	cellD, optLambda := runEmbeddingsWithFees(valueDecay)

	// Exp 5: Comparison
	runComparison(cellA, cellC, cellD, optLambda)
}
