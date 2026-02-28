# v2 Simulation Results (2026-02-27)

Code: `927e784` (cmd/simulate/main.go, embeddings.go)

## Experiment 0: Value Decay Calibration

384D BGE-small embeddings compress all distances into a narrow band.
dist² ranges from 0.29 to 1.32 across all 900 advertiser-query pairs.

### Real pairs at the extremes

| dist² | cos | Advertiser | Query |
|-------|-----|-----------|-------|
| 0.29 | 0.86 | Headspace | meditation app anxiety |
| 0.36 | 0.82 | Lululemon | yoga pants high waist |
| 0.36 | 0.82 | Everlane | capsule wardrobe |
| 0.43 | 0.79 | AppleWatch | GPS running watch |
| 0.46 | 0.77 | Nike | running shoes beginners |
| ... | | | |
| 1.26 | 0.37 | Dyson | therapy vs counseling |
| 1.26 | 0.37 | Gymshark | therapy vs counseling |
| 1.32 | 0.34 | Nike | therapy vs counseling |

### Each advertiser's closest query

| dist² | cos | Advertiser | Query |
|-------|-----|-----------|-------|
| 0.29 | 0.86 | Headspace | meditation app anxiety |
| 0.36 | 0.82 | Lululemon | yoga pants high waist |
| 0.36 | 0.82 | Everlane | capsule wardrobe |
| 0.43 | 0.79 | AppleWatch | GPS running watch |
| 0.46 | 0.77 | Nike | running shoes beginners |
| 0.50 | 0.75 | RogueFitness | squat rack home gym |
| 0.52 | 0.74 | PrecisionNutrition | sports nutrition fuel |
| 0.57 | 0.72 | AthleticGreens | sports nutrition fuel |
| 0.59 | 0.70 | Whoop | GPS running watch |
| 0.60 | 0.70 | Zara | summer outfit women |
| 0.61 | 0.70 | Gymshark | home gym equipment |
| 0.62 | 0.69 | Noom | digital detox plan |
| 0.63 | 0.69 | Peloton | home gym equipment |
| 0.68 | 0.66 | LaSportiva | running shoes beginners |
| 0.83 | 0.59 | Dyson | home gym equipment |

Dyson's best match (cos=0.59) is "home gym equipment" — plausible but weak.
LaSportiva's best is "running shoes beginners" at cos=0.66 — makes sense but
the model doesn't know LaSportiva is a climbing brand. The embeddings are
generated from product descriptions, not brand knowledge.

### Decay comparison

| Advertiser → Query | dist² | d=0.2 | d=0.3 | d=0.4 | d=0.5 |
|--------------------|-------|-------|-------|-------|-------|
| Headspace → meditation app | 0.29 | 24% | 38% | 49% | 56% |
| AthleticGreens → sports nutrition | 0.57 | 6% | 15% | 24% | 32% |
| Zara → adjustable dumbbells | 0.93 | 1% | 5% | 10% | 16% |
| Nike → therapy vs counseling | 1.32 | 0.1% | 1% | 4% | 7% |

Chose **decay = 0.3**. Closest pair → 38%, farthest → 1.2%, ratio 31:1.

The old v1 decay of 0.5 gave a 8:1 ratio. The calibration procedure (v2 first
attempt, decay=1.76) gave a 1.8:1 ratio — too flat, caused universal collapse.

### Open question: near-misses

The query set doesn't have near-miss pairs — queries that are semantically close
to an advertiser but not quite right. Example: Nike ↔ "hiking boots waterproof"
should be moderate distance (related but different product). The current query
clusters are topically distinct (running, yoga, fashion, strength, nutrition,
wellness), so there's no gradient between "close match" and "adjacent category."
This may flatten the value landscape in ways that don't reflect real markets.

## Experiment 1: Lambda Sweep

| λ | AdvSurplus | PubRevenue | TotalSurp | WinDiv | PosVar | Rounds |
|---|-----------|-----------|-----------|--------|--------|--------|
| 0 | -112 | 23,294 | 21,614 | 0.82 | 0.053 | 500 |
| 100 | -197 | 23,330 | 20,369 | 0.81 | 0.089 | 500 |
| 500 | -65 | 22,725 | 21,757 | 0.81 | 0.278 | 500 |
| 1000 | -56 | 22,719 | 21,874 | 0.81 | 0.336 | 500 |
| 2500 | -54 | 22,631 | 21,820 | 0.81 | 0.374 | 500 |
| 5000 | -51 | 22,593 | 21,826 | 0.80 | 0.393 | 500 |
| 10000 | -47 | 22,525 | 21,823 | 0.81 | 0.404 | 500 |
| 25000 | -47 | 22,525 | 21,823 | 0.81 | 0.404 | 500 |
| 50000 | -47 | 22,525 | 21,823 | 0.81 | 0.404 | 500 |

**Key finding: diversity is 0.81 at ALL lambda values, including λ=0.**

With sharp value decay, advertisers naturally specialize. Distant queries aren't
worth chasing, so there's no Hotelling collapse in the diversity metric. The
market self-organizes even without relocation fees.

What λ DOES control is **position variance**:
- λ=0: posVar=0.053 — positions collapse spatially toward centroid
- λ=1000: posVar=0.336 — positions stay spread out
- λ≥10000: posVar=0.404 — positions fully preserved, plateaus

Position collapse without diversity collapse means: everyone moves to the same
spot but still wins different queries because their VALUE functions differ.
Nike at the centroid still values running queries highest and bids accordingly.

λ=10000+ produces identical results — the fee is already high enough that
nobody moves. Optimal λ=1000 by total surplus (marginal).

Nothing converges — all hit 500 rounds. Convergence threshold (0.001) may be
too tight, or the gradient optimizer oscillates.

Advertiser surplus is slightly negative everywhere (-47 to -197). Advertisers
spend more than the value they capture. This might indicate the price gradient
is overshooting, or VCG payments are systematically above value.

## Experiment 2: Switching Cost Recovery

Phase 1 (λ=0 → collapse): 500 rounds, posVar=0.054, diversity=0.82
Phase 2 (λ=1000 → recovery): 5 rounds, posVar=0.054, diversity=0.81

Positions stay collapsed. Recommitting at collapsed positions + high λ just
locks everyone in place. No recovery of position variance.

But diversity was never lost (0.82 → 0.81), so there's nothing to recover.
This experiment is less interesting with sharp decay — the collapse that
motivated it doesn't happen in the diversity dimension.

## Experiment 3: Keyword/Embedding Coexistence

5 keyword (σ=0.10-0.15) + 10 embedding, λ=1000

| Metric | Value |
|--------|-------|
| Avg surplus | -351 |
| Pub revenue | 25,930 |
| Win diversity | 0.80 |
| Position var | 0.340 |
| Avg drift | 0.063 |
| Rounds | 500 |

Diversity stays high (0.80). Keyword bidders coexist with embedding bidders.
Publisher revenue is higher than pure-embedding (25,930 vs 22,719 at same λ)
because keyword bidders bid higher prices (1.5x multiplier in setup).

Surplus is more negative (-351) — keyword bidders are overpaying relative to
value captured. The 1.5x price multiplier may be too aggressive.

## Experiment 4: Dual Market

Same advertisers, same impressions, shared budget.

| Metric | Exchange A (λ=1000) | Exchange B (λ=0) |
|--------|-------------------|-----------------|
| Avg surplus | -53 | -111 |
| Revenue | 22,087 | 22,692 |
| Win diversity | 0.82 | 0.81 |
| Position var | 0.337 | 0.054 |
| Avg drift | 0.066 | 0.610 |

Both exchanges maintain diversity (~0.81). The λ=0 exchange collapses
spatially (posVar 0.054 vs 0.337) but not competitively.

Exchange B (λ=0) generates slightly more revenue (22,692 vs 22,087) — the
collapsed positions create more competitive overlap, driving up VCG payments.

Exchange A has better advertiser surplus (-53 vs -111) because advertisers
aren't wasting budget drifting to the centroid.

## Summary of findings so far

1. **Sharp value decay (0.3) prevents diversity collapse regardless of λ.**
   With the current advertiser/query set, Hotelling collapse doesn't occur
   because the advertisers are too far apart. Nike and Dyson aren't competing
   for the same queries — their value propositions are distant in embedding
   space. There's no incentive to drift because the payoff for poaching a
   distant niche is near zero.

2. **This doesn't test Hotelling.** Hotelling collapse happens between
   near-miss competitors: Nike vs Adidas, Headspace vs Calm, Noom vs
   MyFitnessPal. Advertisers whose ideal centers are close enough that
   drifting into each other's territory is profitable. The current 15
   advertisers span 6 distinct clusters with no overlapping niches. The
   simulation can't exhibit collapse because the setup doesn't contain the
   conditions for it.

3. **λ controls position variance, not diversity.** Positions collapse
   spatially at λ=0 (posVar 0.05 → everyone at centroid), but the right
   advertiser still wins because value-based pricing differentiates bids
   even at identical positions. This is an artifact of distant niches, not
   evidence that fees are unnecessary.

4. **Nothing converges in 500 rounds.** The gradient optimizer may be
   oscillating or the convergence threshold is too tight.

5. **Advertiser surplus is negative everywhere.** Needs investigation —
   may be a VCG payment calibration issue or price gradient overshooting.

## What's missing

- **Near-miss competitors**: the experiment needs advertiser pairs that
  actually compete. Nike vs Adidas (both running shoes), Headspace vs Calm
  (both meditation apps), Noom vs MyFitnessPal (both diet tracking). These
  are the pairs where Hotelling drift is rational — poaching your neighbor's
  queries is profitable because your value there is high. Without near-miss
  pairs, the simulation can't test whether relocation fees prevent the
  collapse that matters.

- **Near-miss queries**: need queries in the gap between adjacent niches.
  "Running shoes vs cross-training shoes", "yoga mat vs pilates mat" — queries
  where two advertisers have similar value and the winner depends on position.

- **Convergence tuning**: nothing converges. Need to either relax epsilon
  or investigate oscillation in the gradient updates.

- **Surplus debugging**: why is advertiser surplus consistently negative?
  Is the price gradient overshooting? Is VCG extracting too much?

---

## Next experiment: Near-miss niches

### Motivation

The CloudX letter argues that embedding-space auctions matter most for
hyperspecific small businesses: a climbing PT, a freelance translator's
financial planner, an ADHD math tutor who uses climbing metaphors. These
businesses can't survive at keyword resolution — their audience is too small.
At embedding resolution, they plant a flag at exactly their niche.

But these businesses have near-miss competitors. A climbing PT is close in
embedding space to a general sports PT, a running PT, and a CrossFit PT.
Their value propositions overlap substantially. A query like "finger injury
from bouldering" is high-value for the climbing PT and moderate-value for
the general sports PT. This is where Hotelling drift is rational: the
general sports PT can profitably drift toward climbing queries because their
value there is nonzero and the volume is attractive.

### Prior art: hyperspecific targeting works in embeddings

The premise — that embeddings can distinguish "climbing PT" from "sports PT"
at targeting resolution — is well-established:

**Embeddings encode near-complete information.** Morris & Kuleshov (EMNLP
2023, [arXiv:2310.06816](https://arxiv.org/abs/2310.06816)) showed 92%
exact text recovery from embeddings via inversion. If the original text is
recoverable, "climbing physical therapist" vs "sports physical therapist"
is certainly preserved.

**MTEB benchmarks test fine-grained discrimination.** The ArXiv clustering
task (Muennighoff et al., EACL 2023, [arXiv:2210.07316](https://arxiv.org/abs/2210.07316))
requires separating "Functional Analysis" from "Numerical Analysis" within
mathematics — closely related subdisciplines analogous to climbing PT vs
sports PT. Top models achieve meaningful scores.

**Hard negative mining trains for exactly this.** BGE-small (our model) uses
hard negative mining during fine-tuning — training to distinguish correct
documents from "deceptively similar" ones. The training signal is literally
near-miss discrimination.

**Long-tail retrieval is validated at scale.** Best Buy's embedding-based
retrieval (RecSys 2024, [arXiv:2505.01946](https://arxiv.org/abs/2505.01946))
showed 3% conversion lift on long-tail queries — the 80%+ of unique queries
with almost no interaction data. Embedding matching works precisely where
keyword matching fails.

**Production systems already operate at this granularity.** Pinterest's
interest taxonomy has 11 levels of depth. Their multi-embedding retrieval
(KDD 2025, [arXiv:2506.23060](https://arxiv.org/html/2506.23060v1))
distinguishes fine-grained tail interests like "friendship bracelets" from
broader categories, with +3% repins for non-core users. Meta's Andromeda
engine ([Meta Engineering, 2024](https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/))
uses hierarchical embedding towers for personalized ad retrieval.

**Sub-category discrimination improves dramatically with fine-tuning.** A
retail classification case study ([ionio.ai](https://www.ionio.ai/blog/how-we-fine-tuned-an-embedding-model-to-solve-retail-misclassification-a-complete-guide-code-included))
showed Pearson correlation improving from 0.528 to 0.991 after fine-tuning.
A base model scored "chocolate ice cream" vs "chocolate body lotion" at
0.849 similarity; after fine-tuning, 0.115. Near-miss discrimination is a
solved problem with domain-specific tuning.

**Ad-specific retrieval literature.** Microsoft's semantic ad matching
(Grbovic et al., 2016, [arXiv:1607.01869](https://arxiv.org/abs/1607.01869))
trained query-ad embeddings on search sessions, outperforming baselines on
relevance, coverage, and revenue — especially for cold-start (new/niche)
ads. AdsGNN (SIGIR 2021, [arXiv:2104.12080](https://arxiv.org/abs/2104.12080))
explicitly studies "long-tail ads matching" using BERT embeddings fused with
behavior graphs.

The question isn't whether embeddings can distinguish near-miss niches —
they can. The question is whether advertisers in near-miss niches exhibit
Hotelling drift in an embedding-space auction, and whether relocation fees
prevent it.

The v2 experiment didn't test this because the 15 advertisers (Nike, Dyson,
Headspace, etc.) are too far apart. Nobody drifts because nobody has a
profitable neighbor to poach from.

### Design principle: keywords are the collapsed baseline

Every advertiser in a cluster would bid on the same keyword. That's not a
hypothetical — it's how the market works today. All five PTs bid on
"physical therapy." All four fitness coaches bid on "fitness coach." The
keyword auction sees identical bids, the highest price wins everything,
and specialists are invisible.

The experiment starts from that collapsed state and asks: does embedding
resolution recover the differentiation? And if so, does it hold without
relocation fees, or does Hotelling drift degrade it back to the keyword
equilibrium?

### Advertiser design: uneven clusters

Not every niche is equally crowded. The densest cluster is the Hotelling
test bed. Sparse clusters show that uncrowded niches are naturally stable.

**Physical therapy (5 advertisers) — keyword: "physical therapy"**

All five would bid on the same keyword. At embedding resolution, they
separate into distinct specialties. This is the dense cluster where
Hotelling is most likely.

| Advertiser | Embedding description |
|---|---|
| Climbing PT | "physical therapy for rock climbers finger pulley A2 injury crimp rehab bouldering" |
| Sports PT | "sports physical therapy ACL recovery athletic injury return to play" |
| Pelvic Floor PT | "pelvic floor physical therapy postpartum incontinence diastasis recti women's health" |
| Pediatric PT | "pediatric physical therapy child motor development cerebral palsy early intervention" |
| General PT | "physical therapy rehabilitation pain management back pain recovery" |

At keyword resolution: 5 identical bids on "physical therapy," one winner.
At embedding resolution: climbing PT is far from pelvic floor PT. But
climbing PT and sports PT are close — that's where drift is profitable.

**Fitness coaching (4 advertisers) — keyword: "fitness coach"**

| Advertiser | Embedding description |
|---|---|
| Climbing Coach | "rock climbing coaching technique bouldering training movement skill beta" |
| Running Coach | "marathon running coach 5k training plan race pace interval speed" |
| CrossFit Coach | "CrossFit coaching WOD functional fitness Olympic lifting competition prep" |
| Personal Trainer | "personal trainer fitness workout strength training exercise coaching" |

Running Coach and CrossFit Coach are near-misses (both high-intensity
cardio/strength). Climbing Coach is more distinct. Personal Trainer is
the generalist who overlaps with everyone.

**Nutrition (4 advertisers) — keyword: "nutritionist"**

| Advertiser | Embedding description |
|---|---|
| Sports Dietitian | "sports dietitian endurance athlete fueling race day nutrition carb loading" |
| Gut Health Specialist | "gut health nutritionist SIBO IBS microbiome digestive wellness elimination diet" |
| Weight Loss Coach | "weight loss nutritionist calorie deficit macro counting portion control meal plan" |
| General Nutritionist | "registered dietitian nutrition counseling healthy eating balanced diet meal planning" |

Sports Dietitian and Weight Loss Coach are near-misses (both macro-focused).
Gut Health is distinct. General Nutritionist overlaps with all.

**Tutoring (2 advertisers) — keyword: "math tutor"**

| Advertiser | Embedding description |
|---|---|
| ADHD Math Tutor | "math tutoring for ADHD students hands-on learning executive function support" |
| General Math Tutor | "math tutoring algebra calculus SAT prep test preparation homework help" |

Sparse cluster. Only one near-miss pair. Tests whether a single specialist
holds position against a generalist.

15 advertisers total. Competition density: 5, 4, 4, 2.

### Query design: specialist → boundary → general

Each cluster gets queries at three resolution levels.

**Physical therapy queries (12):**

Specialist (clear winner at embedding resolution):
- "finger pulley injury from rock climbing crimping"
- "A2 pulley rehab protocol for bouldering"
- "pelvic floor exercises after C-section delivery"
- "potty training regression toddler physical therapy"

Boundary (two specialists compete — this is where drift pays off):
- "shoulder injury from overhead sport" (climbing PT or sports PT?)
- "hip flexor tightness from running and climbing" (sports PT or climbing PT?)
- "core stability exercises postpartum return to sport" (pelvic floor or sports PT?)
- "growing pains in active child athlete" (pediatric PT or sports PT?)

General (keyword-level — any PT could win):
- "physical therapy for lower back pain"
- "how to find a good physical therapist near me"
- "physical therapy vs chiropractor for pain"
- "does physical therapy actually work"

**Fitness coaching queries (10):**

Specialist:
- "how to train finger strength for climbing V7"
- "16 week marathon training plan sub 3 hours"
- "CrossFit open workout strategy tips"

Boundary:
- "strength training for endurance athletes" (running or CrossFit?)
- "grip strength training for athletes" (climbing or CrossFit?)
- "HIIT vs steady state cardio for fat loss" (CrossFit or personal trainer?)

General:
- "how to get in shape as a beginner"
- "best exercise routine for weight loss"
- "finding a good fitness coach online"
- "workout plan for busy professionals"

**Nutrition queries (10):**

Specialist:
- "what to eat before a marathon race day"
- "low FODMAP diet for IBS symptom relief"
- "macro split for cutting weight lifting"

Boundary:
- "protein timing around workouts for muscle" (sports or weight loss?)
- "bloating after high protein diet" (gut health or sports?)
- "meal prep for athletes on a budget" (sports or general?)

General:
- "healthy eating tips for beginners"
- "how to eat better without dieting"
- "should I see a nutritionist or dietitian"
- "balanced meal plan for the week"

**Tutoring queries (6):**

Specialist:
- "math tutor for child with ADHD attention issues"
- "SAT math prep tutoring intensive course"

Boundary:
- "my kid struggles with math motivation focus" (ADHD or general?)
- "hands-on math activities for kids who hate worksheets" (ADHD or general?)

General:
- "find a math tutor near me"
- "online math tutoring for middle school"

Total: 38 queries. Weighted by cluster impression share:
- PT: 35% (dense cluster, most queries)
- Fitness: 25%
- Nutrition: 25%
- Tutoring: 15%

### The 2×2 matrix

|  | **No fees (λ=0)** | **With fees (λ>0)** |
|---|---|---|
| **Keywords (σ=0)** | A: Today's market | B: N/A |
| **Embeddings (σ>0)** | C: Null hypothesis | D: Proof hypothesis |

**Cell A — Keywords, no fees.** Today's market. All PTs bid on "physical
therapy." One winner per keyword. The climbing PT, the pelvic floor PT,
and the general PT are indistinguishable to the auction. The highest
bidder wins everything. This is the collapsed baseline we're trying to
beat. Run this cell to establish the floor: win diversity, specialist win
rate, consumer relevance.

**Cell B — Keywords, with fees.** Doesn't apply. Keywords have no geometry
to drift in. σ=0 means every bid is a point, and "relocation" in keyword
space is meaningless. Skip this cell.

**Cell C — Embeddings, no fees. (Null hypothesis.)** Embeddings alone
differentiate. Each specialist starts at their niche. Initially, the
climbing PT wins climbing queries and the pelvic floor PT wins pelvic
queries. The null hypothesis says: this is sufficient. The value function
is sharp enough that drift isn't profitable, so diversity holds without
fees.

If the null hypothesis is true, Cell C ≈ Cell D. Relocation fees are
unnecessary because embedding resolution alone solves the problem. The v2
experiment (with distant niches) appeared to support this — diversity was
0.81 at all λ values. But with near-miss niches, the value landscape has
profitable drift directions. The sports PT's value for "shoulder injury
from overhead sport" is nearly as high as the climbing PT's. Drifting toward
climbing territory is rational.

If the null hypothesis fails, Cell C degrades toward Cell A over time.
Specialists converge, boundary queries monopolize, and the embedding
resolution is wasted. The market regresses to the keyword equilibrium
despite having better geometry available.

**Cell D — Embeddings, with fees. (Proof hypothesis.)** Relocation fees
make drift expensive. Each specialist stays near their ideal center.
Boundary queries are contested on value, not position — the advertiser
whose ideal center is closest wins because their value function gives them
higher true value, and VCG translates that to a higher bid. Specialist win
rates hold. Position variance holds. The embedding resolution is preserved
by the fee structure.

### What each cell predicts

| Metric | A (kw, λ=0) | C (emb, λ=0) | D (emb, λ>0) |
|---|---|---|---|
| Value efficiency | ~0.3-0.5 | Starts ~0.9, degrades? | ~0.9 stable |
| Boundary query diversity | 0 (one winner) | High initially, degrades? | High, stable |
| Intra-cluster position variance | 0 (all at keyword centroid) | Starts high, shrinks? | Stays high |

The experiment's outcome depends on Cell C:
- If C stays high: fees unnecessary, embeddings are self-stabilizing
- If C degrades toward A: fees necessary, embeddings need the fee structure
  to maintain their resolution advantage

### Metrics

**Primary: value efficiency (consumer relevance).**

```
value_efficiency = mean over queries [ value_winner(q) / max_i value_i(q) ]
```

For each query, how much of the best possible match quality did the
consumer actually get? Bounded [0, 1]. At 1.0, every query was won by
the advertiser with the highest true value for it — the consumer found
the best specialist every time. At 0.3, the consumer got someone who
can kind of help but isn't the right match.

This captures both the consumer experience and the economic argument.
The climbing PT's value for a climbing query is higher than the sports
PT's because a climber who finds the climbing PT converts at a higher
rate, books more sessions, and refers other climbers. That real revenue
difference is what lets the specialist charge more and afford more
targeted ads. When the auction matches correctly, the specialist wins
because they can pay more (their value justifies it), the consumer gets
the best match, and the publisher captures higher auction revenue. Value
efficiency measures whether the auction achieves this alignment.

When a generalist wins because the specialist drifted away: efficiency
drops. The consumer gets a PT who can help but isn't the best match. The
generalist charged less because they deliver less. Everyone left value
on the table — the consumer got a weaker match, the specialist lost a
high-value patient, and the publisher earned less from a lower clearing
price.

This also handles legitimate expansion cleanly. If the climbing PT
genuinely broadens to sports injuries and their value function widens
(they actually get good outcomes for sports patients), their value
efficiency stays high on those queries. The metric doesn't penalize
growth — it penalizes mismatches. You only score poorly if someone with
higher true value existed and lost.

Cell A prediction: ~0.3-0.5. The keyword winner is random within the
cluster, but not zero — even a random PT has some value for any PT query.

Cell D prediction: ~0.9. Positions stay near ideal centers, VCG
translates higher true value into higher bids, the best specialist wins
most queries.

Cell C prediction: this is the measurement. Starts near 0.9 (positions
are differentiated), but does it hold or degrade toward 0.3?

**Secondary metrics:**

- **Boundary query diversity:** Inverse HHI computed only on boundary
  queries within each cluster. Cell A: 0 (monopoly). Cell D: high
  (different specialists win different boundary queries). Cell C: does
  it hold or degrade?

- **Intra-cluster position variance:** Position variance within each
  cluster. Measures whether near-miss competitors stay differentiated.
  The PT cluster (5 members) should show the strongest signal.

- **Keyword regression ratio:** `(metric_C - metric_A) / (metric_D - metric_A)`.
  Applied to value efficiency. A ratio of 1.0 means embeddings without
  fees are as good as with fees. A ratio of 0.0 means embeddings without
  fees collapsed back to the keyword baseline. This is the single number
  that answers "are relocation fees necessary?"

### Implementation

Requires new embeddings. Run the 15 advertiser descriptions and 38 queries
through BGE-small-en-v1.5 and regenerate embeddings.go. The gen_embeddings.py
script handles this — update the input data.

Run three conditions: Cell A (σ=0, λ=0), Cell C (σ>0, λ=0), Cell D (σ>0,
λ=optimal from sweep). Same 50 trials each. Compare metrics across cells.

Before running the full experiment, validate distances: embed all 15
advertiser descriptions and check that intra-cluster pairs (climbing PT ↔
sports PT) are close (cos > 0.7) and inter-cluster pairs (climbing PT ↔
ADHD math tutor) are far (cos < 0.5). If intra-cluster distances aren't
small enough, the value functions won't overlap and Hotelling can't occur.

Keep the same simulation engine. Only the data, metrics, and the addition
of a keyword baseline condition change.

---

# v3.2 Simulation Results (2026-02-28)

Code: `main.go` — keyword bins vs embedding space

## What changed from v3.1

v3.1 was a design failure: the keyword baseline used fixed tight σ=0.20
vs embedding cells using σ≈0.45. The σ width dominated all results — the
comparison was tautological.

v3.2 fixes this by making the keyword cell genuinely different:

- **Cell A — Keyword Bins**: queries binned by k-means advertiser cluster
  centroid. Only advertisers in the matching cluster compete, on price alone.
  No embeddings in scoring. Second-price payment. Price-only adaptation.

- **Cell C — Embeddings, No Fees (λ=0)**: full `log(price) - dist²/σ²`
  scoring. Agents optimize bid + position + σ via gradient adaptation.
  σ is dynamic, per-advertiser.

- **Cell D — Embeddings, With Fees (λ>0)**: same as C but with relocation
  cost penalizing drift from committed position.

σ is now a per-advertiser optimizable parameter in all embedding cells
(not fixed per cell). The `FreezePosition` infrastructure was removed entirely.

## Specialist/Generalist classification

Each advertiser is classified by counting queries where they have >50% of
the max possible value. Median-split: below-median = specialist, above = generalist.

| Advertiser | Role |
|---|---|
| PediatricPT | specialist |
| GeneralPT | specialist |
| ClimbingCoach | specialist |
| RunningCoach | specialist |
| GutHealth | specialist |
| GeneralNutritionist | specialist |
| ADHDMathTutor | specialist |
| GeneralMathTutor | specialist |
| ClimbingPT | generalist |
| SportsPT | generalist |
| PelvicFloorPT | generalist |
| CrossFitCoach | generalist |
| PersonalTrainer | generalist |
| SportsDietitian | generalist |
| WeightLossCoach | generalist |

The "generalists" here are advertisers whose value function covers more
queries — they're competitive on more of the query space. The "specialists"
have concentrated value on fewer queries.

## Keyword bin diagnostics

All PT queries (cluster 0) → bin 1 (PT cluster). Most coaching queries →
bin 0 (coaching cluster). Nutrition queries split between bin 2 (sports
nutrition) and bin 3 (general nutrition/health). Math queries split between
bin 0 (ADHD/kids) and bin 3 (test prep).

The binning is coarse — all 18 PT queries go to the same bin where all 5
PT advertisers compete on price alone. This is the keyword tax: the climbing
PT and the pelvic floor PT are forced into the same auction for every PT query.

## Results

### Cell A — Keyword Bins

| Metric | Mean | Std |
|---|---|---|
| Value efficiency | 0.7471 | 0.0273 |
| Boundary diversity | 0.8152 | 0.0204 |
| Win diversity | 0.8393 | 0.0185 |
| Avg surplus/round | -1.1780 | 0.2515 |
| Specialist surplus | -0.6288 | 0.3555 |
| Generalist surplus | -1.8055 | 0.4618 |
| Pub revenue/round | 79.32 | 2.34 |

All 50 trials converged. No drift (keyword mode doesn't adapt position).

### Cell C — Embeddings, No Fees

| Metric | Mean | Std |
|---|---|---|
| Value efficiency | 0.6139 | 0.0426 |
| Boundary diversity | 0.5934 | 0.0649 |
| Win diversity | 0.6913 | 0.0604 |
| Avg surplus/round | -1.4949 | 0.2974 |
| Specialist surplus | -0.3875 | 0.5296 |
| Generalist surplus | -2.7604 | 0.7847 |
| Pub revenue/round | 73.26 | 3.25 |
| Avg drift | 0.5507 | 0.0029 |

48/50 trials converged.

### Cell D — Embeddings, With Fees (λ=2500)

Lambda sweep selected λ=2500 as optimal by value efficiency:

| λ | ValEff | BoundDiv | WinDiv | Surplus/rnd | PubRev/rnd |
|---|--------|----------|--------|-------------|------------|
| 500 | 0.6197 | 0.5871 | 0.6904 | -1.5542 | 72.90 |
| 1000 | 0.6208 | 0.5840 | 0.6911 | -1.4436 | 72.60 |
| **2500** | **0.6209** | 0.5846 | 0.6882 | -1.4072 | 72.51 |
| 5000 | 0.6189 | 0.5878 | 0.6907 | -1.4013 | 72.44 |
| 10000 | 0.6194 | 0.5863 | 0.6899 | -1.3839 | 72.41 |

Cell D at λ=2500:

| Metric | Mean | Std |
|---|---|---|
| Value efficiency | 0.6209 | 0.0427 |
| Boundary diversity | 0.5846 | 0.0706 |
| Win diversity | 0.6882 | 0.0616 |
| Avg surplus/round | -1.4072 | 0.3033 |
| Specialist surplus | -0.3661 | 0.5198 |
| Generalist surplus | -2.5971 | 0.8033 |
| Pub revenue/round | 72.51 | 3.52 |
| Avg drift | 0.0431 | 0.0025 |

All 50 trials converged.

## Comparison Table

| Metric | Cell A (kw) | Cell C (emb) | Cell D (λ=2500) | A↔C | C↔D |
|---|---|---|---|---|---|
| Value efficiency | 0.7471±0.0273 | 0.6139±0.0426 | 0.6209±0.0427 | *** | ns |
| Boundary diversity | 0.8152±0.0204 | 0.5934±0.0649 | 0.5846±0.0706 | *** | ns |
| Win diversity | 0.8393±0.0185 | 0.6913±0.0604 | 0.6882±0.0616 | *** | ns |
| Avg surplus/round | -1.1780±0.2515 | -1.4949±0.2974 | -1.4072±0.3033 | *** | ns |
| Specialist surplus | -0.6288±0.3555 | -0.3875±0.5296 | -0.3661±0.5198 | ** | ns |
| Generalist surplus | -1.8055±0.4618 | -2.7604±0.7847 | -2.5971±0.8033 | *** | ns |
| Pub revenue/round | 79.32±2.34 | 73.26±3.25 | 72.51±3.52 | *** | ns |
| Avg drift | 0.0000±0.0000 | 0.5507±0.0029 | 0.0431±0.0025 | *** | *** |

## Key Findings

### 1. Keyword bins outperform embeddings on value efficiency

This is the opposite of the v3.1 result and initially surprising. But it
makes sense: keyword bins restrict competition to within-cluster advertisers.
When the bin is correct, the "right" advertiser faces less competition from
distant advertisers who happen to have aggressive bids. The bin acts as a
hard filter that eliminates noise.

Embedding scoring lets every advertiser compete on every query through the
continuous `log(price) - dist²/σ²` formula. An advertiser far from the query
can still win if their price is high enough or σ is wide enough. This creates
more mismatches — the highest-scoring bid isn't always the highest-value
advertiser.

### 2. Embeddings significantly redistribute surplus from generalists to specialists

This is the core finding that supports the thesis:

- Specialist surplus: -0.6288 (kw) → -0.3875 (emb), **p<0.01**
- Generalist surplus: -1.8055 (kw) → -2.7604 (emb), **p<0.001**

In keyword bins, generalists compete on price in every bin they belong to.
Their breadth is an advantage — they win some queries in multiple bins.

In embedding space, specialists can position precisely at their niche and
set tight σ to avoid competing on distant queries. Generalists with wide σ
end up competing everywhere but winning nowhere because their score is
diluted by distance. The embedding mechanism transfers surplus from
generalists to specialists.

### 3. Relocation fees (C↔D) show no significant effect

All C↔D comparisons are non-significant. The λ sweep shows marginal
differences — fees reduce drift (0.55 → 0.04) but don't meaningfully
change value efficiency or surplus distribution.

This echoes the v2 finding: with sharp value decay (0.3), natural
specialization is strong enough that fees don't add much. Drift occurs
(avg drift 0.55 in Cell C) but doesn't degrade outcomes because the
value function already penalizes being far from your ideal center.

### 4. All surplus is negative

Advertiser surplus is negative in all cells (-0.37 to -2.76 per round).
Advertisers systematically overpay relative to value captured. This is
likely a VCG payment calibration issue — the price gradient adapts toward
mean query value, but VCG payments can exceed that when multiple high
bidders compete.

## Interpretation

The thesis was: "Embedding-based auctions create more surplus for
specialists, improve consumer value, possibly at the expense of
generalists."

The specialist/generalist surplus finding **supports** the redistribution
claim. Embeddings do significantly increase specialist surplus and decrease
generalist surplus.

But the value efficiency finding **contradicts** the consumer value claim.
Keyword bins actually deliver better consumer matches (0.747 vs 0.614).
This suggests the discrete binning acts as a useful filter — it's better
to restrict competition to the right cluster than to let everyone compete
via continuous scoring where aggressive bids can override relevance.

The implication: embedding-space auctions may need a hybrid approach.
Use embeddings for candidate retrieval (filter to relevant advertisers)
but use simpler scoring within the relevant set. Or: the continuous
scoring formula needs to weight distance more heavily relative to price
so that relevance dominates.

---

# v3.3 Simulation Results (2026-02-28)

Code: `gen_embeddings.py` + `main.go` — tighter specialist clustering

## What changed from v3.2

v3.2 used semantically distinct advertiser descriptions. The PT cluster
had intra-cluster cosines ranging 0.586–0.784. The tutoring pair didn't
cluster together (ADHDMathTutor landed in coaching cluster,
GeneralMathTutor in nutrition/misc).

v3.3 rewrites descriptions so within-cluster specialists share heavy
domain language and differ only on their niche modifier:

- Base: "licensed physical therapist providing rehabilitation exercise
  therapy injury recovery specializing in ..."
- Modifier: "rock climbing finger pulley" / "sports ACL athletic" / etc.

This targets intra-cluster cosine > 0.80, making:
1. k-means bins honestly coarse (close specialists land in same bin)
2. Embedding separation honestly hard (real work to distinguish near-misses)

## Clustering results

All 4 clusters correct. Tutoring pair clusters together (c3).

Intra-cluster cosine ranges:
- PT (c1): 0.759–0.875 (22/23 pairs > 0.80, only ClimbingPT↔PelvicFloorPT at 0.759)
- Coaching (c0): 0.815–0.888 (all > 0.80)
- Nutrition (c2): 0.852–0.937 (all > 0.80)
- Tutoring (c3): 0.800 (at threshold)

Cross-cluster nearest: ClimbingCoach↔ClimbingPT at 0.755 (below all
intra-cluster pairs except 1).

## Specialist/Generalist classification

Same median-split on niche query count:

| Advertiser | Role |
|---|---|
| PediatricPT | specialist |
| GeneralPT | specialist |
| CrossFitCoach | specialist |
| PersonalTrainer | specialist |
| GutHealth | specialist |
| GeneralNutritionist | specialist |
| ADHDMathTutor | specialist |
| GeneralMathTutor | specialist |
| ClimbingPT | generalist |
| SportsPT | generalist |
| PelvicFloorPT | generalist |
| ClimbingCoach | generalist |
| RunningCoach | generalist |
| SportsDietitian | generalist |
| WeightLossCoach | generalist |

## Results

### Cell A — Keyword Bins

| Metric | Mean | Std |
|---|---|---|
| Value efficiency | 0.8376 | 0.0185 |
| Boundary diversity | 0.8071 | 0.0322 |
| Win diversity | 0.8118 | 0.0272 |
| Avg surplus/round | -1.3950 | 0.1768 |
| Specialist surplus | -0.5437 | 0.5187 |
| Generalist surplus | -2.3680 | 0.5945 |
| Pub revenue/round | 73.05 | 1.87 |

All 50 trials converged.

### Cell C — Embeddings, No Fees

| Metric | Mean | Std |
|---|---|---|
| Value efficiency | 0.7716 | 0.0283 |
| Boundary diversity | 0.7245 | 0.0542 |
| Win diversity | 0.7615 | 0.0509 |
| Avg surplus/round | -1.3057 | 0.2399 |
| Specialist surplus | -0.4025 | 0.4986 |
| Generalist surplus | -2.3379 | 0.6073 |
| Pub revenue/round | 66.81 | 2.74 |
| Avg drift | 0.5873 | 0.0038 |

31/50 trials converged.

### Cell D — Embeddings, With Fees

Lambda sweep selected λ=500 as optimal by value efficiency:

| λ | ValEff | BoundDiv | WinDiv | Surplus/rnd | PubRev/rnd |
|---|--------|----------|--------|-------------|------------|
| 500 | 0.7765 | 0.7274 | 0.7674 | -1.2464 | 65.52 |
| 1000 | 0.7725 | 0.7225 | 0.7620 | -1.2034 | 65.10 |
| 2500 | 0.7718 | 0.7210 | 0.7599 | -1.1893 | 65.21 |
| 5000 | 0.7711 | 0.7193 | 0.7588 | -1.1851 | 65.06 |
| 10000 | 0.7707 | 0.7206 | 0.7596 | -1.1731 | 65.03 |

Cell D at λ=500:

| Metric | Mean | Std |
|---|---|---|
| Value efficiency | 0.7765 | 0.0301 |
| Boundary diversity | 0.7274 | 0.0655 |
| Win diversity | 0.7674 | 0.0589 |
| Avg surplus/round | -1.2464 | 0.2440 |
| Specialist surplus | -0.3255 | 0.4867 |
| Generalist surplus | -2.2989 | 0.6153 |
| Pub revenue/round | 65.52 | 2.90 |
| Avg drift | 0.1471 | 0.0070 |

31/50 trials converged.

## Comparison Table

| Metric | Cell A (kw) | Cell C (emb) | Cell D (λ=500) | A↔C | C↔D |
|---|---|---|---|---|---|
| Value efficiency | 0.8376±0.0185 | 0.7716±0.0283 | 0.7765±0.0301 | *** | ns |
| Boundary diversity | 0.8071±0.0322 | 0.7245±0.0542 | 0.7274±0.0655 | *** | ns |
| Win diversity | 0.8118±0.0272 | 0.7615±0.0509 | 0.7674±0.0589 | *** | ns |
| Avg surplus/round | -1.3950±0.1768 | -1.3057±0.2399 | -1.2464±0.2440 | * | ns |
| Specialist surplus | -0.5437±0.5187 | -0.4025±0.4986 | -0.3255±0.4867 | ns | ns |
| Generalist surplus | -2.3680±0.5945 | -2.3379±0.6073 | -2.2989±0.6153 | ns | ns |
| Pub revenue/round | 73.05±1.87 | 66.81±2.74 | 65.52±2.90 | *** | * |
| Avg drift | 0.0000±0.0000 | 0.5873±0.0038 | 0.1471±0.0070 | *** | *** |

## Key Findings

### 1. Tighter clusters make keyword bins stronger

Value efficiency jumped from 0.747 (v3.2) to 0.838 (v3.3). When
within-cluster cosine similarity is 0.80+, k-means bins capture the
right competitors reliably, and price competition within the bin
selects reasonable winners.

### 2. Specialist surplus redistribution is no longer significant

v3.2: specialist surplus -0.629 → -0.388, p<0.01
v3.3: specialist surplus -0.544 → -0.403, ns

v3.2: generalist surplus -1.806 → -2.760, p<0.001
v3.3: generalist surplus -2.368 → -2.338, ns

The direction is consistent (specialists gain, generalists lose) but
the effect size shrinks dramatically with tighter clusters. The v3.2
result was partly an artifact of loose clustering — descriptions were
distinct enough for embeddings to separate easily, but distinct enough
that k-means bins were coarser than necessary.

### 3. Keywords still outperform on value efficiency

0.838 (kw) vs 0.772 (emb), p<0.001. The hard bin filter eliminates
cross-cluster noise more effectively than continuous scoring. This
holds across all three versions (v3.1, v3.2, v3.3).

### 4. Relocation fees still show no significant effect on surplus

All C↔D surplus comparisons are ns. Fees reduce drift (0.59 → 0.15)
but don't meaningfully change value efficiency or surplus distribution.

### 5. Convergence degraded

Only 31/50 embedding trials converged (vs 48/50 in v3.2). Tighter
clusters create a harder optimization landscape — with less distance
between competitors, the gradient signal for positioning is weaker.

## Comparison: v3.2 vs v3.3

| Metric | v3.2 A↔C | v3.3 A↔C |
|---|---|---|
| Value efficiency | *** | *** |
| Specialist surplus | ** | ns |
| Generalist surplus | *** | ns |
| Pub revenue | *** | *** |

The core claim (embedding surplus redistribution) is sensitive to
cluster tightness. With distinct descriptions, the effect is strong.
With realistic shared vocabulary, the effect exists directionally but
is not statistically significant.

## Interpretation

The v3.2 finding that embeddings redistribute surplus was partly
driven by the semantic gap between advertiser descriptions. When
ClimbingPT was described as "physical therapy for rock climbers finger
pulley A2 injury crimp rehab bouldering" and GeneralPT was "physical
therapy rehabilitation pain management back pain recovery" (cos=0.78),
embedding scoring could clearly separate them. When both share
"licensed physical therapist providing rehabilitation exercise therapy
injury recovery specializing in ..." (cos=0.82+), the separation
signal weakens.

This suggests the embedding advantage depends on how much niche
information is in the advertiser's positioning signal. Real-world
advertisers who differentiate on specific technical vocabulary
(a climbing PT who says "A2 pulley crimp rehab") will benefit more
from embedding auctions than those who position generically
("physical therapy specializing in climbing"). The mechanism rewards
specificity in positioning, not just in the underlying niche.
