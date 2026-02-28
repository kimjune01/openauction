# Experiment Design: Relocation Fee Dividend

## Hypothesis

Relocation fees (`λ · ‖c_new - c_old‖²`) are a net positive externality:
the total market surplus (advertiser profit + publisher revenue) is higher
with fees than without, and every participant class is individually better off.
The fee isn't a tax — it's a coordination mechanism that creates value.

## Market structure

- **Embedding space**: 384D, real embeddings from BGE-small-en-v1.5 (open-weight)
- **15 advertisers**: real product descriptions embedded through same model
- **6 impression clusters**: running (18%), yoga (14%), fashion (22%), strength (15%),
  nutrition (17%), wellness (14%) — 10 real search queries each, 60 total
- **50 impressions/round**, sampled by cluster weight
- **Run until equilibrium** (max 500 rounds), **50 trials** per experiment

## Auction mechanics

Uses `core.RunAuction` from CloudX's openauction. Scoring:
```
score = log(price) - dist²/σ²
```

**Settlement (VCG for scoring auctions):**
```
payment = p_runner · exp(dist_w²/σ_w² - dist_r²/σ_r²)
```
Where `dist_w`, `σ_w` are the winner's distance and sigma for this query, and
`dist_r`, `σ_r` are the runner-up's. This is incentive-compatible: bidding true
value is dominant strategy in a TEE where bids aren't revealed.

If no runner-up, winner pays their own price (degenerate case).

**Convergence criterion:**
A trial runs until equilibrium or max 500 rounds. Equilibrium is detected when
the maximum parameter change across all active agents falls below threshold for
5 consecutive rounds:
```
max_over_agents(‖Δcenter‖ + |Δσ/σ| + |Δprice/price|) < ε
```
where ε is small (e.g., 0.001). This captures position stability, σ stability,
and price stability simultaneously using relative changes so the scales are
comparable.

## Agent model

Each advertiser has:
- **Ideal center** (fixed): embedding of product description. Represents what they sell.
- **Current center** (optimizable): where they bid in embedding space.
- **Committed center** (fixed per trial): declared position at start. Drift fees
  measured from here.
- **σ** (optimizable): bidding reach / targeting breadth.
- **Price** (optimizable): bid price. In VCG, optimal = true value.
- **Budget** (depletes): finite. Primary brake on spending.
- **MaxValue** (fixed): maximum conversion value for a perfect-match impression.
- **Value function**: `value(q) = maxValue · f(dist(q, idealCenter))` where `f`
  is calibrated from Experiment 0 (empirical relevance decay in BGE embedding
  space). CTR proxy — impressions far from what you sell have low conversion value.
  Shape and decay rate derived from data, not assumed.

### Adaptation: gradient ascent on expected profit

After each round, the agent observes its wins, payments, and values. It computes
gradients for all three parameters from the round's query data:

**Bid filtering (pre-filter):**
Agent only bids on queries where `value(q) > bidThreshold` (e.g., 5% of maxValue).
This is the DSP pre-filter: don't bid on impressions you can't convert. Nike
doesn't bid on meditation queries because `value(meditation) ≈ 0` for Nike.

This constrains the agent's information scope to its locale. A specialist near
running queries only sees running queries. A collapsed agent near the centroid
sees moderate-value queries from all clusters. Gradient is computed only from
queries the agent actually bids on — not the full distribution.

**Position gradient (from bid queries only):**
```
∂profit/∂center ∝ Σ_{q: bid on} value(q) · (q - center)
```
Move toward queries weighted by their value to this advertiser. High-CTR
impressions pull harder than low-CTR ones. Agent cannot see queries outside
its bidding radius, so the gradient reflects local information only.

**σ gradient (from bid queries only):**
```
∂score/∂σ = 2 · dist² / σ³  (for each query bid on)
```
Weighted by marginal profit `(value(q) - avgPayment)`:
- Positive for profitable distant queries → widen (capture more volume)
- Negative for unprofitable distant queries → narrow (avoid junk wins)

σ naturally finds its equilibrium. No clamp needed except >0 for numerical
stability. If σ→∞, agent wins everything but at terrible ROAS, and the gradient
pulls it back.

**Price gradient:**
In VCG/second-price, bidding true value is dominant. So price should converge
toward the agent's expected value for queries in its bidding region:
```
target_price ≈ E[value(q) | q bid on]
```
Agent adjusts price toward this target each round. If current price is too low
(losing profitable queries), raise. If too high (budget draining too fast), lower.

**Learning rate:**
Single lr for all agents (e.g., 0.025). This is a numerical parameter of the
optimizer, not a market parameter. All agents follow the same gradient — they
differentiate because they have different ideal centers, different maxValues,
and different budgets, not because of different "strategies."

With convergence-based stopping, lr only affects speed, not destination.

### Relocation fees

Cumulative drift from committed position:
```
cost = λ · (‖new_center - committed‖² - ‖current - committed‖²)
```
- Moving away from commitment: increasingly expensive
- Returning toward commitment: free (cost clamped to 0)
- Agent only moves if expected value gain > relocation cost

## Pre-registered metrics

Chosen before running. Not to be changed after seeing results.

**Surplus metrics (the hypothesis test):**

1. **Advertiser surplus** = total value won - total spend (including fees),
   averaged across advertisers. This is what advertisers take home. If fees
   are a positive externality, this is HIGHER with fees than without — despite
   advertisers paying more.

2. **Publisher revenue** = total auction clearing prices. What the publisher
   earns from selling impressions.

3. **Total market surplus** = advertiser surplus + publisher revenue. The pie.
   If fees grow the pie (not just redistribute), this increases with λ.

4. **Fee revenue** = total relocation fees collected. The new revenue category.
   This is the "dividend" — money that didn't exist at λ=0.

**Structure metrics (diagnostics, not the claim):**

5. **Win diversity** = normalized inverse HHI of wins.
   0 = monopoly, 1 = equal. Diagnostic for market health.

6. **Position variance** = mean squared distance of centers from their centroid.
   Diagnostic for whether agents differentiate.

## Experiments

### Experiment 0: Calibrate value decay from embedding geometry

Before running any market simulation, measure the actual relevance falloff in
our embedding space. Compute `cosine_similarity(advertiser, query)` for all
15×60 = 900 pairs. This gives us the empirical distribution of advertiser-query
distances.

From this data:
- Plot the distribution of L2 distances (or equivalently, 1 - cosine for
  normalized embeddings)
- Identify the distance ranges: same-cluster (e.g., Nike ↔ running queries),
  adjacent-cluster (Nike ↔ strength queries), unrelated (Nike ↔ wellness queries)
- Fit the value function to the empirical distance distribution rather than
  assuming `exp(-dist²/0.5)` with an arbitrary decay constant

The output is a calibrated `valueDecay` parameter (or a non-parametric value
function) derived from data, not assumed. If the empirical falloff is steeper
or shallower than Gaussian, we use whatever shape the data shows.

This also serves as a sensitivity check: run the λ sweep with 2-3 different
decay values spanning the plausible range, and verify the qualitative result
(collapse vs. differentiation) holds across all of them.

### Experiment 1: λ sweep

Sweep λ from 0 to 50,000 in steps (e.g., 0, 100, 500, 1000, 2500, 5000, 10000,
25000, 50000). For each λ, run 50 trials to convergence. Plot all metrics vs λ.

The hypothesis predicts: as λ increases from 0, advertiser surplus, publisher
revenue, AND total market surplus all increase — the pie grows. At some λ, they
plateau or decline (fees too high, agents frozen). The optimal λ is where total
surplus peaks.

If advertiser surplus decreases with λ (fees are a net cost), the hypothesis is
falsified. If publisher revenue decreases (fees suppress bidding), also falsified.
The hypothesis requires ALL participants to benefit.

### Experiment 2: Switching cost

λ=0 until equilibrium (market collapses), then switch to λ from sweep. Continue
until new equilibrium. Shows whether introducing fees into an already-collapsed
market produces recovery, and how long recovery takes.

### Experiment 3: Keyword vs embedding coexistence

5 keyword-like bidders (narrow σ, initialized near specific query clusters) +
10 embedding bidders (broad σ). Shows whether the scoring function naturally
accommodates both strategies.

### Experiment 4: Dual market

Same 15 advertisers, same impressions, shared budget. Two exchanges:
- Exchange A: λ = value from sweep that maximizes total surplus
- Exchange B: λ = 0

This tests the competitive argument: can a low-fee exchange attract advertisers
by undercutting? If Exchange A produces higher advertiser surplus despite
charging fees, rational advertisers prefer it. The fee-charging exchange offers
a better product (a market with positive externalities), not just a higher price.

## Sensitivity checks

Quick robustness checks, not full experiments. Each re-runs the λ sweep with
one parameter varied. Results go in an appendix table.

1. **Budget tightness**: halve all budgets. Verify budgets actually bind and
   agents drop out. If budgets never bind, the "budget is the brake" argument
   is hollow.
2. **Bid threshold**: vary from 1% to 20% of maxValue. Check whether information
   scope changes the collapse dynamics.

## What's deliberately simplified

- **15 agents, not thousands.** Hotelling requires only 2. More agents make
  collapse more dramatic, not less.
- **Stationary impression distribution.** Shifting distribution would make
  relocation fees more important (constantly moving density peaks), not less.
- **No entry/exit.** First position declaration incurs no fee, so entry is free.
- **Single embedding model.** Real markets may have multiple. The protocol
  specifies `embedding_model` for this reason.
- **Second-price (VCG), not first-price.** Correct for TEE-based exchange where
  bids are hidden. First-price would require modeling bid shading.
