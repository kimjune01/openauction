# Embedding-Space Auction Fork

This fork of [CloudX's OpenAuction](https://github.com/cloudx-io/openauction) adds optional embedding-space scoring to the core auction:

```
score = log(price) - distance² / σ²
```

Bids carry three optional fields — `embedding`, `embedding_model`, `sigma` — passed to `RunAuction` as a variadic query embedding. Bids without embeddings fall back to pure price ranking. Existing callers compile unchanged.

For context: [An Open Letter to CloudX](https://june.kim/letter-to-cloudx) · [Power Diagrams for Ad Auctions](https://june.kim/power-diagrams-ad-auctions) (series start)

## Simulation

`cmd/simulate/` runs a multi-agent market simulation validating that relocation fees (`λ · ||c_new - c_old||²`) stabilize advertiser positions in embedding space. See [It Costs Money to Move](https://june.kim/relocation-fees).

```bash
go run ./cmd/simulate/
```

## Development

See the [upstream README](https://github.com/cloudx-io/openauction) for build, test, and enclave documentation.

```bash
go test ./...
```
