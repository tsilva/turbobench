# Usage and evidence

Install and basic commands are in the [README](../README.md). Use the
[parity guide](parity.md) for development checks and exact-wheel certification,
and the [two-host workflow guide](comparison-workflow.md) for policy-backed
showcases in the development checkout.

## Workloads and provider selectors

The published profiles are `supermario/world1-v1`, `breakout/start-v1`, and
`vizdoom/basic-v1`. The development checkout also includes
`breakout/firstwall-policy-v1` for captured-policy comparisons. A profile's same
ID selects the performance workload and parity contract. Run
`turbobench profiles list` to inspect the profiles in your installed version.

Provider references accept:

- `provider` or `provider@latest` for the latest eligible package release.
- `provider@VERSION` for an exact package version.
- `provider@artifact:/absolute/path.whl` for a final local distribution.
- `provider@checkout:/absolute/path` for a local source checkout.

`latest` excludes prereleases, yanked releases, incompatible artifacts, and
releases still inside the seven-day quarantine. Every run resolves its selectors
to exact, isolated, hash-recorded artifacts. Local checkout and artifact selectors
are supported by independent comparisons and parity, not the two-host release
workflow.

## Breakout comparisons

The `breakout/start-v1` profile compares the Atari 2600 Breakout `Start` workload
against original Stable Retro or Stable Retro Turbo. Supply lawful local assets
with `TURBOBENCH_ROM_PATH`, `TURBOBENCH_ASSET_ROOT`, or `RETRO_DATA_PATH`.
Run `turbobench doctor breakout/start-v1` to check their availability.

Run these examples from the repository root or an installed CLI environment.
Replace `VERSION` with an exact candidate release; `@latest` and clean
`@checkout:/absolute/path` selectors are also supported.

```bash
turbobench compare breakout/start-v1 \
  --left env-breakoutatari2600-turbo-native@VERSION \
  --right stable-retro@1.0.1 \
  --output turbobench-results/breakout-vs-stable-retro

turbobench compare breakout/start-v1 \
  --left env-breakoutatari2600-turbo-native@VERSION \
  --right env-stableretro-turbo@VERSION \
  --output turbobench-results/breakout-vs-stable-retro-turbo
```

## Reuse parity evidence

Comparisons check only their selected pair. They never prepare an original
authority unless it is selected. A compatible parity receipt can replace matching
correctness shapes; missing shapes are checked directly during comparison.

One receipt covers an authority/candidate pair. Two receipts against the same
exact authority can establish equality between two Turbo providers. Each receipt
must bind the selected artifacts and a compatible workload. Replace the version
and receipt paths in this example:

```bash
turbobench compare breakout/start-v1 \
  --left env-breakoutatari2600-turbo-native@VERSION \
  --right env-stableretro-turbo@VERSION \
  --parity-receipt /external/evidence/native-receipt \
  --parity-receipt /external/evidence/stable-retro-turbo-receipt
```

Release parity requires the exact final local wheel and the full workload on the
chosen canonical host. Checkout snapshots include tracked edits and nonignored
untracked source; `--allow-dirty --quick` checks remain diagnostic. See
[cross-provider parity](parity.md) for commands and receipt validation.

## Measurement and validity

For the standard profiles, `compare --quick` checks selected-pair correctness and
runs two timing pairs at one environment. Full comparison starts with those two
pairs, continues to seven, then measures 16 and 32 environments. The first two
pairs are retained as light statistics. Counts and workloads come from each
immutable profile; the two-host smoke workflow has its own diagnostic design.

Official results must pass provider compatibility, matched correctness,
system-load, alternating paired-measurement, statistical uncertainty, provenance,
and asset gates. Quick runs and explicit overrides remain diagnostic. No
unmarked promotional video is emitted from invalid or inconclusive evidence.

Every distinct provider configuration and vector shape is preflighted in a
consuming, phase-isolated subprocess against the normative
[Turbo Vector API v2 contract](TURBO_VECTOR_API_V2.md). Attestations bind exact
portable execution-spec fingerprints. Trace, warmup, timing, reset, and replay
use fresh processes and environments that never run mutable contract validation.
Providers that do not declare a conforming contract stop before construction and
dependent workloads.

Published versions **1.0.3 through 2.0.6** ran mutable Turbo API validation inside
workload processes. Their bundles may remain structurally intact, but performance
evidence can be contaminated if validation enables persistent provider state.
Rerun those comparisons with the phase-isolated protocol introduced in 2.0.7
before using them to support performance claims.

## Outputs and privacy

Result bundles contain the exact provider lock, shape-local statistics, report,
chart, raw evidence, verification records, and optional media. `manifest.json`
binds portable files by size and SHA-256. Use `turbobench verify PATH` to check
integrity and consistency, and `turbobench report PATH` to read the report.

Optional legacy `promo` replays the locked providers and one canonical semantic
action trajectory. Displayed ratios come from the bound benchmark; diagnostic
media is watermarked. Two-host showcases additionally bind policy provenance,
both host roles, replay evidence, and each derived asset in versioned proofs.

ROMs and local asset paths never enter portable bundles; only canonical digests
are recorded. TurboBench does not upload or publish bundles. Long-running commands
write progress to standard error and final machine-readable JSON to standard
output.
