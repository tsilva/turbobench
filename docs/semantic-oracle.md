# Exact semantic oracle

TurboBench v2 profiles compare environment behavior against a pinned original
Stable Retro authority. This is a fidelity test, not a speed benchmark, and it
does not apply compatibility shims or tolerance-based comparisons.

The canonical profiles are:

- `supermario/canonical-v2`: original `stable-retro==1.0.1`, all four World 1
  start states, shapes 1 and 4, and 4,096 seeded transitions per shape.
- `breakout/start-v2`: original `stable-retro==1.0.1`, the `Start` state,
  shapes 1 and 4, and 4,096 seeded transitions per shape.

Each transition compares processed observations, lossless native-frame pixel
codes, rewards, termination and truncation flags, selected info values, lane
resets, and snapshot continuation. Mario additionally compares the canonical
2 KiB CPU RAM address space. Stable Retro's RGB888 expansion of NES RGB565 is
decoded to the underlying native pixel code; no lossy frame conversion is
allowed. Breakout compares the public RGB bytes directly.

Run the full candidate matrix from clean checkouts while developing:

```bash
turbobench oracle supermario/canonical-v2 \
  --left stable-retro@1.0.1 \
  --right supermariobrosnes-turbo@checkout:/path/to/SuperMarioBros-Nes-turbo \
  --output /external/evidence/mario-smb

turbobench oracle supermario/canonical-v2 \
  --left stable-retro@1.0.1 \
  --right stable-retro-turbo@checkout:/path/to/stable-retro-turbo \
  --output /external/evidence/mario-stable-retro-turbo

turbobench oracle breakout/start-v2 \
  --left stable-retro@1.0.1 \
  --right breakout-turbo-env@checkout:/path/to/breakout-turbo-env \
  --output /external/evidence/breakout-native

turbobench oracle breakout/start-v2 \
  --left stable-retro@1.0.1 \
  --right stable-retro-turbo@checkout:/path/to/stable-retro-turbo \
  --output /external/evidence/breakout-stable-retro-turbo
```

Keep receipts outside source repositories because they contain machine and
checkout provenance. An ordinary integrity check accepts checkout or diagnostic
workload evidence. After publishing, regenerate the release receipt with exact
PyPI candidate versions; the canonical gate requires the full workload, the
pinned PyPI authority, and a PyPI candidate artifact:

```bash
turbobench verify-oracle /external/evidence/receipt
turbobench verify-oracle /external/evidence/receipt \
  --require-canonical \
  --require-provider supermariobrosnes-turbo
```

For example, replace a checkout reference with
`supermariobrosnes-turbo@VERSION`, `stable-retro-turbo@VERSION`, or
`breakout-turbo-env@VERSION` for the published-release run. `--allow-dirty`,
`--steps`, `--shapes`, and every checkout selector are useful for development,
but their receipts cannot satisfy the canonical release gate.
