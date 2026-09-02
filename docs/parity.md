# Cross-provider parity

TurboBench parity profiles compare environment behavior against a pinned original
authority. This is a fidelity test, not a speed benchmark, and it
does not apply compatibility shims or tolerance-based comparisons.

The canonical profiles are:

- `supermario/world1-v2`: original `stable-retro==1.0.1`, all four World 1
  start states, shapes 1 and 4, and 4,096 seeded transitions per shape.
- `breakout/start-v2`: original `stable-retro==1.0.1`, the `Start` state,
  shapes 1 and 4, and 4,096 seeded transitions per shape.
- `vizdoom/basic-v2`: original `vizdoom==1.3.0`, shapes 1 and 4, and 4,096
  seeded transitions per shape.

Each transition compares processed observations, lossless native-frame pixel
codes, rewards, termination and truncation flags, selected info values, lane
resets, and snapshot continuation. Mario additionally compares the canonical
2 KiB CPU RAM address space. Stable Retro's RGB888 expansion of NES RGB565 is
decoded to the underlying native pixel code; no lossy frame conversion is
allowed. Breakout decodes Stable Retro's public BGR/RGB565 transport to the
same canonical Stella pixel codes used by the native provider.

Run a quick check against the current working tree while developing. Tracked
changes and nonignored untracked source are copied into an isolated snapshot;
the live checkout is never installed directly.

```bash
turbobench parity supermario/world1-v2 \
  --candidate env-supermariobrosnes-turbo-emu@checkout:/path/to/env-SuperMarioBrosNes-turbo-emu \
  --allow-dirty --quick
```

Keep receipts outside source repositories because they contain machine and
checkout provenance. An ordinary integrity check accepts checkout or diagnostic
workload evidence. For a release, build the exact final wheel first. The
canonical gate requires the full workload, pinned authority, and that wheel:

```bash
turbobench parity breakout/start-v2 \
  --candidate env-breakoutatari2600-turbo-native@artifact:/absolute/path/to/candidate.whl \
  --output /external/evidence/receipt
turbobench verify-parity /external/evidence/receipt
turbobench verify-parity /external/evidence/receipt \
  --require-canonical \
  --require-provider env-breakoutatari2600-turbo-native
```

`--quick`, `--allow-dirty`, `--steps`, `--shapes`, `--seed`, authority
overrides, version selectors, and checkout selectors are diagnostic. Their
receipts cannot satisfy the canonical release gate.

Every receipt also contains an exact deterministic correctness trace for the
same provider artifacts. A later `compare` run may use
`--parity-receipt /external/evidence/receipt` to reuse that trace. TurboBench
accepts it only when the provider artifact digests, environment semantics,
lane shapes, and required action-stream prefix match the benchmark exactly.
