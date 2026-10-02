# Benchmark and showcase in one command

Run this workflow from the **render machine**. It stages the exact TurboBench
source and frozen dependency lock over SSH, measures on the benchmark machine,
verifies the downloaded immutable proof, then replays and renders locally.
The benchmark host does not create charts, videos, or raw-frame artifacts.
Nothing is pushed or published.

## Breakout on two hosts

Replace `benchmark.example.com` in the examples with your configured SSH
benchmark host. Both machines need their own lawful canonical Breakout assets. The
benchmark host needs SSH access, `uv`, and available provider wheels for the
chosen Python minor; the render host also needs FFmpeg/FFprobe and `img2webp`
(`brew install ffmpeg webp` if absent).

Place assets on each host under
`~/.local/share/turbobench/assets/Breakout-Atari2600-v0`. Discovery checks the ROM and
`Start.state` hashes. This folder is private host-local input, never part of a
proof or source snapshot. Place `rom.a26`, `Start.state`,
`data.json`, and `scenario.json` in the same layout, or configure
`TURBOBENCH_ASSET_ROOT` for a local command. The remote coordinator discovers
this standard directory without forwarding private local paths.

Import a verified policy package using the command below, then provide its path
with `--policy`. The package contains the exact checkpoint, saved recipe/model metadata, GradLab capture,
effective raw actions, run link, limitations, schema documents, and hash manifest.
The tested example checkpoint is `dcfd8a41bcc7426ac0ef2f108d07a20f2ffb2c0c22745e6690c6065c01d5935a`,
step 191,561,728 from
[FirstWall PPO](https://mlflow-beast3.tsilva.eu/#/experiments/6/runs/e10b9f9dfec247b881d2eac3979dda37).

```bash
# Run from your TurboBench checkout on the render machine.
uv run --frozen turbobench configure \
  --benchmark-host benchmark.example.com \
  --policy /path/to/verified-policy-package

uv run --frozen turbobench compare --showcase --smoke
```

For the full run, **remove `--smoke`**. A fresh timestamped directory under
`turbobench-results/` is chosen automatically and printed on completion.
Use `--output turbobench-results/breakout-full` when you want a fixed path,
including for resumable runs. Keep the benchmark host idle during
an official measurement; the load gate waits up to 15 minutes. Stop other jobs
through their normal controls before starting. Full measurements use the
protocol's environment counts **1, 16, 32**, one warmup pair, and seven alternating
AB/BA pairs with three repetitions per invocation. Shape-local confidence
intervals use a deterministic paired bootstrap. Exact package references make
the tested provider versions explicit. Use a different eligible candidate
version deliberately when comparing a newer release; its locked policy excerpt
must still pass correctness and replay, without changing the training cadence.

Defaults are stored on the render machine in
`$XDG_CONFIG_HOME/turbobench/config.json` (normally
`~/.config/turbobench/config.json`), outside the repository. Run `turbobench
configure` without flags to inspect them. The selected policy proof ID is pinned:
replacing its contents requires an explicit `configure --policy PATH` selection.
The comparison definition is inferred from the saved policy game. It pins the
upstream authority/version; the candidate/version comes from saved training
metadata. FirstWall resolves to `breakout/policy-v1`, protocol
`paired-environment/v1`, `stable-retro@1.0.1`, and
`env-breakoutatari2600-turbo-native@0.5.13`. The workload ID includes a digest
of the recipe-derived settings, exact provider declarations, and comparison
rules. No policy-specific TOML needs to be maintained.
Rendering defaults to local. No training or sampling parameters are guessed.

Explicit flags take priority. `TURBOBENCH_BENCHMARK_HOST` overrides the configured
host when `--benchmark-host` is absent. Without a configured policy, a single
verified matching package under `turbobench-results/policies/` is discovered
automatically; multiple distinct packages require an explicit selection. To test
a newer candidate release, supply `--right PROVIDER@VERSION` or
`--right PROVIDER@latest`. Ordinary comparisons also default the left provider
from the profile and the right provider to its sole candidate's latest eligible
release; profiles with multiple candidates require `--right`.

Smoke uses **1 and 2 environments, one pair, one repetition per provider,
zero warmups**, retaining the protocol's 256 measurement steps. It checks the
entire transfer/proof/replay/render flow, returns success only when it completes,
and emits visibly diagnostic assets. It has no CI, significance outcome, or
official performance claim. Package quarantine and host-load gates remain
recorded; a release-only smoke may proceed diagnostically through those gates.
Correctness, runtime-contract, exact replay, hash, or rendering failures still
fail the pipeline. Smoke and full requests have different IDs and cannot reuse
one another's evidence.

```bash
uv run --frozen turbobench verify turbobench-results/breakout-full
uv run --frozen turbobench report turbobench-results/breakout-full
```

Substitute the printed output directory if you did not specify `--output`.

The command preserves `.partial` output and the remote content-addressed job on
failure. Rerunning the **same command and harness source** resumes verified
measurements, rather than rerunning timing for a render retry. Changed inputs or
source require a new output. Completed outputs are immutable; they are never
overwritten. Local provider checkout/artifact selectors and workload overrides
are deliberately unsupported in the two-host release workflow. Independent
`compare`, `parity`, `promo`, and legacy bundle verification remain available.

## Import another captured GradLab policy

```bash
uv run --frozen turbobench policy-pack \
  --model /absolute/path/to/gradlab/public-model \
  --capture /absolute/path/to/policy-playback.json \
  --actions /absolute/path/to/comparison-actions.json \
  --mlflow-url 'https://tracking.example/#/experiments/6/runs/RUN_ID' \
  --limitations 'Describe excerpt/attempt selection and known full-episode failures.' \
  --output turbobench-results/policies/my-policy
```

The current importer accepts GradLab model format 3, recipe format 4, recorded
`gradlab.readme-playback/v1`, and `turbobench.imported-policy-actions/v1`. It
imports an existing inference capture; it does not execute the checkpoint.
The importer normalizes the saved policy contract once. A compatible Breakout
recipe can change frameskip, stack, resize, crop, or action ordering without a
new policy-specific profile. The current upstream image adapter supports CHW
uint8 grayscale with area resize, zero-filled remove/mask crop, no maxpool, no
sticky actions, and no fire reset. Unsupported operations fail before timing.
Other games need a versioned comparison definition and compatible provider
adapters; other saved model formats need an importer. Shared measurement,
proof, host-separation, and rendering code remains reusable. Missing or mismatched
training metadata fails closed. Never substitute another frame skip to improve
a displayed speedup.

The importer verifies the checkpoint and saved recipe digests, captured model
identity, full action/wrapper contract, training environment contract hash,
frame skip/stack/crop/resize/grayscale/pooling/sticky actions, reset/state,
and every expanded action against the recorded effective policy decisions.
The complete saved task/context/input recipe remains in the package. Neutral
reset-prefix controls are added after the three trained action IDs in the
FirstWall raw recording table; policy decision IDs and cadence remain unchanged.

The `breakout/policy-v1` comparison definition explicitly declares a bijection from Linux Stable Retro's
nine RGB565 palette IDs to the macOS BGR transport saved by this policy's
training contract. Unknown colors are rejected. Canonical rendered frames,
preprocessed observations, rewards, lifecycle, and selected infos must still
match exactly. Existing canonical parity profiles and their hashes, including
`breakout/start-v1` and `breakout/firstwall-policy-v1`, remain unchanged.

The archived excerpt ends at 429 points with four lives. The previously captured
full episode diverged at raw frame 5914; neither the excerpt nor its showcase
asserts wall completion. New packages must record their own selection and known
limitations rather than inheriting these example values.

## Evidence and exports

```text
comparison/
  manifest.json                # turbobench.showcase-proof/v2
  benchmark/
    manifest.json              # turbobench.benchmark-proof/v2
    request.json               # turbobench.comparison-request/v2
    result.json                # result/v2, or smoke-only result/v3
    resolved-lock.json
    benchmark-machine.json
    profile.toml                # generated pointer to the frozen workload
    resolved-workload.json      # full inputs, settings, source hashes and origins
    raw/                       # contracts, traces, raw timing samples
    verification/              # parity, replay hashes, phase attestations
      provider-declarations/   # actual benchmark-host artifact capabilities
  policy/
    manifest.json              # turbobench.policy-proof/v2
    model.zip                  # exact checkpoint, not loaded by verification
    model.json
    recipe.json
    policy-contract.json        # normalized training contract, including wrappers
    capture.json
    actions.json
    provenance.json
  render-lock.json             # platform-specific rendering artifacts
  verification/showcase-replay.json
  media/
    comparison.mp4             # silent 1672x940 H.264, 60 fps
    comparison.webp            # full-resolution lossless, 20 fps, loop=0
    poster.png
    card.png
  chart.svg                    # candidate/upstream speedup and inverted 95% CI
  report.md                    # hardware, policy link, method, limits, SPS table
  README-snippet.md
  schemas/                     # versioned JSON schema documents (also in children)
```

Proof IDs hash canonical manifests containing every artifact's byte digest and
size. The final showcase binds immutable measurement and policy child IDs,
actual hashed host identities, rendering source/style, runtime locks, replay
commitments, and all exports. Child manifests are themselves inventoried; there
are no circular IDs. `verify` rejects unknown versions, missing/extra files,
unsafe paths or symlinks, model/recipe/action substitution, host reuse, sampling
mismatch, recomputed-statistics mismatch, unbound assets, and replay divergence.
Schemas live in `src/turbobench/schemas` and ship in the package and proofs.
Integrity establishes internal consistency, not authentication of the author.

Linux and macOS wheels have different bytes. Both runtime locks are retained,
with exact release/source identities and per-side cross-host replay equality;
we never claim they are the same binary. No ROM, snapshot payload, or raw
reference-provider frames are transferred with the measurement proof. Raw
rendering frames are temporary on the asset host.

Timing measures matched environment work with seeded controls, including
preprocessing, IPC, infos, terminal detection, and selective resets. It excludes
inference, training task/context/reward wrappers, construction, initial reset,
capture, chart generation, and encoding. The benchmark uses `obs_copy=copy` and
`num_threads=n_envs`, which may differ from training buffer/thread settings; the
saved training recipe and those differences are disclosed. Only the comparison
video follows the captured trained policy actions. Video playback illustrates
shape-1 throughput with common 4x compression and the measured ratio; it is not
a wall-clock recording. Slower candidates retain their real sub-1x ratio.

A full unmarked video requires official, conclusive shape-1 measurements and
exact replay. Failed gates preserve evidence and refuse the video; a diagnostic
chart/report can remain in partial output. Other shapes, including slow or
inconclusive points, remain on the chart. Do not extrapolate unmeasured counts.
Copy only verified exports into the environment README, preserving its adjacent
method/limitations caption and an archive reference to the proof. Models and
result packages stay under ignored `turbobench-results`, outside Git and wheels.

Use the project skill `.codex/skills/comparison-showcase/SKILL.md` and the
[approved presentation reference](comparison-media.md) when refreshing assets.

## Workload ownership and migration

The exact provider package declares supported actions, states and observation
operations. The saved policy chooses requested preprocessing, reset behavior,
ordered controls and input/wrapper contracts. TurboBench owns the upstream
authority, canonical ROM/state digests, required checks, compatibility and
sampling protocol. The resolver
validates these sources and freezes them in `resolved-workload/v1` with a
content-derived ID and field-level source references.

Both providers are probed in fresh excluded processes on the render host before
the request is frozen. The benchmark host independently probes its platform
artifacts against the same settings and canonical assets, then uses separate instances for correctness,
timing and hash-only replay. Its declarations and attestations are archived.
Rendering follows that exact workload and requires cross-host replay equality.

New imports produce `policy-proof/v2`; the workflow produces request, benchmark
and showcase documents at v2. Existing v1 policy packages are verified and
migrated to a content-addressed sibling automatically; originals stay intact.
Explicit legacy profile imports and old proof verification remain supported.
To pin the migrated package in local defaults, select its printed path with
`configure --policy PATH`. No checkpoint bytes are changed or executed by import.

Verification reconstructs the workload from its archived inputs, validates the
recognized TurboBench definition/protocol version, and checks model, actions,
artifact, host and asset bindings. It does not consult the current profile
registry or live provider defaults. Changing a trusted comparison rule requires
a new version; an embedded document cannot weaken checks.
