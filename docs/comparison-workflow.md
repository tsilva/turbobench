# Benchmark and showcase in one command

Run this workflow from the **render machine**. Installed releases carry the
frozen workflow dependency lock, install the same published TurboBench version
on the benchmark host, and verify matching harness source hashes. Development
checkouts instead stage their exact source and frozen lock over SSH. It measures on the benchmark machine,
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
FirstWall PPO.

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
protocol's adaptive environment counts **1, 2, 4, 8, ...**, one warmup pair, and
seven alternating AB/BA pairs with three repetitions per invocation at each count.
Counts continue until both providers plateau or downgrade; see the stopping rule below. Shape-local confidence
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
`paired-environment/v2`, `stable-retro@1.0.1`, and
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

## Adaptive environment counts

New full policy benchmarks and showcases use versioned v2 comparison protocols.
Start at `n_envs=1`, then double: `2,4,8,16,32,64,...`. Before timing a new count,
both providers must pass isolated contract validation and matched correctness.
Each measured count keeps the full seven-pair, three-repetition design and its
paired speedup confidence interval. Policy inference remains outside timing.

The stopping rule compares **each provider's median SPS with its own best
previous median**, rather than comparing speedup ratios. A provider plateaus
when two successive counts gain less than 3%; a decline of at least 5% qualifies
as a downgrade immediately. A subsequent gain of at least 3% clears its low-gain
streak. Continue while either provider has not yet qualified. Retain every measured
count, including the confirming plateau or slower count, in the grouped bars.
The rule is a declared throughput heuristic; it does not claim a statistical
confidence interval for the location of a throughput peak.

Worker deadlines have a 15-minute floor and scale with the requested lane,
frame-skip, capture-length, repetition, warmup, and snapshot-replay work. A
larger healthy full-policy invocation can therefore finish without shortening
the capture or changing the sample design. Workers that exceed their finite
budget are still terminated and leave partial evidence.

The locked safety cap is 1024 environments. Reaching it without both providers
qualifying is `safety_cap`, not evidence of saturation: the measurement remains
diagnostic and cannot produce an official showcase. The coordinator preserves
that benchmark proof under `.partial` for inspection. Resource or parity failures
also preserve partial evidence without claiming a completed search.

`result/v4` and `verification/scaling.json` record the rule, medians, gains,
low-gain streaks, provider statuses and stopping reason. Verification recomputes
the entire decision history from bound raw measurements, rejects missing/skipped
counts, premature stops and evidence past the required stop. Resuming the same
partial request reuses its completed shape measurements and makes the same next
count decision. Smoke still measures only `n_envs=1,2` and tests the pipeline;
it does not search for saturation. Explicit historical profiles keep their
versioned fixed counts, and existing completed bundles are unchanged.

## Benchmark the locked policy without videos

Select a verified policy package and pin its proof ID in local defaults:

```bash
uv run --frozen turbobench configure --benchmark-host benchmark.example.com --policy /path/to/verified-policy-package
uv run --frozen turbobench compare --policy-benchmark --right env-breakoutatari2600-turbo-native@latest
```

An explicit `compare --policy PATH --benchmark-host HOST` also selects this mode.
Use `--smoke` for the diagnostic pipeline check; omit it for seven alternating
pairs with three repetitions at each adaptive `n_envs` count. The complete locked capture is
used for both correctness and timing. It is never silently shortened. The model
is not executed during measurement: the imported effective policy actions are
collapsed from raw frames to decisions at the saved training frame skip.

`paired-policy/v2` freezes the checkpoint, recipe, action digest, captured reset
seed and no-op prefix, decision count, and provider declarations in
`resolved-workload/v2` and `comparison-request/v3`. Every lane starts from the
same captured seed and follows the same effective actions. Each timing
repetition resets to that seed; the initial seeded no-op reset, warmup, inference,
correctness hashing, trajectory recording, rendering and encoding are excluded.
Stepping, observation preprocessing, IPC, infos, terminal detection and required
selective resets are included. Training frame skip, stack and preprocessing are
preserved; task/context/reward wrappers are excluded, buffers use `obs_copy=copy`,
and threads equal the environment count. Correlated lanes are an explicit
workload choice, not an estimate of independent policy performance.

Both providers must pass the entire decision-level trace at every measured
shape, then hash-only raw capture replay on the benchmark host. A partial
import remains partial; neither a matching excerpt nor one successful episode
establishes full-game parity or success rate. Standard canonical parity profiles
remain unchanged. Future seeded-action showcases also use adaptive scaling;
archived v1 comparison protocols retain their fixed `1,16,32` counts and verification.

The output is a portable `policy-benchmark-proof/v1` containing the selected
`policy/`, a `benchmark-proof/v3` under `benchmark/`, a grouped bar chart and a
method report. No video or image-frame recording is generated. Existing trace
records retain observation/frame digests, rewards, selected infos and lifecycle
evidence for auditing; they are not pixel trajectories for future encoding.
Future video generation requires a separate untimed replay of the bound actions.
Verify with `turbobench verify OUTPUT`. Replacing a configured package is rejected
until explicitly selected again with `configure --policy PATH`. A new model
still needs an untimed inference capture imported through `policy-pack` below.

Add `--showcase` to `--policy-benchmark` to render the video and animated WebP
from the same policy-timed measurements. Rendering follows an untimed, exact
cross-host replay and produces `showcase-proof/v3` with the `benchmark-proof/v3`
child. All inference and rendering remain excluded from timing. The README
snippet identifies captured policy controls rather than seeded random controls.

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
    result.json                # adaptive full result/v4, or smoke-only result/v3
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
  chart.svg                    # paired provider SPS bars per n_envs, speedup and 95% CI
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

Timing measures matched environment work with the request’s locked controls, including
preprocessing, IPC, infos, terminal detection, and selective resets. It excludes
inference, training task/context/reward wrappers, construction, initial reset,
capture, chart generation, and encoding. The benchmark uses `obs_copy=copy` and
`num_threads=n_envs`, which may differ from training buffer/thread settings; the
saved training recipe and those differences are disclosed. Policy-benchmark
requests use captured policy controls for timing and playback; legacy showcase
requests time seeded controls and use policy actions only for playback. Video playback illustrates
shape-1 throughput with common 4x compression and the measured ratio; it is not
a wall-clock recording. Slower candidates retain their real sub-1x ratio.

A full unmarked video requires official, conclusive shape-1 measurements and
exact replay. Failed gates preserve evidence and refuse the video; a diagnostic
chart/report can remain in partial output. Other shapes, including slow or
inconclusive points, remain on the chart. Do not extrapolate unmeasured counts.
Every successful `compare` also creates a sibling `<proof-name>-publication/`
directory. Copy its exports into the environment repository; keep models and
proof archives under ignored `turbobench-results`, outside Git and wheels.
The immutable proof and its original chart/report remain unchanged.

```text
comparison-publication/
  demo.mp4, demo.webp          # verified showcase only, approved Same Actions frame
  benchmark-readme.svg/json   # compact vertical bars through candidate peak
  benchmark.svg/json          # complete counts, simplified publication labels
  benchmarks.md               # latest results, method, limits, proof, pinned verifier
  README-snippet.md           # assets and one link to benchmarks.md
  publication.json            # proof identity, orientation, exporter and asset hashes
```

The README contains only the animation, readable chart, and a link to
`benchmarks.md`. That document describes the latest benchmark; previous runs
retain proof references rather than old results or explanatory prose. All
hardware, provider versions, settings, controls, uncertainty and limits come
from verified evidence. Diagnostic runs remain marked. The profile authority
appears on the left even when supplied as `--right`; otherwise the caller's
left provider is the reference and right provider is the candidate. The
publication record binds that orientation without modifying the proof.

Re-export an existing verified proof with its public archive/release link:

```bash
turbobench export-publication comparison comparison-publication \
  --proof-url https://github.com/OWNER/ENV/releases/tag/PROOF_TAG
# For a subsequent benchmark, preserve only earlier proof links:
turbobench export-publication next-comparison next-publication \
  --proof-url https://github.com/OWNER/ENV/releases/tag/NEXT_PROOF_TAG \
  --previous-publication comparison-publication
```

Output directories must be fresh and outside the proof. A failed export leaves
the completed proof available for this command; it never requires rerunning
measurements. The default local proof link should be replaced with a public
link before copying exports to GitHub. The exporter pins its exact clean source
commit or installed release in verification instructions. A dirty source export
explicitly requires a clean source or published verifier pin before publication.
Nothing is uploaded, pushed, or published automatically.

The README chart retains the measured prefix through the first candidate
median-SPS maximum, including any earlier dips and recoveries. Omitted counts
are disclosed in `benchmarks.md`; the complete chart and table retain them.
Standalone chart export remains available with
`python -m turbobench.readme_chart BENCHMARK_PROOF OUTPUT.svg [--full]`.
See the [README presentation constraints](comparison-media.md#readme-view).

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
