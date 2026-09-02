## PROJECT PURPOSE

turbobench gives reinforcement-learning environment authors and users a provider-neutral way to check cross-provider semantic parity during development, certify exact release artifacts, run reproducible matched performance comparisons, and generate evidence-bound promotional comparison videos across compatible environment providers.

## PROJECT REQUIREMENTS

### Parity

- Centralize cross-provider semantic parity in turbobench; provider repositories may retain internal consistency tests and diagnostic tools but must not duplicate cross-provider comparison logic.
- Expose semantic-fidelity workflows as `parity` and `verify-parity`, using explicit candidate and authority roles without retaining oracle-named commands.
- Keep official parity profiles immutable, versioned, declarative, and stored in turbobench; each profile must pin its exact authority, workload, quick workload, and required checks, and every change to those commitments must create a new profile version that environments adopt explicitly.
- Use the original upstream provider as the semantic authority whenever direct comparison is possible; a Turbo authority is permitted only when direct comparison is impossible and the reason is declared by the profile.
- Allow profiles to select only standard turbobench check types rather than execute arbitrary callbacks.
- Require every parity profile to check exact action identity, observations, native frames, rewards, termination, truncation, resets, and selected information, plus every profile-specific exact or distributional check; missing required evidence must fail, and only declared lossless representation conversion may precede comparison.
- Drive compatible Turbo candidates through the shared Turbo Vector interface and confine provider-specific adapters to upstream systems that cannot implement it.
- Support isolated diagnostic parity for current repository work, including dirty tracked files and non-ignored untracked source files, by snapshotting and hashing the tested source before building it outside the working directory.
- Make each fixed quick workload exercise every required state and action with fewer steps and lane shapes; quick runs, checkouts, dirty sources, authority overrides, shortened workloads, and other overrides must remain diagnostic.
- Cache authority artifacts and repository builds only under their exact source, dependency, profile, Python, platform, and tool identities so unchanged development checks can run without network access.
- Accept release parity evidence only for the exact final distribution artifact on one canonical host, with the provider repository remaining responsible for cross-platform consistency.
- Produce one fail-closed, self-verifying parity receipt per provider pair and profile that binds the exact candidate artifact, authority artifact, profile, tool, workload, results, and focused mismatch evidence without including private assets or local asset paths.
- Allow a parity receipt to satisfy a benchmark parity gate only when it binds the exact same provider artifacts and a compatible workload.

### Comparisons

- Use one workload definition across parity, light comparison, and full comparison while keeping parity as a separate command; make full comparison include light-comparison guarantees, and reuse compatible evidence without requiring comparisons to run canonical parity.
- Compare providers only when they are explicitly compatible with the same logical environment and workload; never rank or compare different games or unmatched workloads.
- The first complete release must support `env-supermariobrosnes-turbo-emu`, `env-breakoutatari2600-turbo-native`, `env-stableretro-turbo`, and `env-vizdoom-turbo` against their compatible upstream or Turbo providers.
- Support latest eligible package releases, exact package versions, exact local distribution artifacts, and clean local checkouts while resolving every run to isolated, exact, hash-recorded runtime artifacts.
- Use matched correctness checks, alternating paired measurements, statistical uncertainty, system-load gates, and complete provenance before treating a result as valid.
- Allow a fully gated result from any host to be an official claim, while keeping failed or overridden runs clearly diagnostic and non-promotable.
- Produce portable, self-verifying result bundles without uploading or publishing them in the initial release.

### Promotional media

- Generate comparison videos only from a valid matched benchmark and an exact provider-pair replay of one canonical semantic action trajectory.
- Bind every video and displayed speed ratio to the exact providers, versions, workload, replay evidence, benchmark result, and output hashes used to create it.
- Never emit an unmarked promotional asset from invalid, incomparable, inconclusive, or unverifiable evidence.

### Assets

- Keep ROMs and other user-supplied game payloads out of the repository and distributions; require provider workflows to supply lawful assets and validate them by canonical digest without exposing local paths in portable evidence.
