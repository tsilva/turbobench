## PROJECT PURPOSE

turbobench gives reinforcement-learning environment authors and users a provider-neutral way to run reproducible matched performance comparisons and generate evidence-bound promotional comparison videos across compatible environment providers.

## PROJECT REQUIREMENTS

### Comparisons

- Compare providers only when they are explicitly compatible with the same logical environment and workload; never rank or compare different games or unmatched workloads.
- The first complete release must support SuperMarioBros-Nes-turbo, breakout-turbo-env, stable-retro-turbo, and ViZDoom-turbo against their compatible upstream or Turbo providers.
- Support latest eligible package releases, exact package versions, and clean local checkouts while resolving every run to isolated, exact, hash-recorded runtime artifacts.
- Use matched correctness checks, alternating paired measurements, statistical uncertainty, system-load gates, and complete provenance before treating a result as valid.
- When a provider is designated as a semantic authority, require unmodified native-transition parity against it; declared lossless representation conversion is permitted, but compatibility normalization used for performance matching must not satisfy or conceal the semantic-fidelity gate.
- Allow a fully gated result from any host to be an official claim, while keeping failed or overridden runs clearly diagnostic and non-promotable.
- Produce portable, self-verifying result bundles without uploading or publishing them in the initial release.

### Promotional media

- Generate comparison videos only from a valid matched benchmark and an exact provider-pair replay of one canonical semantic action trajectory.
- Bind every video and displayed speed ratio to the exact providers, versions, workload, replay evidence, benchmark result, and output hashes used to create it.
- Never emit an unmarked promotional asset from invalid, incomparable, inconclusive, or unverifiable evidence.

### Assets

- Keep ROMs and other user-supplied game payloads out of the repository and distributions; validate required assets by canonical digest without exposing local paths in portable reports.
