## Product Specifications

When specifications exist, use `$specs-author` for substantive work and
specification changes. Follow its shared read/reuse cadence, provisional-scenario
guidance, proportional detail, decision-based approval, and closing intent check.
The maintained procedure is `/Users/tsilva/.codex/skills/private-skills/specs-author/SKILL.md`.

## Release Builds

Use `$turbobench_build-release` whenever asked to build, cut, publish, monitor, or verify a turbobench release. Normal builds run only in GitHub Actions; local preparation handles version metadata and Git operations.

## Comparison Media

Before creating or refreshing comparison media for any environment, read
`docs/comparison-media.md` and follow its user-approved style and constraints.
Use `.codex/skills/turbobench_comparison-showcase/SKILL.md` for benchmark-backed comparison
showcases. Executable workflow documentation is in `docs/comparison-workflow.md`.

## Shared release procedure

The project `turbobench_build-release` skill composes `$release-workflow` from
`/Users/tsilva/.codex/skills/release-workflow/SKILL.md`.
Read both for release work; keep project commands, version policy, artifact
requirements, and approval gates in the project adapter.

## Benchmark infrastructure privacy

Never publish or print benchmark machine hostnames, IP addresses, ports, SSH aliases, login names, private tracking URLs, or other machine access/location details. Report hardware specifications only. Keep historical proofs with access details private; do not export or upload them.
