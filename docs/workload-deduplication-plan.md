# Deduplicate workload definitions

Status: proposed implementation plan. No execution behavior changes in this task.

## Goal and ownership

Keep independent comparison rules in TurboBench while eliminating manually
copied environment and policy settings. Preserve the short `compare --showcase`
command, the approved presentation, two-host execution, and portable proofs.

| Definition | Authoritative owner |
| --- | --- |
| Supported actions, states, observation operations, signals, lifecycle capabilities | Exact environment package |
| Requested preprocessing, action ordering, reset behavior, policy inputs and wrappers | Saved training recipe and capture |
| Upstream authority, compatibility rules, required checks, budgets, sampling, allowed representation conversions | Versioned TurboBench protocol and environment comparison definition |
| Exact resolved workload and all contributing identities | Immutable TurboBench proof |

Provider capabilities describe what is supported; they do not choose what to
benchmark or establish semantic truth. The original upstream remains the
authority. TurboBench keeps independent compatibility/check requirements and
reference asset commitments; candidate declarations cannot weaken them.

## Current duplication and seams

- `workload_profiles/breakout--firstwall-policy-v1.toml` combines environment
  facts, copied training preprocessing/actions, parity rules, timing budgets,
  and presentation settings. The recipe/capture already retain training values.
- `proofs.policy_contract` checks those copies against the recipe rather than
  producing one reusable normalized policy contract.
- Native Breakout already exposes Turbo API capabilities and packaged
  `metadata.json` action/state data. Reuse these interfaces before adding one.
- `Profile` is flattened across execution, rendering, and proof code.
  `profile_hash`, `profile_toml`, and bundle verification currently look up a
  built-in profile by ID; they cannot verify a newly resolved workload offline.
- Provider entry points identify adapters; they do not currently define an
  environment-declaration interchange format.

## Implementation sequence

### 1. Freeze existing behavior and define the data contracts

Add regression fixtures for the current FirstWall resolved settings, action IDs,
reset prefix, effective controls, sampling, frame hashes, and proof verification.
Keep existing profile TOMLs and their hashes unchanged.

Define three strict, versioned data documents: an environment declaration, a
normalized policy contract, and a resolved workload. A resolved workload records
its protocol ID/hash, comparison-definition ID/hash, policy contract hash,
declaration identities, effective settings, and field-level sources.

Environment declarations contain supported capabilities and named semantic
controls; policy contracts contain requested values and ordered policy actions.
The policy contract retains context/task/reward wrapper identities even when
the timing protocol deliberately excludes their execution.

### 2. Normalize authoritative inputs once

Extract GradLab recipe/capture parsing and cadence validation from
`proofs.policy_contract` into one normalization module. Use that same result for
policy packaging, workload resolution, runtime requests, and renderer labels.
Do not load weights during normalization or verification.

Read declarations from exact isolated provider artifacts in fresh preflight
processes. Reuse Turbo API capabilities, action metadata, and state catalogs.
Add a small declarative export in an environment repo only for genuinely missing
facts, with tests there. Upstream adapters supply equivalent normalized data for
providers without Turbo API support. Archive observed runtime attestations too.
Snapshots in evidence are generated records, not a second maintained source.

### 3. Add one deterministic workload resolver

Introduce a module with one main interface:
`resolve_workload(comparison_definition, protocol, policy_contract, declarations)`.
It returns an immutable resolved workload and reports incompatible inputs before
measurement. It performs no timing or rendering.

Resolution is not a general dictionary merge. The policy owns requested
preprocessing, action ordering, state/reset behavior, and decision cadence.
TurboBench owns check requirements, compatibility, authority and sampling.
Declarations must prove that both sides support the selected configuration.
Missing fields, conflicting contracts, unsupported operations, and unknown
representation conversions fail; do not silently fall back to provider defaults.

Explicitly record benchmark differences such as threads, observation ownership,
and excluded wrappers. Keep reset-prefix raw controls separate from trained
action IDs. Expand replay decisions by the training frame skip without changing
the displayed or measured cadence. Use the same resolved workload for all
correctness and timing phases; require its exact identity for parity reuse.

### 4. Thread the resolved workload through execution and proofs

Update `engine`, `runner`, `workflow`, `showcase`, and `bundle` to consume the
resolved workload through existing execution seams. Keep any temporary legacy
`Profile` projection internal to the resolver; remove duplicate downstream
derivation instead of maintaining two independently editable representations.

Resolve provider identities before freezing the workload. Bind platform-specific
declarations and runtime attestations to their exact artifacts. Transfer the
locked request to the benchmark host; it validates that its artifacts support
the same workload rather than selecting settings again. The render host follows
the same locked workload and verifies cross-host replay commitments.

Introduce new schema versions only for changed proof/request documents. Archive
the normalized inputs, full resolved workload, schema documents, and central
protocol/definition identities. Verification reconstructs the workload and
checks all bindings without consulting current provider defaults or the current
profile registry. Recognized TurboBench protocol versions remain trusted check
definitions; arbitrary embedded documents cannot remove required gates.
Workload hashes participate in caches, receipt compatibility, and resume IDs.

### 5. Migrate FirstWall, then demonstrate reuse

Add a new Breakout comparison definition referencing a shared TurboBench
sampling protocol and the existing independent semantic requirements. Derive
FirstWall preprocessing/actions from its packaged policy instead of writing a
new policy-specific TOML for every recipe. Leave old profile IDs working.

Migrate CLI inference and the new proof path together: inferred defaults must
not recreate removed profile copies. Preserve explicit overrides and show the
selected checkpoint, protocol, workload, providers, and output before execution.
Validate the design with a second compatible Breakout policy using different
preprocessing, plus a non-Atari fixture, so Atari assumptions cannot leak into
generic resolution. Other live environment integrations can migrate incrementally.

## Acceptance and rollout

- Current FirstWall resolves to frameskip=2, stack=4, mask_top=17, area 84x84
  grayscale, no maxpool, and the original ordered trained controls.
- A compatible changed recipe produces a new workload hash without editing
  copied preprocessing in TurboBench. Unsupported preprocessing fails clearly.
- Provider declarations cannot remove required checks or supply a different
  timing workload per side. Conflicting semantic mappings fail before timing.
- Mutated recipes, declarations, protocols, actions, workload snapshots, or
  runtime bindings are rejected, including after recomputing file inventories.
- Old proofs still verify; new proofs verify offline after provider defaults
  change. Receipts and caches cannot cross different resolved workload hashes.
- Smoke remains n_envs=1,2 once per provider, with zero warmups and no official
  claim. Full sampling, two-host separation, and media style remain unchanged.
- Run the existing suite and a real smoke on `beast3-ssh.tsilva.eu`, rendering
  locally, before switching the default showcase path. Public docs use a generic
  SSH host. An official full performance run remains a separate validation step.

## Specification alignment before implementation

Canonical official parity profiles stay immutable and stored in TurboBench.
Policy-derived comparison workloads need an explicit distinction from those
canonical profiles; do not silently reinterpret the existing requirement that
canonical parity profiles pin their exact workload.

Propose this exact addition under Comparisons in root `SPECS.md` for approval:

> Resolve policy-backed comparison workloads in turbobench from exact provider
> declarations, saved policy contracts, and turbobench-owned comparison rules;
> preserve one authoritative source for each setting and freeze the complete
> resolved workload and its source identities into portable evidence.

Do not edit `SPECS.md` until this exact addition is explicitly approved. New
schema and migration details belong in scoped design documentation, not root
stakeholder requirements. No existing published profile or schema is rewritten.
