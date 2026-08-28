# Turbo Vector API v2

TurboBench defines Turbo Vector API v2 as the common, breaking contract for
native vector environment providers. Provider repositories may mirror local
assertions, but they do not depend on TurboBench.

## Constructor

The common parameters are ordered exactly as follows. Provider-specific
extensions may follow `state_catalog` only as keyword-only parameters. Catch-all
keyword parameters are forbidden, so Python reports unknown arguments with its
normal `TypeError`.

```python
(
    game,
    state=None,
    scenario=None,
    info=None,
    use_restricted_actions="default",
    record=False,
    players=1,
    inttype="stable",
    obs_type="image",
    render_mode=None,
    *,
    num_envs=1,
    num_threads=None,
    rom_path=None,
    transport="default",
    obs_copy="safe_view",
    obs_resize=(84, 84),
    obs_crop=None,
    obs_crop_mode="remove",
    obs_crop_fill=0,
    obs_grayscale=True,
    obs_resize_algorithm="area",
    obs_layout="chw",
    frame_skip=4,
    frame_stack=4,
    maxpool_last_two=False,
    noop_reset_max=0,
    use_fire_reset=False,
    sticky_action_prob=0.0,
    reward_clip=False,
    info_filter="all",
    info_frame_stack_keys=None,
    state_catalog=None,
)
```

An unsupported common feature accepts its neutral default and raises
`ValueError` for a non-neutral request. `state` and `state_catalog` are mutually
exclusive when `state` is non-`None`. Catalogs are immutable ordered tuples,
non-empty, duplicate-free, and reset to index zero unless the caller supplies
lane-aligned `int32` indices.

## Lifecycle and transport

Providers declare `metadata["turbo_api_version"] == 2` and a resolved
`metadata["transition_transport"]`. Mario, Breakout, Stable Retro Turbo, and
ViZDoom Turbo use NumPy. `env-Doom-turbo-torch` uses Torch tensors on
`env.device` and never offers NumPy transition transport.

Every lane-aligned transition value uses the declared transport: actions,
reset masks, state indices, observations, rewards, termination and truncation
flags, info values, and Gymnasium `_key` masks. Autoreset is disabled. A
terminal lane cannot be stepped before selective reset.

Reset infos always include:

| Key | dtype | Meaning |
| --- | --- | --- |
| `state_index` | `int32` | active immutable-catalog index |
| `start_source` | `int8` | `0` environment, `1` snapshot |
| `noop_reset_count` | `int64` | sampled reset advancement |
| `_state_index`, `_start_source`, `_noop_reset_count` | `bool` | lane-presence masks |

`noop_reset_max=0` disables reset no-ops. Positive `N` samples uniformly from
the inclusive range `1..N`. Object and string transition arrays are forbidden.

`step_async(actions)` and `step_wait()` are the only standardized asynchronous
names. Rendering is enabled only with `render_mode="rgb_array"`; then `render()`
returns lane zero, `render_lane(i)` returns one native RGB NumPy frame, and
`get_images()` returns one NumPy frame per lane.

## Introspection

`signal_schema` is an immutable mapping. Every entry is an immutable mapping
with exactly `dtype` (canonical string), `shape` (tuple),
`available_on_reset` (bool), and `available_on_step` (bool).

`capabilities` is immutable and contains every key exported as
`turbobench.turbo_api.CAPABILITY_KEYS` in order. TurboBench may also recognize
named sequence extensions exported through
`turbobench.turbo_api.SEQUENCE_CAPABILITY_EXTENSIONS`; unknown keys remain
invalid. Sequence values are tuples. Feature values are booleans accurate for
the configured instance and coherent with the methods that instance exposes.

Observation ownership is explicit: `copy` is permanently caller-owned,
`safe_view` survives one subsequent observation-producing call, and
`unsafe_view` may alias that next call.

## Validation and promotion

TurboBench inspects API metadata before construction. Declared v2 providers
must pass constructor and runtime validation before a workload starts. A failed
v2 report is recorded and execution stops. Historical v1 providers may run only
diagnostically and are never promotable. API validity is independent from
semantic parity and is required by official-claim and self-verification gates.

Each validation emits a portable
`turbobench.turbo-contract-report/v1` document whose `report_sha256` is the
canonical JSON hash of the report excluding that hash field.
