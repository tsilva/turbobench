<p align="center">
  <img src="https://raw.githubusercontent.com/tsilva/turbobench/main/logo.png" alt="turbobench logo" width="360" />
  <br />
  <!-- repo-tagline:start -->
  <strong>⚖️ Matched environments, measured fairly ⚖️</strong>
  <!-- repo-tagline:end -->
</p>

<p align="center">
  <a href="https://github.com/tsilva/turbobench/actions/workflows/ci.yml"><img src="https://github.com/tsilva/turbobench/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status on main" /></a>
  <a href="https://pypi.org/project/turbobench-cli/"><img src="https://img.shields.io/pypi/v/turbobench-cli" alt="Published PyPI version" /></a>
  <a href="https://github.com/tsilva/turbobench/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue" alt="Python 3.11 or newer" /></a>
  <a href="https://github.com/tsilva/turbobench/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/turbobench-cli" alt="MIT license" /></a>
</p>

`turbobench` is a Python command-line tool for reinforcement-learning environment
authors, researchers, and provider maintainers. It checks whether compatible
implementations behave the same and measures their performance under matched
workloads. Use it to test a provider during development or compare exact releases.

Results stay local in portable bundles with recorded versions, hashes, and
verification evidence. Optional comparison videos replay the same actions in both
providers; diagnostic videos are visibly watermarked.

## Install

Requires Python 3.11+, `uv`, FFmpeg, and FFprobe. Isolated provider environments
default to CPython 3.14.

Install [turbobench-cli 2.0.11](https://pypi.org/project/turbobench-cli/2.0.11/):

```bash
uv tool install turbobench-cli==2.0.11
```

Alternatively, install it in an active virtual environment with
`python -m pip install turbobench-cli==2.0.11`. The installed command and Python
import remain `turbobench`.

For development, use the checkout:

```bash
git clone https://github.com/tsilva/turbobench.git
cd turbobench
uv sync --frozen --group dev
```

Run `turbobench profiles list` and `turbobench providers list` to choose a
compatible workload and provider pair. In a checkout, prefix commands with
`uv run --frozen`.

## Commands

```bash
turbobench profiles list               # list immutable workloads
turbobench providers list              # list available provider adapters
turbobench doctor vizdoom/basic-v1     # check the host, tools, and assets

# Check behavior without benchmarking; quick runs are diagnostic.
turbobench parity vizdoom/basic-v1 \
  --candidate env-vizdoom-turbo@1.3.0.post27 --quick \
  --output turbobench-results/parity-vizdoom
turbobench verify-parity turbobench-results/parity-vizdoom

# Measure a matched provider pair.
turbobench compare vizdoom/basic-v1 \
  --left env-vizdoom-turbo@1.3.0.post27 \
  --right vizdoom@1.3.0 \
  --output turbobench-results/vizdoom

turbobench verify turbobench-results/vizdoom  # verify integrity and consistency
turbobench report turbobench-results/vizdoom  # print the generated report
turbobench promo turbobench-results/vizdoom --diagnostic

uv run --frozen ruff check .                   # lint a checkout
uv run --frozen pytest -m "not acceptance"     # test without proprietary assets
```

See the [usage guide](https://github.com/tsilva/turbobench/blob/main/docs/usage.md)
for provider selectors, Breakout comparisons, evidence reuse, and measurement
details. The [parity guide](https://github.com/tsilva/turbobench/blob/main/docs/parity.md)
covers dirty-checkout diagnostics and certification of an exact final wheel.

## Two-host showcases

Available in the published CLI. Run from the render machine, replacing the
example SSH host and policy path with your own:

```bash
uv run --frozen turbobench configure \
  --benchmark-host benchmark.example.com \
  --policy /path/to/verified-policy-package

uv run --frozen turbobench compare --showcase --smoke  # diagnostic end-to-end check
uv run --frozen turbobench compare --policy-benchmark --showcase  # full policy timing and video
```

Measurements run on the other machine; replay, video, animated WebP, and the
scaling chart are generated locally. Saved policy settings, declarations from
exact provider packages, and versioned TurboBench comparison rules resolve to
one frozen workload. Its sources, checkpoint, host roles, measurements and
exports are bound into versioned proofs; existing proofs still verify. Full
workflows double the environment count until both providers plateau or slow down. Output goes
to a fresh directory under `turbobench-results/`. Follow the
[workflow guide](https://github.com/tsilva/turbobench/blob/main/docs/comparison-workflow.md)
for policy import, assets on both hosts, extra rendering tools, and proof checks.

## Notes

- Official comparisons require compatible providers, matched correctness, paired
  timing, uncertainty estimates, provenance, asset validation, and acceptable
  system load. Quick runs and overrides remain diagnostic.
- Supply lawful game assets locally through `TURBOBENCH_ROM_PATH`,
  `TURBOBENCH_ASSET_ROOT`, or `RETRO_DATA_PATH`. ROMs and their local paths stay
  out of portable evidence.
- Official v1 hosts are Apple-silicon macOS and x86-64 Linux. Third-party
  adapters register through the `turbobench.providers` entry-point group.
- Long-running commands send progress to standard error and final
  machine-readable JSON to standard output. Bundles are not uploaded or published.
- **Rerun performance claims from versions 1.0.3–2.0.6.** Those releases ran
  mutable API validation inside workload processes, which could contaminate
  timing. Version 2.0.7 isolates validation from measured workloads; see the
  [usage guide](https://github.com/tsilva/turbobench/blob/main/docs/usage.md#measurement-and-validity).

## Architecture

The core matched-comparison pipeline:

![turbobench matched-comparison architecture](https://raw.githubusercontent.com/tsilva/turbobench/main/architecture.png)

## License

[MIT](https://github.com/tsilva/turbobench/blob/main/LICENSE)
