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

`turbobench` checks whether reinforcement-learning environment providers
behave the same and measures their performance under matched workloads.
It records exact versions, hashes, and verification evidence in local bundles.

## Quickstart

Requires Python 3.11+, uv, FFmpeg, and FFprobe. Isolated provider environments
default to CPython 3.14.

Install [turbobench-cli 2.0.11](https://pypi.org/project/turbobench-cli/2.0.11/):

```bash
uv tool install turbobench-cli==2.0.11
```

Choose a workload and check the host:

```bash
turbobench profiles list
turbobench providers list
turbobench doctor vizdoom/basic-v1
```

The host check reports missing tools or assets before you run parity or
benchmark measurements. Use the [usage guide](https://github.com/tsilva/turbobench/blob/main/docs/usage.md)
for a matched comparison and how to verify the output bundle. Quick runs are
diagnostic.

- [Certify provider parity](https://github.com/tsilva/turbobench/blob/main/docs/parity.md)
- [Run benchmarks and export charts or videos](https://github.com/tsilva/turbobench/blob/main/docs/comparison-workflow.md)
- [Measurement rules and prior timing correction](https://github.com/tsilva/turbobench/blob/main/docs/usage.md#measurement-and-validity)

Provide any required game assets locally. Results stay on disk until you
choose to share them.

## Architecture

![turbobench matched-comparison architecture](https://raw.githubusercontent.com/tsilva/turbobench/main/architecture.png)

## License

[MIT License](https://github.com/tsilva/turbobench/blob/main/LICENSE).
