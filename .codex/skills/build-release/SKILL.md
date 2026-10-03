---
name: build-release
description: Build, audit, publish, monitor, or verify turbobench-cli Python releases. Use when the user invokes /build-release or $build-release; asks to build release artifacts or a local candidate; requests a specific version; asks to cut, tag, publish, or monitor the turbobench-cli PyPI project; or wants to confirm that an exact version is live.
---

# Build Release

Read and apply the shared `$release-workflow` skill at
`/Users/tsilva/.codex/skills/release-workflow/SKILL.md` before execution.
It owns common preflight, publication safeguards, `$push` integration,
workflow monitoring, verification, and reporting. The rules below are this
project's adapter; they retain its invocation default and required gates.
If the shared skill is unavailable, stop and report the missing dependency.

Use the repository-owned release path and preserve the distinction between a
local candidate and external publication. A local candidate is reversible;
pushing a release tag publishes externally when the trusted-publishing workflow
is installed.

Publish to the PyPI project `turbobench-cli`. Keep the installed command and
Python import named `turbobench`; normalized distribution files use the
`turbobench_cli-<version>` prefix.

Treat an unqualified `/build-release` or `$build-release` invocation as a
request to complete the publication flow. Use the local-candidate flow only
when the user explicitly asks for artifacts, a candidate, a dry run,
validation-only work, or no publication.

Use normal project-owned PEP 440 versions. There is no upstream-derived or
mandatory `.postN` scheme. Treat an untagged version absent from PyPI as
pending; otherwise select the next unused patch version. For prerelease and
development versions, increment the existing `aN`, `bN`, `rcN`, or `.devN`
suffix. Accept an exact valid version selected by the user, including an
explicit `.postN`, but never add a post-release suffix automatically.

Keep the version identical in `pyproject.toml`,
`src/turbobench/__init__.py`, and the root `turbobench-cli` entry in `uv.lock`.

## Build or validate in GitHub Actions

Read `AGENTS.md` and apply `$specs-author`. Normal release and validation builds
run only in Actions. A local candidate requires an explicit request for a local
source build; use the existing `build` helper only for that outcome.

For validation without publication, fetch the current branch's configured
upstream on main, resolve its full commit SHA, and run:

```bash
gh workflow run release.yml --ref main -f ref=<full-pushed-main-sha>
```

The runner enforces the frozen lock, matching three source versions, Ruff,
portable tests with FFmpeg, wheel/sdist audits, isolated wheel import and CLI
entry-point metadata. The host/asset-dependent acceptance suite remains outside
the portable release gate. Download `release-v<version>` into a fresh external
directory, then audit its existing artifacts:

```bash
python3 .codex/skills/build-release/scripts/release_build.py audit \
  --version <version> --dist-dir <download-directory>
```

Monitor the exact dispatched SHA and compare downloaded SHA-256 digests with
the runner log. A validation dispatch never tags or publishes. It builds pushed
source; dirty local changes are excluded and must not be described as tested.

## Publish a release

Require all of the following before tagging or publishing:

- a clean worktree on the current branch;
- the branch synchronized with its configured upstream;
- consistent version metadata for the selected version;
- an unused `turbobench-cli` PyPI version and unused
  `turbobench-cli-v<version>` tag;
- metadata checks without a local artifact build; and
- a checked-in `.github/workflows/release.yml` that builds and audits the same
  wheel and sdist, publishes through PyPI Trusted Publishing, and creates a
  GitHub Release only for a pushed release tag.

Require the workflow's PyPI job to use the GitHub `pypi` environment and OIDC
trusted publishing for the `turbobench-cli` project. If the workflow is absent
or no longer matches this contract, stop before tagging or pushing and repair
the repository-owned path. Do not replace it with a local upload.

Start clean, fetch the configured remote and tags, confirm synchronization, run
`prepare-version --write`, run `check-version`, `check-pypi`,
`uv lock --check --config-file uv-tool.toml`, and `git diff --check`. Do not
install a local environment, run source tests, or build a local candidate. If
version preparation changed metadata, commit exactly `pyproject.toml`,
`src/turbobench/__init__.py`, and `uv.lock` as `Release <version>`. Verify that all three committed source versions agree. Actions validates and
builds the exact tagged commit before publication.

Create an annotated tag only after the metadata requirements pass, then atomically
push the current branch and tag:

```bash
git tag -a turbobench-cli-v<version> -m "Release turbobench-cli-v<version>"
git push --atomic <remote> HEAD turbobench-cli-v<version>
```

## Verify publication

Follow the shared monitoring and verification procedure for the `release.yml`
tag-push run at the full `turbobench-cli-v<version>` commit SHA. A `workflow_dispatch` run
validates artifacts but never publishes. Verify PyPI project `turbobench-cli` and
the GitHub Release for the same tag.

Use the existing exact-version verifier:

```bash
python3 .codex/skills/build-release/scripts/release_build.py \
  wait-pypi --version <version>
```

Require `turbobench_cli-<version>-py3-none-any.whl` and
`turbobench_cli-<version>.tar.gz` on PyPI and the GitHub Release for the tag.
