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

## Build a local candidate

1. Read `AGENTS.md` and use `$specs-author` as required there.

2. Inspect the worktree and current metadata without mutating either:

```bash
git status --short --branch
python3 .codex/skills/build-release/scripts/release_build.py check-version
```

Dirty files do not prevent an explicitly requested local candidate, but report
that it is not eligible for publication and preserve every existing change.

3. Select the release version:

```bash
python3 .codex/skills/build-release/scripts/release_build.py prepare-version
```

On a clean worktree, add `--write` to apply an automatic bump when the checked-in
version is already tagged or published. For an exact user-requested version,
add `--to <version>`. The helper checks local `turbobench-cli-v*` tags and the
`turbobench-cli` PyPI project and
transactionally updates all three version locations. If the worktree is dirty,
do not add `--write`; proceed only when the reported pending version requires no
bump. Never layer an automatic version edit onto existing user changes.

4. Install and run the locked source gates:

```bash
command -v ffmpeg
uv sync --frozen --group dev
uv run --frozen ruff check .
uv run --frozen pytest -m "not acceptance"
```

Require FFmpeg because the portable media end-to-end test exercises MP4 and GIF
generation. Do not require provider assets for a release candidate. The
`acceptance` suite is host- and asset-dependent and remains outside the portable
release gate.

5. Confirm the selected version remains unused on PyPI:

```bash
python3 .codex/skills/build-release/scripts/release_build.py \
  check-pypi --version <version>
```

Skip only this availability check when diagnosing artifacts from an already
published version, and state why. Never overwrite or republish an existing PyPI
version.

6. Build into a fresh version-scoped directory:

```bash
python3 .codex/skills/build-release/scripts/release_build.py build \
  --version <version> --out-dir dist/release-v<version>
```

The helper uses `uv build --no-sources`, requires exactly one universal wheel
and one source distribution, audits metadata and repository-owned package
contents, imports the wheel from an isolated working directory, validates the
console entry point metadata, and prints SHA-256 digests. It refuses to reuse an
output directory so stale artifacts cannot enter the candidate.

7. Report both artifact paths, their SHA-256 digests, the selected version,
whether metadata was bumped, and every completed gate. Preserve failed
artifacts and exact error output for diagnosis. An uncommitted automatic bump is
not eligible for publication.

## Publish a release

Require all of the following before tagging or publishing:

- a clean worktree on the current branch;
- the branch synchronized with its configured upstream;
- consistent version metadata for the selected version;
- an unused `turbobench-cli` PyPI version and unused
  `turbobench-cli-v<version>` tag;
- a passing local candidate from the exact commit; and
- a checked-in `.github/workflows/release.yml` that builds and audits the same
  wheel and sdist, publishes through PyPI Trusted Publishing, and creates a
  GitHub Release only for a pushed release tag.

Require the workflow's PyPI job to use the GitHub `pypi` environment and OIDC
trusted publishing for the `turbobench-cli` project. If the workflow is absent
or no longer matches this contract, stop before tagging or pushing and repair
the repository-owned path. Do not replace it with a local upload.

Start clean, fetch the configured remote and tags, confirm synchronization, run
`prepare-version --write`, and complete all source and candidate gates. If
version preparation changed metadata, commit exactly `pyproject.toml`,
`src/turbobench/__init__.py`, and `uv.lock` as `Release <version>`. Verify that
the committed tree is identical to the source used for the passing candidate.

Create an annotated tag only after every requirement passes, then atomically
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
