---
name: comparison-showcase
description: Run a two-host TurboBench comparison with a locked GradLab policy, verified proof data, a README animation, and a scaling chart. Use when creating or refreshing comparison showcases for any environment.
---

Read `docs/comparison-media.md` for the approved presentation and constraints.
Read `docs/comparison-workflow.md` for executable commands and proof contracts.

Use an immutable profile matching the policy's saved preprocessing and action
contract. Never guess missing training metadata or change playback frame skip.
Import the checkpoint, recipe, capture, and effective actions with `policy-pack`.
Keep benchmark measurements on a different physical host from rendering.
Run `compare --showcase` with the locked policy package and benchmark host.
Use `--smoke` for the complete diagnostic pipeline at n_envs=1 and 2 with one
measurement per provider, no warmups, no CI, and no official claim.
Verify the final package before copying its WebP and chart into an environment
README. Preserve the proof package and model outside Git. Do not upload or
publish unless the user requests it. Report any failed gate or partial excerpt.
