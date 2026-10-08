from dataclasses import replace
from xml.etree import ElementTree

import pytest

from turbobench.bundle import verify_bundle
from turbobench.cli import main
from turbobench.profiles import get_profile
from turbobench.publication import export_publication, render_benchmarks
from turbobench.readme_chart import PUBLICATION_NAMES
from turbobench.stats import paired_statistics
from turbobench.util import read_json, sha256_file


def inventory(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}


def test_export_verified_bundle_is_portable_and_keeps_proof_unchanged(fake_bundle, capsys):
    before = inventory(fake_bundle)
    output = export_publication(fake_bundle, proof_url="https://example.com/proof")
    publication = read_json(output / "publication.json")
    assert inventory(fake_bundle) == before
    assert verify_bundle(fake_bundle)["passed"]
    assert publication["proof_id"] == read_json(fake_bundle / "manifest.json")["bundle_id"]
    assert not publication["media"]
    assert "demo.webp" not in (output / "README-snippet.md").read_text()
    assert (output / "README-snippet.md").read_text().count("](benchmarks.md)") == 1
    assert "DIAGNOSTIC" in (output / "benchmarks.md").read_text()
    assert "DIAGNOSTIC" in (output / "benchmark-readme.svg").read_text()
    assert not any(p.suffix in {".zip", ".a26", ".rom"} for p in output.iterdir())
    for name, record in publication["files"].items():
        assert sha256_file(output / name) == record["sha256"]
        assert (output / name).stat().st_size == record["size"]
    assert main(["export-publication", str(fake_bundle), str(output.with_name("cli-export"))]) == 0
    assert '"publication"' in capsys.readouterr().out


def test_tampered_proof_fails_before_publication_directory_exists(fake_bundle):
    (fake_bundle / "report.md").write_text("tampered")
    output = fake_bundle.with_name("export")
    with pytest.raises(ValueError, match="verification failed"):
        export_publication(fake_bundle, output)
    assert not output.exists()
    with pytest.raises(ValueError, match="outside the immutable"):
        export_publication(fake_bundle, fake_bundle / "export")
    with pytest.raises(ValueError, match="outside the immutable"):
        export_publication(fake_bundle, fake_bundle.parent)


def test_authority_stays_on_left_when_cli_provider_order_is_reversed(fake_bundle, monkeypatch):
    result = read_json(fake_bundle / "result.json")
    profile = replace(
        get_profile(result["profile"]["id"]), authority=result["comparison"]["right"]["provider"]
    )
    monkeypatch.setattr("turbobench.publication.bundle_profile", lambda *_: profile)
    output = export_publication(fake_bundle)
    publication = read_json(output / "publication.json")
    assert publication["source_candidate_side"] == "left"
    svg = ElementTree.fromstring((output / "benchmark-readme.svg").read_text())
    left_bar = next(el for el in svg.iter() if el.attrib.get("data-provider") == "left")
    assert (
        float(left_bar.attrib["data-sps"])
        == result["comparison"]["shapes"]["1"]["statistics"]["median_right_sps"]
    )
    assert "0.50\u00d7" in (output / "benchmarks.md").read_text()
    assert read_json(output / "benchmark-readme.json")["source_candidate_side"] == "left"


@pytest.mark.parametrize(
    "profile_id,reference,candidate",
    [
        ("supermario/world1-v1", "stable-retro", "env-supermariobrosnes-turbo-emu"),
        ("vizdoom/basic-v1", "vizdoom", "env-vizdoom-turbo"),
    ],
)
def test_report_derives_environment_settings_and_keeps_later_counts(
    profile_id, reference, candidate
):
    profile = get_profile(profile_id)
    result = {
        "comparison": {
            "left": {"provider": reference, "version": "1.2"},
            "right": {"provider": candidate, "version": "3.4"},
            "outcome": "right_faster",
            "shapes": {
                str(2**i): {
                    "statistics": paired_statistics(
                        [{"left_sps": [100] * 3, "right_sps": [speed] * 3}] * 7
                    )
                }
                for i, speed in enumerate([200, 150, 400, 300])
            },
        },
        "tool": {"version": "9.9"},
        "sampling": {"mode": "full", "pairs": 7, "repetitions": 3, "warmup_pairs": 1},
        "actions": {"1": {"version": "captured-policy/v1", "measurement_steps": 987}},
        "system": {
            "host": {"cpu": "A different processor", "os": "Linux", "architecture": "aarch64"}
        },
    }
    metadata = {
        "diagnostic": False,
        "verifier": {"source_revision": "abcdef", "installed_release": False},
        "media": {},
        "proof_url": "https://example.com/proof",
    }
    report = render_benchmarks(result, profile, metadata, None)
    assert PUBLICATION_NAMES.get(candidate, candidate) in report and profile.game in report
    assert "A different processor" in report and "7600X" not in report
    assert f"frame skip {profile.frame_skip}" in report
    assert "987 decisions" in report
    assert "400 steps/s at 4 environments" in report
    assert "Omitted from the README chart: 8." in report
    assert "| 8 | 100.0 | 300.0 | 3.00\u00d7 | [3.00, 3.00] |" in report
    assert "Breakout" not in report and "FirstWall" not in report
    assert len(report.split()) < 600
    contract = {
        "mlflow_url": "https://example.com/run",
        "decisions": 987,
        "total_captured_decisions": 1000,
        "reset_noop_prefix": 3,
        "benchmark_differences": ["seeded controls"],
        "limitations": "Known excerpt limitation.",
    }
    policy_report = render_benchmarks(result, profile, metadata, contract)
    assert "captured policy controls for both timing and replay" in policy_report
    assert "seeded controls" not in policy_report
    assert "Known excerpt limitation." in policy_report
    result["sampling"]["mode"] = "smoke"
    for row in result["comparison"]["shapes"].values():
        row["statistics"]["bootstrap"] = None
    metadata["diagnostic"] = True
    smoke = render_benchmarks(result, profile, metadata, None)
    assert "**DIAGNOSTIC" in smoke and "No confidence interval" in smoke
    assert "| unavailable |" in smoke
    assert "**2.00\u00d7 candidate/reference throughput" not in smoke


def test_history_keeps_only_prior_proof_references_and_rejects_tampering(fake_bundle):
    previous = export_publication(fake_bundle, proof_url="https://example.com/old")
    # Same proof re-export must not add its old prose or duplicate reference.
    output = export_publication(fake_bundle, previous.with_name("new"), previous=previous)
    assert read_json(output / "publication.json")["earlier_proofs"] == []
    (previous / "benchmarks.md").write_text("old benchmark prose")
    with pytest.raises(ValueError, match="previous publication artifact hash mismatch"):
        export_publication(fake_bundle, previous.with_name("tampered-history"), previous=previous)


def test_verifier_pin_uses_clean_source_and_never_labels_dirty_source_as_published(monkeypatch):
    from turbobench.publication import _verifier

    revision = "a" * 40
    monkeypatch.setattr(
        "turbobench.publication.subprocess.check_output",
        lambda args, **kwargs: "" if "status" in args else revision + "\n",
    )
    assert _verifier()["source_revision"] == revision
    monkeypatch.setattr(
        "turbobench.publication.subprocess.check_output",
        lambda *args, **kwargs: " M src/turbobench/publication.py\n",
    )
    verifier = _verifier()
    assert verifier["source_revision"] is None
    assert not verifier["installed_release"]
