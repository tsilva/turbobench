import json

import pytest

from turbobench.privacy import private_url, public_text, require_public_proof, training_reference


@pytest.mark.parametrize(
    "url",
    [
        "https://tracker.private.invalid:5000/run",
        "http://192.0.2.4:8000/metrics",
        "http://[2001:db8::1]:8080/run",
        "https://login:password@github.com/repo",
        "https://github.com:8443/repo",
    ],
)
def test_private_access_urls_are_removed(url):
    assert private_url(url)
    output = public_text("Measured data: " + url)
    assert url not in output and "private endpoint omitted" in output


def test_hardware_and_public_artifact_links_remain():
    text = "AMD Ryzen CPU, x86_64 Linux; https://github.com/owner/repo/releases/tag/v1"
    assert public_text(text) == text
    assert (
        public_text("remote host: server.private.invalid")
        == "remote host: [access details omitted]"
    )
    assert (
        training_reference("https://tracker.private.invalid/#/experiments/1/runs/abc123")
        == "training run `abc123`"
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"tracking": {"uri": "http://192.0.2.4:8000"}},
        {"hostname": "private-host"},
        {"ssh_alias": "private-machine"},
        {"ip_address": "2001:db8::1"},
        {"port": 2222},
        {"mlflow_url": "https://github.com/owner/run"},
    ],
)
def test_public_proof_gate_rejects_access_details_without_echoing_them(tmp_path, metadata):
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="private machine access details") as exc:
        require_public_proof(tmp_path)
    assert "192.0.2.4" not in str(exc.value) and "2222" not in str(exc.value)


def test_non_routable_policy_identity_is_allowed(tmp_path):
    (tmp_path / "contract.json").write_text(
        json.dumps({"mlflow_url": "training-run:abc123", "hardware": {"cpu": "Example CPU"}})
    )
    require_public_proof(tmp_path)


def test_existing_proof_with_private_metadata_cannot_be_exported(fake_bundle, monkeypatch):
    from turbobench.publication import export_publication

    # Privacy is a distinct gate after integrity verification; exercise a previously verified legacy proof.
    monkeypatch.setattr("turbobench.publication.verify_bundle", lambda _: {"passed": True})
    (fake_bundle / "private.json").write_text(
        json.dumps({"tracking_uri": "https://tracker.private.invalid:5000"})
    )
    output = fake_bundle.with_name("public")
    with pytest.raises(ValueError, match="private machine access details"):
        export_publication(fake_bundle, output)
    assert not output.exists()


def test_configure_does_not_echo_private_machine_or_policy_paths(monkeypatch, capsys):
    from turbobench.cli import main

    monkeypatch.setattr(
        "turbobench.defaults.configure",
        lambda *_: {
            "benchmark_host": "ssh-user@measurement.private.invalid",
            "policy": "/private/policy/path",
            "policy_id": "opaque-proof-id",
        },
    )
    assert main(["configure", "--benchmark-host", "measurement.private.invalid"]) == 0
    text = capsys.readouterr().out
    assert "measurement.private.invalid" not in text and "ssh-user" not in text
    assert "/private/policy/path" not in text and "opaque-proof-id" in text


def test_remote_command_failure_never_forwards_ssh_stderr(monkeypatch, capsys):
    import subprocess

    from turbobench.workflow import _run

    def failing(*args, **kwargs):
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(
            args[0],
            255,
            stdout="",
            stderr="ssh: connect to host measurement.private.invalid port 2222: refused",
        )

    monkeypatch.setattr("turbobench.workflow.subprocess.run", failing)
    with pytest.raises(RuntimeError) as exc:
        _run(["ssh", "measurement.private.invalid", "run"])
    assert "measurement.private.invalid" not in str(exc.value) and "2222" not in str(exc.value)
    assert capsys.readouterr().err == ""


def test_private_proof_url_is_rejected_before_export(fake_bundle):
    from turbobench.publication import export_publication

    with pytest.raises(ValueError, match="proof URL"):
        export_publication(fake_bundle, proof_url="http://192.0.2.4:8000/proof")
    assert not fake_bundle.with_name(fake_bundle.name + "-publication").exists()
