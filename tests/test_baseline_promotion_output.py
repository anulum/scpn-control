# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real baseline publication and failure recovery tests
"""Publish complete historical declarations and interrupt actual filesystem operations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from baseline_promotion_fixtures import make_corpus

from tools.baseline_promotion_output import PromotionCollisionError, publish_promotion
from tools.baseline_promotion_payloads import PromotionInputError, build_baseline
from tools.promote_benchmark_baseline import REPO_ROOT


def test_public_publisher_verifies_and_reuses_actual_predecessor_archive(tmp_path: Path) -> None:
    """Two recorded copy promotions preserve the previous bytes and immutable first receipt."""
    corpus = make_corpus(tmp_path / "corpus")
    report = cast(dict[str, object], json.loads(corpus.artifact.read_bytes()))
    baseline = build_baseline(
        report,
        suite="custody",
        source_manifest=corpus.manifest.relative_to(corpus.root).as_posix(),
        source_sha256=corpus.source_digest,
        authority_ref="copy",
        hardware_compatibility="initial-baseline",
        promoted_utc="2026-10-08T00:00:00Z",
    )
    target = corpus.root / "benchmarks/baselines/copy.json"
    target.parent.mkdir(parents=True)
    previous = (corpus.root / "historical-baseline.json").read_bytes()
    target.write_bytes(previous)
    archive = (
        corpus.root / "benchmarks/baseline_history/custody/baselines" / (hashlib.sha256(previous).hexdigest() + ".json")
    )
    archive.parent.mkdir(parents=True)
    archive.write_bytes(previous)
    first = publish_promotion(
        baseline,
        promotion_id="one",
        repository_root=corpus.root,
        manifest_path=corpus.manifest,
        artifact_path=corpus.artifact,
        baseline_path=target,
        campaign_id="historical-replay",
        artifact_role="report",
    )
    assert archive.read_bytes() == previous
    first_bytes = first.read_bytes()
    second = publish_promotion(
        baseline,
        promotion_id="two",
        repository_root=corpus.root,
        manifest_path=corpus.manifest,
        artifact_path=corpus.artifact,
        baseline_path=target,
        campaign_id="historical-replay",
        artifact_role="report",
    )
    assert first.read_bytes() == first_bytes
    assert (
        json.loads(second.read_bytes())["previous_baseline"]["sha256"]
        == hashlib.sha256(target.read_bytes()).hexdigest()
    )
    with pytest.raises(PromotionCollisionError):
        publish_promotion(
            baseline,
            promotion_id="one",
            repository_root=corpus.root,
            manifest_path=corpus.manifest,
            artifact_path=corpus.artifact,
            baseline_path=target,
            campaign_id="historical-replay",
            artifact_role="report",
        )
    assert first.read_bytes() == first_bytes
    archive.write_bytes(b"corrupt retained archive")
    target.write_bytes(previous)
    with pytest.raises(PromotionInputError):
        publish_promotion(
            baseline,
            promotion_id="three",
            repository_root=corpus.root,
            manifest_path=corpus.manifest,
            artifact_path=corpus.artifact,
            baseline_path=target,
            campaign_id="historical-replay",
            artifact_role="report",
        )
    assert target.read_bytes() == previous and archive.read_bytes() == b"corrupt retained archive"


_AUDITED = r"""
import sys,json
from pathlib import Path
from tools.promote_benchmark_baseline import main, promote
from tools.inventory_file_output import InventoryOutputError
root,mode=Path(sys.argv[1]),sys.argv[2]
manifest=root/'records/runs/custody/historical-replay/manifest.json'
entry=json.loads(manifest.read_bytes())['artifacts'][0]
target=root/'benchmarks/baselines/copied.json'
receipt=root/'benchmarks/baseline_history/custody/promotions/one.json'
def deny(event,arguments):
    if event=='os.link' and (mode=='unsupported' or str(arguments[1])==str(receipt)):
        if mode=='race':receipt.write_bytes(b'another writer receipt')
        if mode=='changed':target.write_bytes(b'another writer baseline')
        if mode!='race':raise PermissionError('private permission detail')
    if mode=='recovery' and event=='os.rename' and str(arguments[0]).endswith('.backup'):
        raise PermissionError('private recovery detail')
sys.addaudithook(deny)
try:
    promote(source_manifest=manifest,artifact_role='report',expected_source_sha256=entry['sha256'],
            baseline_path=target,suite='custody',authority_ref='owned-copy-no-owner-approval',
            hardware_compatibility='initial-baseline',promotion_id='one',repository_root=root)
except InventoryOutputError as error:
    print(json.dumps({'kind':'recovery','message':str(error),'retained':[str(p) for p in error.recovery_paths]}))
except OSError:
    print(json.dumps({'kind':'native','retained':[]}))
else:raise AssertionError('publication unexpectedly completed')
"""


@pytest.mark.parametrize("mode", ["link", "race", "unsupported", "changed", "recovery"])
def test_real_exclusive_publication_recovers_or_retains_predecessors(tmp_path: Path, mode: str) -> None:
    """OS audit permissions and name collisions exercise the public operation without private mocks."""
    corpus = make_corpus(tmp_path / "corpus")
    target = corpus.root / "benchmarks/baselines/copied.json"
    target.parent.mkdir(parents=True)
    previous = (corpus.root / "historical-baseline.json").read_bytes()
    target.write_bytes(previous)
    before = corpus.manifest.read_bytes(), corpus.artifact.read_bytes()
    result = subprocess.run(
        [sys.executable, "-c", _AUDITED, str(corpus.root), mode],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert before == (corpus.manifest.read_bytes(), corpus.artifact.read_bytes())
    if mode in {"changed", "recovery"}:
        assert receipt["kind"] == "recovery"
        retained = [Path(name) for name in receipt["retained"]]
        assert retained and all(path.is_file() for path in retained)
        assert any(path.read_bytes() == previous for path in retained)
        if mode == "changed":
            assert target.read_bytes() == b"another writer baseline"
        else:
            assert json.loads(target.read_bytes())["schema_version"] == "scpn-control.benchmark-baseline.v1"
    else:
        assert receipt["kind"] == "native"
        assert target.read_bytes() == previous
        assert not list(corpus.root.rglob("*.tmp")) and not list(corpus.root.rglob("*.backup"))
    final = corpus.root / "benchmarks/baseline_history/custody/promotions/one.json"
    if mode == "race":
        assert final.read_bytes() == b"another writer receipt"
    else:
        assert not final.exists()


@pytest.mark.parametrize(
    "kind",
    [
        "id",
        "campaign",
        "role",
        "manifest-outside",
        "artifact-outside",
        "output-outside",
        "no-runs",
        "history",
        "source",
        "schema",
        "suite",
        "production",
        "promotion",
        "reference",
        "digest",
        "hardware",
        "metrics",
    ],
)
def test_public_publisher_refuses_incoherent_or_escaping_declarations(tmp_path: Path, kind: str) -> None:
    """Invalid caller declarations cannot escape or mutate an actual recorded corpus."""
    corpus = make_corpus(tmp_path / "corpus")
    report = cast(dict[str, object], json.loads(corpus.artifact.read_bytes()))
    baseline = build_baseline(
        report,
        suite="custody",
        source_manifest=corpus.manifest.relative_to(corpus.root).as_posix(),
        source_sha256=corpus.source_digest,
        authority_ref="copy",
        hardware_compatibility="initial-baseline",
        promoted_utc="2026-10-08T00:00:00Z",
    )
    manifest, artifact = corpus.manifest, corpus.artifact
    target = Path("benchmarks/baselines/copied.json")
    identifier, campaign, role = "one", "historical-replay", "report"
    metadata = cast(dict[str, object], baseline["promotion"])
    if kind == "id":
        identifier = "../escape"
    elif kind == "campaign":
        campaign = " "
    elif kind == "role":
        role = " "
    elif kind == "manifest-outside":
        manifest = tmp_path / "external.json"
    elif kind == "artifact-outside":
        artifact = tmp_path / "external.json"
    elif kind == "output-outside":
        target = tmp_path / "external.json"
    elif kind == "no-runs":
        manifest = corpus.root / "manifest.json"
    elif kind == "history":
        target = Path("benchmarks/baseline_history/illegal.json")
    elif kind == "source":
        target = corpus.artifact
    elif kind == "schema":
        baseline["schema_version"] = "unsupported"
    elif kind == "suite":
        baseline["suite"] = "../escape"
    elif kind == "production":
        baseline["production_claim_allowed"] = True
    elif kind == "promotion":
        baseline["promotion"] = []
    elif kind == "reference":
        metadata["source_manifest"] = "other-manifest"
    elif kind == "digest":
        metadata["source_artifact_sha256"] = "invalid"
    elif kind == "hardware":
        metadata["hardware_compatibility"] = "guessed"
    elif kind == "metrics":
        baseline["baseline_sha256"] = "0" * 64
    else:
        raise AssertionError(kind)
    before = {path.relative_to(corpus.root): path.read_bytes() for path in corpus.root.rglob("*") if path.is_file()}
    with pytest.raises(PromotionInputError):
        publish_promotion(
            baseline,
            repository_root=corpus.root,
            manifest_path=manifest,
            artifact_path=artifact,
            baseline_path=target,
            campaign_id=campaign,
            artifact_role=role,
            promotion_id=identifier,
        )
    assert {
        path.relative_to(corpus.root): path.read_bytes() for path in corpus.root.rglob("*") if path.is_file()
    } == before
    assert not (tmp_path / "external.json").exists()
