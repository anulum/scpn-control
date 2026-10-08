# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Historical baseline custody fixtures
"""Record real custody of preserved historical metrics without new measurement claims."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tools.promote_benchmark_baseline import REPO_ROOT


@dataclass(frozen=True)
class PromotionCorpus:
    """Owned recorded replay of the unchanged maintained baseline.

    Parameters
    ----------
    root, manifest, artifact : pathlib.Path
        Temporary repository and immutable recorded replay input files.
    source_digest : str
        Exact captured artifact-byte SHA-256 for public promotion calls.
    """

    root: Path
    manifest: Path
    artifact: Path
    source_digest: str

    def arguments(self) -> list[str]:
        """Return explicit copy-only CLI parameters without owner approval claims.

        Returns
        -------
        list of str
            Fixed suite/receipt identifiers and declared historical compatibility.
        """
        return [
            "--repository-root",
            str(self.root),
            "--source-manifest",
            str(self.manifest),
            "--expected-source-sha256",
            self.source_digest,
            "--baseline",
            "benchmarks/baselines/copied.json",
            "--suite",
            "custody",
            "--authority-ref",
            "owned-copy-no-owner-promotion",
            "--hardware-compatibility",
            "initial-baseline",
            "--promotion-id",
            "one",
        ]


def make_corpus(root: Path) -> PromotionCorpus:
    """Capture a declared replay through the actual recorded-campaign CLI.

    Parameters
    ----------
    root : pathlib.Path
        New owned repository for isolated metadata/file-custody assertions.

    Returns
    -------
    PromotionCorpus
        Successful captured source with unchanged historical numerical values.

    Notes
    -----
    This is metadata/IO evidence only. No fresh measurement, hardware match,
    owner promotion, physics or production admission is asserted.
    """
    root.mkdir(parents=True)
    baseline = root / "historical-baseline.json"
    baseline.write_bytes((REPO_ROOT / "benchmarks/baselines/capacitor_bank.json").read_bytes())
    producer = root / "replay.py"
    producer.write_text(
        """from pathlib import Path
import json,hashlib,sys
original=json.loads(Path(sys.argv[1]).read_bytes())
payload={"schema_version":"scpn-control.benchmark-regression.v1", "generated_utc":original["measured_utc"], "evidence_class":"historical_baseline_custody_replay", "production_claim_allowed":False, "provenance":original["provenance"], "benchmarks":original["benchmarks"], "new_numerical_measurement":False}
payload["payload_sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
output=Path(sys.argv[2]);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\\n",encoding="utf-8")
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/run_recorded_benchmark.py"),
            "--repository-root",
            str(root),
            "--records-root",
            "records",
            "--family",
            "custody",
            "--campaign-id",
            "historical-replay",
            "--evidence-class",
            "historical_baseline_custody_replay",
            "--artifact",
            "report=report.json",
            "--",
            sys.executable,
            str(producer),
            str(baseline),
            str(root / "report.json"),
        ],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = root / "records/runs/custody/historical-replay/manifest.json"
    value = cast(dict[str, object], json.loads(manifest.read_bytes()))
    entries = cast(list[dict[str, object]], value["artifacts"])
    entry = next(row for row in entries if row["role"] == "report")
    artifact = root / str(entry["immutable_path"])
    return PromotionCorpus(root, manifest, artifact, str(entry["sha256"]))
