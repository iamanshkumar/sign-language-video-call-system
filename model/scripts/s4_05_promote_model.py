"""Step 4.5: reversible promotion of a validated candidate to models/best_model.pt.

  # show what would happen
  PYTHONPATH=src .venv/bin/python scripts/s4_05_promote_model.py --candidate models/candidates/<id> --dry-run
  # promote (backs up the current files first, prints the restore command)
  PYTHONPATH=src .venv/bin/python scripts/s4_05_promote_model.py --candidate models/candidates/<id>
  # undo
  PYTHONPATH=src .venv/bin/python scripts/s4_05_promote_model.py --restore models/backups/<stamp>

Only a candidate whose export_report.json says all_checks_pass can be promoted. The files being
replaced are copied to models/backups/<UTC stamp>/ (with SHA-256s) before anything is written;
every copy is verified by checksum. A restore also backs up the files it replaces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from asl.config import PROJECT_ROOT, STEP2_CONFIG, load_yaml
from asl.inference.recognizer import SignRecognizer

FILES = ("best_model.pt", "class_mapping.json")
MODELS = PROJECT_ROOT / "models"
STEP3 = PROJECT_ROOT / "configs" / "step3_sequences.yaml"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def backup_current(reason: str) -> Path | None:
    present = [f for f in FILES if (MODELS / f).exists()]
    if not present:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = MODELS / "backups" / stamp
    dest.mkdir(parents=True)
    manifest = {"created_utc": stamp, "reason": reason, "files": {}}
    for f in present:
        shutil.copy2(MODELS / f, dest / f)
        if sha(dest / f) != sha(MODELS / f):
            raise SystemExit(f"backup of {f} does not match; aborting before any change")
        manifest["files"][f] = sha(dest / f)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return dest


def install(src_dir: Path) -> dict:
    out = {}
    for f in FILES:
        tmp = MODELS / f".{f}.tmp"
        shutil.copy2(src_dir / f, tmp)
        if sha(tmp) != sha(src_dir / f):
            tmp.unlink()
            raise SystemExit(f"copy of {f} does not match its source")
        tmp.replace(MODELS / f)  # atomic per file
        out[f] = sha(MODELS / f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--candidate", help="models/candidates/<experiment_id>")
    g.add_argument("--restore", help="models/backups/<stamp>")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    src = PROJECT_ROOT / (args.candidate or args.restore)
    missing = [f for f in FILES if not (src / f).exists()]
    if missing:
        raise SystemExit(f"{src} lacks {missing}")
    if args.candidate:
        rep = json.loads((src / "export_report.json").read_text())
        if not rep.get("all_checks_pass"):
            raise SystemExit("candidate did not pass its export checks; refusing to promote")
    # a promoted candidate must load through the inference loader (architecture/mapping/preprocessing);
    # a restore puts back an exact earlier state, which may predate the loader (warn only)
    try:
        SignRecognizer.load(src / "best_model.pt", src / "class_mapping.json", load_yaml(STEP2_CONFIG), load_yaml(STEP3))
    except Exception as e:  # noqa: BLE001
        if args.candidate:
            raise SystemExit(f"candidate does not load through the inference loader: {e}")
        print(f"warning: restored files do not load through the inference loader ({e}); restoring anyway")
    current = {f: sha(MODELS / f) for f in FILES if (MODELS / f).exists()}
    incoming = {f: sha(src / f) for f in FILES}
    plan = {"action": "promote" if args.candidate else "restore", "source": str(src.relative_to(PROJECT_ROOT)),
            "current": current, "incoming": incoming}
    if args.dry_run:
        print(json.dumps({**plan, "dry_run": True}, indent=1))
        return
    backup = backup_current(f"before {plan['action']} from {plan['source']}")
    installed = install(src)
    print(json.dumps({**plan, "backup": str(backup.relative_to(PROJECT_ROOT)) if backup else None,
                      "installed": installed}, indent=1))
    if backup:
        print(f"\nTo undo: PYTHONPATH=src .venv/bin/python scripts/s4_05_promote_model.py "
              f"--restore {backup.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
