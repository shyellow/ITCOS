"""Compare the bundled 10 m inputs or a reproduced run with archived checkpoints."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent


def file_digest(path):
    """Ignore only the dBASE last-update date; all fields and records are checked."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if path.suffix.lower() == ".dbf":
            header = bytearray(stream.read(32))
            header[1:4] = bytes(3)
            digest.update(header)
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digests(path):
    """Hash each NPY array including dtype and shape, independently of ZIP metadata."""
    values = {}
    with np.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            stream = io.BytesIO()
            np.lib.format.write_array(stream, arrays[name], version=(1, 0), allow_pickle=False)
            values[name] = hashlib.sha256(stream.getbuffer()).hexdigest()
    return values


def event_digest(path):
    """Compare the ordered proposals and accepted moves, excluding elapsed time."""
    count = 0
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            event.pop("seconds", None)
            text = json.dumps(event, sort_keys=True, separators=(",", ":"))
            digest.update((text + "\n").encode("utf-8"))
            count += 1
    return {"records": count, "sha256": digest.hexdigest()}


def verify_inputs(expected):
    """Input hashes are literal file hashes, including the original DBF header."""
    failed = []
    for name, digest in expected.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failed.append(name)
    return failed


def verify_outputs(folder, expected):
    """Check geometry, every DBF record, intermediate paths and vertex/face arrays."""
    failed = []
    for name, digest in expected["files"].items():
        path = folder / name
        if not path.is_file() or file_digest(path) != digest:
            failed.append(name)
    for name, reference in expected["arrays"].items():
        path = folder / name
        if not path.is_file():
            failed.append(name)
            continue
        actual = array_digests(path)
        for key in sorted(reference.keys() | actual.keys()):
            if reference.get(key) != actual.get(key):
                failed.append(f"{name} / {key}")
    for name, digest in expected["search_logs"].items():
        path = folder / name
        if not path.is_file() or event_digest(path) != digest:
            failed.append(name)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    checks = json.loads((ROOT / "reference/checkpoints.json").read_text(encoding="utf-8"))
    failed = verify_inputs(checks["inputs"])
    report = {"passed": not failed, "inputs_checked": len(checks["inputs"])}
    if not args.inputs_only:
        if not args.output.is_dir():
            parser.error(f"Output directory not found: {args.output}")
        outputs = checks["outputs"]
        failed.extend(verify_outputs(args.output, outputs))
        report.update(files_checked=len(outputs["files"]),
                      arrays_checked=sum(len(group) for group in outputs["arrays"].values()),
                      search_logs_checked=len(outputs["search_logs"]))
    report.update(passed=not failed, mismatches=failed)
    message = json.dumps(report, indent=2)
    print(message)
    if not args.inputs_only:
        (args.output / "reproduction_check.json").write_text(message + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
