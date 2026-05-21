#!/usr/bin/env python3
"""Run TUM Open3D CPU/SYCL comparison benchmarks and write reports."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODES = {
    "odometry": "tum_tensor_rgbd_odometry.py",
    "dense-slam": "tum_dense_slam.py",
    "vbg-slam": "tum_vbg_slam.py",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def sanitize(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def clean_line(value: str) -> str:
    return ANSI_RE.sub("", value).strip()


def extract_error(stdout: str, stderr: str) -> str:
    lines = [clean_line(line) for line in (stderr + "\n" + stdout).splitlines()]
    lines = [line for line in lines if line]
    for line in reversed(lines):
        if "RuntimeError:" in line or "Open3D Error" in line:
            return line
    return lines[-1] if lines else "failed"


def run_case(
    examples_dir: Path,
    mode: str,
    sequence: Path,
    device: str,
    max_frames: int,
    output_dir: Path,
    timeout_s: int,
    write_artifacts: bool,
) -> dict[str, Any]:
    json_out = output_dir / f"{mode}_{sanitize(device)}.json"
    cmd = [
        sys.executable,
        str(examples_dir / MODES[mode]),
        "--sequence",
        str(sequence),
        "--device",
        device,
        "--max-frames",
        str(max_frames),
        "--json-out",
        str(json_out),
    ]

    if write_artifacts and mode in {"dense-slam", "vbg-slam"}:
        cmd += ["--pointcloud-out", str(output_dir / f"{mode}_{sanitize(device)}.ply")]

    t0 = time.time()
    completed = subprocess.run(
        cmd,
        cwd=examples_dir.parent,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    wall_s = time.time() - t0

    case: dict[str, Any] = {
        "mode": mode,
        "device": device,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "wall_s": wall_s,
        "command": cmd,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode == 0 and json_out.exists():
        case["metrics"] = json.loads(json_out.read_text())
    else:
        case["error"] = extract_error(completed.stdout, completed.stderr)
    return case


def metric_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# TUM Open3D CPU vs ROCm Benchmark",
        "",
        f"- Sequence: `{summary['sequence']}`",
        f"- Max frames: `{summary['max_frames']}`",
        f"- Generated: `{summary['generated_at']}`",
        "",
        "| Mode | Device | Status | FPS / edges/s | Mean frame s | Mean track s | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for case in summary["cases"]:
        if case["ok"]:
            metrics = case["metrics"]
            rate_key = "edges_per_second" if case["mode"] == "odometry" else "frames_per_second"
            frame_key = "mean_edge_s" if case["mode"] == "odometry" else "mean_frame_s"
            notes_key = "processed_edges" if case["mode"] == "odometry" else "tracked_edges"
            notes = f"tracked={metrics.get(notes_key, '')}"
            if metrics.get("failed_edges"):
                notes += f", failed={metrics['failed_edges']}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        case["mode"],
                        f"`{case['device']}`",
                        "ok",
                        metric_value(metrics, rate_key),
                        metric_value(metrics, frame_key),
                        metric_value(metrics, "mean_track_s"),
                        notes,
                    ]
                )
                + " |"
            )
        else:
            err = case.get("error") or "failed"
            lines.append(
                f"| {case['mode']} | `{case['device']}` | failed |  |  |  | `{err}` |"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--devices", nargs="+", default=["CPU:0", "SYCL:0"])
    parser.add_argument("--modes", nargs="+", choices=sorted(MODES), default=list(MODES))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output/benchmark"))
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--write-artifacts", action="store_true")
    args = parser.parse_args()

    examples_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sequence": str(args.sequence.resolve()),
        "max_frames": args.max_frames,
        "devices": args.devices,
        "modes": args.modes,
        "cases": [],
    }

    for mode in args.modes:
        for device in args.devices:
            print(f"running {mode} on {device}", flush=True)
            try:
                case = run_case(
                    examples_dir,
                    mode,
                    args.sequence,
                    device,
                    args.max_frames,
                    output_dir,
                    args.timeout_s,
                    args.write_artifacts,
                )
            except subprocess.TimeoutExpired as exc:
                case = {
                    "mode": mode,
                    "device": device,
                    "ok": False,
                    "returncode": None,
                    "wall_s": args.timeout_s,
                    "error": [f"timeout after {exc.timeout}s"],
                }
            summary["cases"].append(case)

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "summary.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary, report_path)
    print(json.dumps({"summary": str(summary_path), "report": str(report_path)}, indent=2))
    return 0 if any(case["ok"] for case in summary["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
