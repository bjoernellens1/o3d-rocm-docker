#!/usr/bin/env python3
"""Probe Open3D tensor/SYCL capabilities from simple ops toward SLAM."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


Result = dict[str, Any]
CaseFn = Callable[[o3d.core.Device], dict[str, Any]]

RESULT_PREFIX = "CAPABILITY_RESULT_JSON="
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _sycl_available() -> bool:
    try:
        return bool(o3d.core.sycl.is_available())
    except TypeError:
        return bool(o3d.core.sycl.is_available(o3d.core.Device("SYCL:0")))


def _device_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "open3d_version": o3d.__version__,
        "o3d_sycl_backend": os.environ.get("O3D_SYCL_BACKEND", ""),
        "oneapi_device_selector": os.environ.get("ONEAPI_DEVICE_SELECTOR", ""),
        "sycl_device_filter": os.environ.get("SYCL_DEVICE_FILTER", ""),
        "sycl_available": False,
        "sycl_devices": [],
    }
    try:
        summary["sycl_available"] = _sycl_available()
        summary["sycl_devices"] = [str(d) for d in o3d.core.sycl.get_available_devices()]
    except Exception as exc:  # noqa: BLE001 - diagnostic summary should survive.
        summary["sycl_error_type"] = type(exc).__name__
        summary["sycl_error"] = str(exc)
    return summary


def case_sycl_discovery(device: o3d.core.Device) -> dict[str, Any]:
    summary = _device_summary()
    if str(device).startswith("SYCL"):
        summary["requested_device_available"] = str(device) in summary["sycl_devices"]
    else:
        summary["requested_device_available"] = True
    if not summary["requested_device_available"]:
        raise RuntimeError(f"requested device {device} is not in {summary['sycl_devices']}")
    return summary


def case_tensor_empty(device: o3d.core.Device) -> dict[str, Any]:
    tensor = o3d.core.Tensor.empty((8,), o3d.core.Dtype.Float32, device)
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def case_tensor_host_copy(device: o3d.core.Device) -> dict[str, Any]:
    tensor = o3d.core.Tensor(np.arange(8, dtype=np.float32), o3d.core.Dtype.Float32, device)
    return {"sum": float(tensor.sum().cpu().item())}


def case_tensor_zeros(device: o3d.core.Device) -> dict[str, Any]:
    tensor = o3d.core.Tensor.zeros((1024,), o3d.core.Dtype.Float32, device)
    return {"sum": float(tensor.sum().cpu().item())}


def case_tensor_elementwise(device: o3d.core.Device) -> dict[str, Any]:
    data = o3d.core.Tensor.ones((1024,), o3d.core.Dtype.Float32, device)
    value = (((data + 2.0) * 3.0) - 1.0).sum().cpu().item()
    return {"sum": float(value)}


def case_tensor_linalg(device: o3d.core.Device) -> dict[str, Any]:
    matrix = o3d.core.Tensor(
        [[4.0, 1.0], [2.0, 3.0]],
        o3d.core.Dtype.Float32,
        device,
    )
    inverse = o3d.core.inv(matrix)
    product = o3d.core.matmul(matrix, inverse).cpu().numpy()
    return {"trace": float(np.trace(product))}


def case_image_to_device(device: o3d.core.Device) -> dict[str, Any]:
    image = o3d.t.geometry.Image(np.zeros((16, 16, 3), dtype=np.uint8)).to(device)
    return {"rows": int(image.rows), "columns": int(image.columns)}


def case_raycasting_scene(device: o3d.core.Device) -> dict[str, Any]:
    scene = o3d.t.geometry.RaycastingScene(1, device)
    vertices = o3d.core.Tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        o3d.core.Dtype.Float32,
        device,
    )
    triangles = o3d.core.Tensor([[0, 1, 2]], o3d.core.Dtype.UInt32, device)
    scene.add_triangles(vertices, triangles)
    rays = o3d.core.Tensor(
        [[[0.25, 0.25, 1.0, 0.0, 0.0, -1.0]]],
        o3d.core.Dtype.Float32,
        device,
    )
    answer = scene.cast_rays(rays)
    t_hit = answer["t_hit"].cpu().numpy()
    return {"t_hit": float(t_hit.reshape(-1)[0])}


def case_tensor_icp(device: o3d.core.Device) -> dict[str, Any]:
    treg = o3d.t.pipelines.registration
    source = o3d.t.geometry.PointCloud(device)
    target = o3d.t.geometry.PointCloud(device)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    source.point.positions = o3d.core.Tensor(points, o3d.core.Dtype.Float32, device)
    target.point.positions = o3d.core.Tensor(points + np.array([0.01, 0.0, 0.0]), o3d.core.Dtype.Float32, device)
    init = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64, device)
    result = treg.icp(
        source,
        target,
        0.2,
        init,
        treg.TransformationEstimationPointToPoint(),
        treg.ICPConvergenceCriteria(max_iteration=2),
    )
    return {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "translation_x": float(result.transformation.cpu().numpy()[0, 3]),
    }


def case_rgbd_odometry(device: o3d.core.Device) -> dict[str, Any]:
    odom = o3d.t.pipelines.odometry
    y, x = np.mgrid[0:48, 0:64]
    color = np.stack(
        [
            (x * 4) % 256,
            (y * 5) % 256,
            ((x + y) * 3) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    depth = (1000 + x * 2 + y).astype(np.uint16)
    source = o3d.t.geometry.RGBDImage(
        o3d.t.geometry.Image(color).to(device),
        o3d.t.geometry.Image(depth).to(device),
    )
    target = o3d.t.geometry.RGBDImage(
        o3d.t.geometry.Image(color).to(device),
        o3d.t.geometry.Image(depth).to(device),
    )
    intrinsic = o3d.core.Tensor(
        [[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]],
        o3d.core.Dtype.Float64,
        device,
    )
    result = odom.rgbd_odometry_multi_scale(
        source,
        target,
        intrinsic,
        o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64, device),
        1000.0,
        3.0,
        [odom.OdometryConvergenceCriteria(1)],
        odom.Method.Hybrid,
    )
    return {"fitness": float(result.fitness), "inlier_rmse": float(result.inlier_rmse)}


def case_voxel_block_grid(device: o3d.core.Device) -> dict[str, Any]:
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=["tsdf", "weight"],
        attr_dtypes=[o3d.core.Dtype.Float32, o3d.core.Dtype.UInt16],
        attr_channels=[1, 1],
        voxel_size=0.02,
        block_resolution=4,
        block_count=16,
        device=device,
    )
    return {"active_blocks": int(vbg.hashmap().active_buf_indices().shape[0])}


def case_dense_slam_model(device: o3d.core.Device) -> dict[str, Any]:
    model = o3d.t.pipelines.slam.Model(
        0.02,
        4,
        16,
        o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64, device),
        device,
    )
    return {"model": type(model).__name__}


CASES: list[dict[str, Any]] = [
    {
        "name": "sycl_discovery",
        "stage": "00_device",
        "expected_sycl": "must pass before any SYCL workload can run",
        "fn": case_sycl_discovery,
    },
    {
        "name": "tensor_empty_allocation",
        "stage": "01_tensor_basic",
        "expected_sycl": "core tensor allocation",
        "fn": case_tensor_empty,
    },
    {
        "name": "tensor_host_copy_reduction",
        "stage": "01_tensor_basic",
        "expected_sycl": "host-to-device copy plus reduction",
        "fn": case_tensor_host_copy,
    },
    {
        "name": "tensor_zeros_reduction",
        "stage": "01_tensor_basic",
        "expected_sycl": "fill kernel plus reduction",
        "fn": case_tensor_zeros,
    },
    {
        "name": "tensor_elementwise_reduction",
        "stage": "02_tensor_kernels",
        "expected_sycl": "elementwise kernels plus reduction",
        "fn": case_tensor_elementwise,
    },
    {
        "name": "tensor_linalg_matmul_inv",
        "stage": "02_tensor_kernels",
        "expected_sycl": "SYCL linalg path if MKL SYCL backend is working",
        "fn": case_tensor_linalg,
    },
    {
        "name": "image_to_device",
        "stage": "03_tensor_geometry",
        "expected_sycl": "tensor image copy",
        "fn": case_image_to_device,
    },
    {
        "name": "raycasting_scene",
        "stage": "03_tensor_geometry",
        "expected_sycl": "Open3D has a SYCL RaycastingScene implementation",
        "fn": case_raycasting_scene,
    },
    {
        "name": "tensor_icp",
        "stage": "04_registration",
        "expected_sycl": "ICP pose kernels dispatch CPU/CUDA only after basic tensor ops work",
        "fn": case_tensor_icp,
    },
    {
        "name": "rgbd_odometry",
        "stage": "05_rgbd",
        "expected_sycl": "RGB-D odometry kernels dispatch CPU/CUDA only in Open3D v0.19",
        "fn": case_rgbd_odometry,
    },
    {
        "name": "voxel_block_grid",
        "stage": "06_vbg_slam",
        "expected_sycl": "expected CPU/CUDA-only hash backend in Open3D v0.19 source",
        "fn": case_voxel_block_grid,
    },
    {
        "name": "dense_slam_model",
        "stage": "06_vbg_slam",
        "expected_sycl": "inherits VoxelBlockGrid and RGB-D odometry limitations",
        "fn": case_dense_slam_model,
    },
]


def case_by_name(name: str) -> dict[str, Any]:
    for case in CASES:
        if case["name"] == name:
            return case
    raise KeyError(name)


def run_child(case_name: str, device_name: str) -> int:
    case = case_by_name(case_name)
    device = o3d.core.Device(device_name)
    started = time.time()
    result: Result = {
        "name": case["name"],
        "stage": case["stage"],
        "device": device_name,
        "expected_sycl": case["expected_sycl"],
        "ok": False,
    }
    try:
        details = case["fn"](device)
        result["ok"] = True
        result["details"] = details
    except Exception as exc:  # noqa: BLE001 - this is a compatibility probe.
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=12)
    result["elapsed_s"] = time.time() - started
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return 0


def parse_child_output(case: dict[str, Any], completed: subprocess.CompletedProcess[str]) -> Result:
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return {
        "name": case["name"],
        "stage": case["stage"],
        "expected_sycl": case["expected_sycl"],
        "ok": False,
        "returncode": completed.returncode,
        "error_type": "ProcessError",
        "error": "probe child did not emit JSON result",
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def run_parent(device_name: str, timeout_s: int) -> list[Result]:
    results: list[Result] = []
    for case in CASES:
        print(f"probing {case['stage']} {case['name']} on {device_name}", flush=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--device",
            device_name,
            "--case",
            case["name"],
        ]
        started = time.time()
        try:
            completed = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
            result = parse_child_output(case, completed)
            result["returncode"] = completed.returncode
            if completed.returncode != 0 and result.get("ok"):
                result["ok"] = False
                result["error_type"] = "ProcessError"
                result["error"] = f"child exited with {completed.returncode}"
            if completed.stderr:
                result["stderr_tail"] = completed.stderr[-4000:]
            if completed.stdout and RESULT_PREFIX not in completed.stdout:
                result["stdout_tail"] = completed.stdout[-4000:]
        except subprocess.TimeoutExpired as exc:
            result = {
                "name": case["name"],
                "stage": case["stage"],
                "device": device_name,
                "expected_sycl": case["expected_sycl"],
                "ok": False,
                "error_type": "TimeoutExpired",
                "error": f"timeout after {exc.timeout}s",
            }
        result["wall_s"] = time.time() - started
        results.append(result)
    return results


def compact_error(result: Result) -> str:
    error = result.get("error") or result.get("stderr_tail") or ""
    error = ANSI_RE.sub("", str(error))
    return " ".join(error.replace("\n", " ").split())[:300]


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Open3D SYCL Capability Probe",
        "",
        f"- Device: `{summary['device']}`",
        f"- Generated: `{summary['generated_at']}`",
        f"- Open3D: `{summary['environment'].get('open3d_version', '')}`",
        f"- SYCL devices: `{summary['environment'].get('sycl_devices', [])}`",
        f"- O3D_SYCL_BACKEND: `{summary['environment'].get('o3d_sycl_backend', '')}`",
        "",
        "| Stage | Case | Expected SYCL status | Result | Elapsed s | Error |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for result in summary["results"]:
        elapsed = result.get("elapsed_s", result.get("wall_s", 0.0))
        status = "ok" if result.get("ok") else "failed"
        error = "" if result.get("ok") else f"`{compact_error(result)}`"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result["stage"]),
                    str(result["name"]),
                    str(result["expected_sycl"]),
                    status,
                    f"{float(elapsed):.4f}",
                    error,
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="SYCL:0")
    parser.add_argument("--case", choices=[case["name"] for case in CASES])
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    if args.case:
        return run_child(args.case, args.device)

    summary: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": args.device,
        "environment": _device_summary(),
        "results": run_parent(args.device, args.timeout_s),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(summary, args.md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
