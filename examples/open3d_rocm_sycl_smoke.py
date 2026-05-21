#!/usr/bin/env python3
"""Probe Open3D SYCL device discovery and a tiny tensor kernel."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

import open3d as o3d


def _sycl_available() -> bool:
    try:
        return bool(o3d.core.sycl.is_available())
    except TypeError:
        return bool(o3d.core.sycl.is_available(o3d.core.Device("SYCL:0")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="SYCL:0")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "open3d_version": o3d.__version__,
        "sycl_available": False,
        "sycl_devices": [],
        "device": args.device,
        "tensor_kernel_ok": False,
    }

    try:
        result["sycl_available"] = _sycl_available()
        result["sycl_devices"] = [str(device) for device in o3d.core.sycl.get_available_devices()]

        device = o3d.core.Device(args.device)
        data = o3d.core.Tensor.ones((1024, 1024), o3d.core.Dtype.Float32, device)
        value = ((data * 2.0) + 1.0).sum().cpu().item()
        result["tensor_kernel_ok"] = True
        result["tensor_kernel_sum"] = float(value)
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic probe.
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        if not args.json:
            traceback.print_exc()

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Open3D: {result['open3d_version']}")
        print(f"SYCL available: {result['sycl_available']}")
        print(f"SYCL devices: {result['sycl_devices']}")
        print(f"Tensor kernel on {args.device}: {result['tensor_kernel_ok']}")
        if "tensor_kernel_sum" in result:
            print(f"Tensor kernel sum: {result['tensor_kernel_sum']}")
        if "error" in result:
            print(f"{result['error_type']}: {result['error']}", file=sys.stderr)

    return 0 if result["tensor_kernel_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
