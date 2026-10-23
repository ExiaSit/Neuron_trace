"""Resample and sort G-Cut SWCs with the Vaa3D plugins.

The filename is kept for compatibility with existing callers. In the production
pipeline this stage now runs after G-Cut, with a resampling step of 1.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from pipeline_config import PATHS


DEFAULT_VAA3D = Path("/home/pzy/pzy/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x")
SWC_COLUMNS = ("#id", "type", "x", "y", "z", "r", "p")


def _run_vaa3d(
    vaa3d: str | Path,
    plugin: str,
    function: str,
    swc_in: str | Path,
    swc_out: str | Path,
    parameters: list[str] | None = None,
    timeout: float = 300.0,
) -> None:
    command = [
        "xvfb-run", "-a", "-s", "-screen 0 640x480x16",
        str(vaa3d), "-x", plugin, "-f", function,
        "-i", str(swc_in), "-o", str(swc_out),
    ]
    if parameters:
        command.extend(["-p", *parameters])
    try:
        subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        output = exc.output.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"Vaa3D {plugin}/{function} failed: {output}") from exc

    output_path = Path(swc_out)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Vaa3D did not create a non-empty output: {output_path}")


def _correct_resample_output(path: Path) -> None:
    """Apply the two corrections from the supplied resample implementation."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("pid1", "pid\n1").replace(" -1 -1", "")
    path.write_text(text, encoding="utf-8")


def _read_swc_dataframe(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        sep=r"\s+",
        names=SWC_COLUMNS,
        comment="#",
        index_col=False,
        engine="python",
    )
    if dataframe.empty:
        raise ValueError(f"SWC contains no nodes: {path}")
    if dataframe.isna().any().any():
        raise ValueError(f"SWC contains malformed rows: {path}")
    return dataframe


def output_is_valid(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        dataframe = _read_swc_dataframe(path)
        return int(dataframe.iloc[0]["type"]) == 1 and bool(
            (dataframe.iloc[1:]["type"].astype(int) == 3).all()
        )
    except (OSError, TypeError, ValueError):
        return False


def resample_swc(
    swc_in: str | Path,
    swc_out: str | Path,
    step: float = 2,
    vaa3d: str | Path = DEFAULT_VAA3D,
    correction: bool = True,
    timeout: float = 300.0,
) -> bool:
    if step <= 0 or not math.isfinite(step):
        raise ValueError(f"step must be a finite positive number, got {step}")
    print(swc_in, swc_out)
    _run_vaa3d(
        vaa3d,
        "resample_swc",
        "resample_swc",
        swc_in,
        swc_out,
        [f"{step:g}"],
        timeout,
    )
    if correction:
        _correct_resample_output(Path(swc_out))
    return True


def sort_swc(
    swc_in: str | Path,
    swc_out: str | Path | None = None,
    vaa3d: str | Path = DEFAULT_VAA3D,
    retype: bool = True,
    timeout: float = 300.0,
) -> bool:
    source = Path(swc_in)
    destination = Path(swc_out) if swc_out is not None else source
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vaa3d_sort_", dir=destination.parent) as temp_dir:
        plugin_output = Path(temp_dir) / "sorted.swc"
        _run_vaa3d(
            vaa3d,
            "sort_neuron_swc",
            "sort_swc",
            source,
            plugin_output,
            timeout=timeout,
        )
        dataframe = _read_swc_dataframe(plugin_output)
        if retype:
            dataframe["type"] = 3
            dataframe.loc[dataframe.index[0], "type"] = 1
        dataframe.to_csv(destination, sep=" ", index=False)
    return True


def resample_sort_swc(
    swc_in: str | Path,
    swc_out: str | Path,
    step: float = 2,
    vaa3d: str | Path = DEFAULT_VAA3D,
    retype: bool = True,
    correction: bool = True,
    timeout: float = 300.0,
) -> bool:
    destination = Path(swc_out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vaa3d_resample_", dir=destination.parent) as temp_dir:
        resampled = Path(temp_dir) / "resampled.swc"
        sorted_swc = Path(temp_dir) / "sorted.swc"
        resample_swc(swc_in, resampled, step, vaa3d, correction, timeout)
        sort_swc(resampled, sorted_swc, vaa3d, retype, timeout)
        os.replace(sorted_swc, destination)
    return True


def resample_swc_file(
    source: str | Path,
    destination: str | Path,
    step: float = 1.0,
    overwrite: bool = False,
    vaa3d_path: str | Path = DEFAULT_VAA3D,
    timeout: float = 300.0,
) -> dict[str, object]:
    """Pipeline-compatible wrapper around the supplied resample+sort logic."""
    source = Path(source)
    destination = Path(destination)
    vaa3d_path = Path(vaa3d_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not vaa3d_path.is_file():
        raise FileNotFoundError(vaa3d_path)
    input_nodes = len(_read_swc_dataframe(source))
    if not overwrite and output_is_valid(destination):
        return {
            "status": "skipped",
            "source": str(source),
            "destination": str(destination),
            "input_nodes": input_nodes,
            "output_nodes": len(_read_swc_dataframe(destination)),
        }

    resample_sort_swc(
        source,
        destination,
        step=step,
        vaa3d=vaa3d_path,
        retype=True,
        correction=True,
        timeout=timeout,
    )
    return {
        "status": "processed",
        "source": str(source),
        "destination": str(destination),
        "input_nodes": input_nodes,
        "output_nodes": len(_read_swc_dataframe(destination)),
        "engine": "Vaa3D resample_swc + sort_swc",
    }


def _worker(task: tuple[str, str, float, bool, str, float]) -> dict[str, object]:
    try:
        return resample_swc_file(*task)
    except Exception as exc:
        return {
            "status": "failed",
            "source": task[0],
            "destination": task[1],
            "error": f"{type(exc).__name__}: {exc}",
        }


def resample_folder(
    source_dir: Path,
    output_dir: Path,
    step: float,
    workers: int,
    overwrite: bool,
    log_path: Path | None,
    vaa3d_path: Path,
    timeout: float,
) -> int:
    sources = sorted(source_dir.glob("*.swc"))
    if not sources:
        raise FileNotFoundError(f"No .swc files found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(source), str(output_dir / source.name), step, overwrite, str(vaa3d_path), timeout)
        for source in sources
    ]
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    started = time.time()
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(_worker, task) for task in tasks]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                status = str(result["status"])
                counts[status] += 1
                if status == "failed" and log_handle is not None:
                    log_handle.write(f"{result['source']}\t{result['error']}\n")
                    log_handle.flush()
                if index == 1 or index % 100 == 0 or index == len(tasks):
                    elapsed = time.time() - started
                    print(f"[{index}/{len(tasks)}] {counts} rate={index / elapsed:.1f} files/s", flush=True)
    finally:
        if log_handle is not None:
            log_handle.close()
    return 1 if counts["failed"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample and sort selected G-Cut SWCs.")
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vaa3d", type=Path, default=Path(PATHS.get("vaa3d_resample_path", DEFAULT_VAA3D)))
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return resample_folder(
        Path(PATHS["gcut_selected_swc_dir"]),
        Path(PATHS["gcut_resampled_swc_dir"]),
        args.step,
        args.workers,
        args.overwrite,
        Path(PATHS["gcut_resample_error_log"]),
        args.vaa3d,
        args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
