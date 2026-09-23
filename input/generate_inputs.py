#!/usr/bin/env python3
"""Generate deterministic synthetic tensors for NPU AVS workload testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VERSION = 1
MASTER_SEED = 0x4E50555F41565331  # ASCII-like marker: NPU_AVS1


class Pcg32:
    """Small, explicitly specified PCG-XSH-RR generator."""

    MASK64 = (1 << 64) - 1

    def __init__(self, seed: int, sequence: int = 0xDA3E39CB94B95BDB):
        self.state = 0
        self.increment = ((sequence << 1) | 1) & self.MASK64
        self.next_u32()
        self.state = (self.state + (seed & self.MASK64)) & self.MASK64
        self.next_u32()

    def next_u32(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + self.increment) & self.MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rotation = (old >> 59) & 31
        return ((xorshifted >> rotation) | (xorshifted << ((-rotation) & 31))) & 0xFFFFFFFF

    def integer(self, low: int, high: int) -> int:
        if high < low:
            raise ValueError("invalid integer range")
        return low + self.next_u32() % (high - low + 1)


def seed_for(workload: int, group: int, pattern: int, index: int) -> int:
    value = MASTER_SEED
    for part in (workload, group, pattern, index):
        value ^= (part + 0x9E3779B97F4A7C15 + ((value << 6) & Pcg32.MASK64) + (value >> 2))
        value &= Pcg32.MASK64
    return value


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_sample(records: list[dict], relative_path: str, data: bytes, *, workload: str,
                 group: str, pattern: str, seed: int, dtype: str, shape: list[int],
                 source: dict | None = None) -> None:
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    item = {
        "path": relative_path.replace("\\", "/"),
        "workload": workload,
        "group": group,
        "pattern": pattern,
        "seed": f"0x{seed:016x}",
        "dtype": dtype,
        "shape": shape,
        "byte_order": "little" if dtype == "float32" else "not_applicable",
        "bytes": len(data),
        "sha256": sha256(data),
    }
    if source:
        item["source"] = source
    records.append(item)


def pack_i8(values: list[int]) -> bytes:
    return bytes(v & 0xFF for v in values)


def pack_f32(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def generate_integer_tensor(size: int, pattern: str, rng: Pcg32, low: int, high: int,
                            width: int, channels: int = 1, variant: int = 0) -> list[int]:
    midpoint = (low + high + 1) // 2
    values: list[int] = []
    for i in range(size):
        pixel = i // channels
        x = pixel % width
        y = pixel // width
        channel = i % channels
        if pattern == "zero":
            value = 0
        elif pattern == "midpoint":
            value = midpoint
        elif pattern == "minimum":
            value = low
        elif pattern == "maximum":
            value = high
        elif pattern == "gradient":
            span = high - low
            value = low + ((x + variant + channel * 3) % width) * span // max(1, width - 1)
        elif pattern == "checker":
            block = 1 + variant % 4
            value = high if ((x // block) + (y // block) + channel) % 2 else low
        elif pattern == "stripes":
            period = 2 + variant % 7
            value = high if ((x + y + channel + variant) % period) == 0 else midpoint
        elif pattern == "impulse":
            target = (size // 2 + variant * 97) % size
            value = high if i == target else midpoint
        elif pattern == "low_random":
            radius = max(1, (high - low) // 16)
            value = rng.integer(max(low, midpoint - radius), min(high, midpoint + radius))
        elif pattern == "dense_random":
            value = rng.integer(low, high)
        elif pattern == "boundary":
            selector = (i + variant) % 8
            value = low if selector < 3 else high if selector < 6 else midpoint
        elif pattern == "sparse":
            value = rng.integer(low, high) if rng.integer(0, 15) == 0 else midpoint
        elif pattern == "banded":
            band = (y + variant) % 12
            value = max(low, min(high, midpoint + (band - 6) * max(1, (high - low) // 24)))
        elif pattern == "pulsed":
            value = high if ((y + variant) % 16 in (0, 1)) else midpoint
        else:
            raise ValueError(f"unknown integer pattern: {pattern}")
        values.append(value)
    return values


def generate_float_tensor(size: int, pattern: str, rng: Pcg32, width: int,
                          variant: int = 0) -> list[float]:
    values: list[float] = []
    for i in range(size):
        x = i % width
        y = i // width
        if pattern == "zero":
            value = 0.0
        elif pattern == "minimum":
            value = -1.0
        elif pattern == "maximum":
            value = 1.0
        elif pattern == "gradient":
            value = ((x * 256 // max(1, width - 1)) - 128) / 128.0
        elif pattern == "checker":
            value = 1.0 if ((x // (1 + variant % 4)) + y) % 2 else -1.0
        elif pattern == "impulse":
            value = 1.0 if i == (size // 2 + variant * 31) % size else 0.0
        elif pattern == "low_random":
            value = rng.integer(-8, 8) / 128.0
        elif pattern == "dense_random":
            value = rng.integer(-128, 127) / 128.0
        elif pattern == "band_spike":
            center = (variant * 17 + 23) % width
            value = 1.0 if abs(x - center) <= 2 else rng.integer(-4, 4) / 128.0
        elif pattern == "boundary":
            value = (-1.0, 1.0, 0.0, 1.0)[(i + variant) % 4]
        else:
            raise ValueError(f"unknown float pattern: {pattern}")
        values.append(value)
    return values


def generate_fixed_workload(records: list[dict], *, name: str, workload_id: int,
                            dtype: str, shape: list[int], width: int, low: int = 0,
                            high: int = 255, channels: int = 1) -> None:
    size = 1
    for dim in shape:
        size *= dim
    smoke_patterns = ["zero", "minimum", "maximum", "gradient", "checker",
                      "impulse", "low_random", "dense_random"]
    stress_patterns = ["low_random", "dense_random", "stripes", "boundary"]

    for index, pattern in enumerate(smoke_patterns):
        seed = seed_for(workload_id, 0, index, index)
        rng = Pcg32(seed)
        if dtype == "float32":
            data = pack_f32(generate_float_tensor(size, pattern, rng, width, index))
        else:
            data = pack_i8(generate_integer_tensor(size, pattern, rng, low, high, width,
                                                    channels, index))
        write_sample(records, f"{name}/smoke/{index:02d}_{pattern}.bin", data,
                     workload=name, group="smoke", pattern=pattern, seed=seed,
                     dtype=dtype, shape=shape)

    for pattern_id, pattern in enumerate(stress_patterns):
        for variant in range(8):
            index = pattern_id * 8 + variant
            seed = seed_for(workload_id, 1, pattern_id, variant)
            rng = Pcg32(seed)
            if dtype == "float32":
                mapped = "band_spike" if pattern == "stripes" else pattern
                data = pack_f32(generate_float_tensor(size, mapped, rng, width, variant))
                recorded_pattern = mapped
            else:
                data = pack_i8(generate_integer_tensor(size, pattern, rng, low, high,
                                                        width, channels, variant))
                recorded_pattern = pattern
            write_sample(records,
                         f"{name}/stress/{recorded_pattern}/{index:02d}_{variant:02d}.bin",
                         data, workload=name, group="stress", pattern=recorded_pattern,
                         seed=seed, dtype=dtype, shape=shape)


def generate_sww(records: list[dict]) -> None:
    name = "sww01"
    workload_id = 4
    window_shape = [30, 40]
    window_size = 1200
    smoke_patterns = ["zero", "minimum", "maximum", "gradient", "checker",
                      "impulse", "low_random", "dense_random"]
    for index, pattern in enumerate(smoke_patterns):
        seed = seed_for(workload_id, 0, index, index)
        data = pack_i8(generate_integer_tensor(window_size, pattern, Pcg32(seed),
                                                -128, 127, 40, 1, index))
        write_sample(records, f"{name}/smoke/{index:02d}_{pattern}.bin", data,
                     workload=name, group="smoke", pattern=pattern, seed=seed,
                     dtype="int8", shape=window_shape)

    frames: list[int] = []
    segment_patterns = ["low_random", "banded", "pulsed", "dense_random"]
    segment_seeds: list[str] = []
    for segment, pattern in enumerate(segment_patterns):
        seed = seed_for(workload_id, 2, segment, 0)
        segment_seeds.append(f"0x{seed:016x}")
        frames.extend(generate_integer_tensor(64 * 40, pattern, Pcg32(seed),
                                              -128, 127, 40, 1, segment))
    stream_data = pack_i8(frames)
    stream_path = f"{name}/stress/stream_features_256x40_int8.bin"
    write_sample(records, stream_path, stream_data, workload=name, group="stress_stream",
                 pattern="four_64_frame_segments", seed=MASTER_SEED, dtype="int8",
                 shape=[256, 40], source={"segments": segment_patterns,
                                          "segment_seeds": segment_seeds})

    max_start = 256 - 30
    for index in range(32):
        start = (index * max_start + 15) // 31
        values = frames[start * 40:(start + 30) * 40]
        data = pack_i8(values)
        write_sample(records, f"{name}/stress/windows/{index:02d}_frame_{start:03d}.bin",
                     data, workload=name, group="stress", pattern="stream_window",
                     seed=MASTER_SEED, dtype="int8", shape=window_shape,
                     source={"stream": stream_path, "start_frame": start,
                             "frame_count": 30})


def build_manifest(records: list[dict]) -> dict:
    workloads = {}
    for name in ("ic01", "kws01", "ad01", "sww01"):
        selected = [record for record in records if record["workload"] == name]
        workloads[name] = {
            "file_count": len(selected),
            "groups": {
                group: sum(1 for record in selected if record["group"] == group)
                for group in sorted({record["group"] for record in selected})
            },
        }
    return {
        "schema_version": 1,
        "generator_version": VERSION,
        "generator": "PCG-XSH-RR 64/32 with explicit integer mapping",
        "master_seed": f"0x{MASTER_SEED:016x}",
        "dataset_kind": "synthetic",
        "source_dataset": None,
        "derived_from_official_samples": False,
        "design_basis": "input shapes and dtypes in this repository plus AVS stress patterns",
        "purpose": "deterministic synthetic inputs for NPU AVS/DVFS testing",
        "semantic_accuracy_dataset": False,
        "workloads": workloads,
        "files": sorted(records, key=lambda item: item["path"]),
    }


def write_manifests(records: list[dict]) -> None:
    manifest = build_manifest(records)
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in manifest["workloads"]:
        items = [item for item in manifest["files"] if item["workload"] == name]
        sub = {
            "schema_version": 1,
            "generator_version": VERSION,
            "master_seed": manifest["master_seed"],
            "dataset_kind": manifest["dataset_kind"],
            "source_dataset": manifest["source_dataset"],
            "derived_from_official_samples": manifest["derived_from_official_samples"],
            "design_basis": manifest["design_basis"],
            "workload": name,
            "summary": manifest["workloads"][name],
            "files": items,
        }
        (ROOT / name / "manifest.json").write_text(
            json.dumps(sub, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate() -> list[dict]:
    records: list[dict] = []
    generate_fixed_workload(records, name="ic01", workload_id=1, dtype="uint8",
                            shape=[32, 32, 3], width=32, low=0, high=255, channels=3)
    generate_fixed_workload(records, name="kws01", workload_id=2, dtype="int8",
                            shape=[49, 10, 1], width=10, low=-128, high=127)
    generate_fixed_workload(records, name="ad01", workload_id=3, dtype="float32",
                            shape=[5, 128], width=128)
    generate_sww(records)
    write_manifests(records)
    return records


def verify() -> None:
    manifest_path = ROOT / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("input/manifest.json does not exist; run generator first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.exists():
            failures.append(f"missing: {item['path']}")
            continue
        data = path.read_bytes()
        if len(data) != item["bytes"]:
            failures.append(f"size: {item['path']}")
        if sha256(data) != item["sha256"]:
            failures.append(f"sha256: {item['path']}")
    if failures:
        raise SystemExit("verification failed:\n" + "\n".join(failures))
    print(f"verified {len(manifest['files'])} generated input files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="verify existing files against input/manifest.json")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    records = generate()
    print(f"generated {len(records)} input files under {ROOT}")
    verify()


if __name__ == "__main__":
    main()
