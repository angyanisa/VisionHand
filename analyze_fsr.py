#!/usr/bin/env python3
"""
Offline FSR threshold analysis tool.

Reads a paired FSR log CSV (from fsr_logger.py or the fsr_publisher ROS node),
auto-detects pre-grasp / grabbing / hold / release phases, and writes a JSON
file with per-finger thresholds that EMG_to_nano loads at runtime.

FSR sensor mapping (adjust if wiring differs):
    fsr0 = thumb
    fsr1 = index
    fsr2 = middle
    fsr3 = ring
    fsr4 = pinky

Usage:
    python3 analyze_fsr.py <fsr_log.csv> --recording <name> [options]

Examples:
    python3 analyze_fsr.py fsr_log_20240428_120000.csv --recording bottle_body
    python3 analyze_fsr.py fsr_log_20240428_120000.csv --recording mug_handle --no-plot

Output:
    ros_ws/src/nano_hand/rokoko_csv/<recording>_fsr.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SENSOR_NAMES = ["fsr0", "fsr1", "fsr2", "fsr3", "fsr4"]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# --- Tunable parameters ---
BASELINE_DURATION_S = 0.5    # seconds at start used to compute resting baseline
SMOOTH_WINDOW_S = 0.05       # rolling-mean smoothing window (seconds)
CONTACT_STD_MULT = 4.0       # contact detected when value > baseline + N*std
CONTACT_MIN_ABS = 30.0       # minimum absolute rise above baseline to count as contact
CONTACT_SUSTAIN_S = 0.05     # must stay above threshold for this long to confirm contact
RELEASE_FRACTION = 0.4       # release detected when signal drops below N * peak
GRASP_WINDOW_FRACTION = 0.5  # fraction of hold period (from midpoint) used for grasp_force


def load_fsr(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"host_time_s", "fsr0", "fsr1", "fsr2", "fsr3", "fsr4"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: FSR log missing columns: {missing}")
    df = df.sort_values("host_time_s").reset_index(drop=True)
    df["t"] = df["host_time_s"] - df["host_time_s"].iloc[0]
    return df


def smooth(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series.copy()
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode="same")


def analyze_sensor(t: np.ndarray, raw: np.ndarray, fs: float) -> dict:
    """
    Detect phases and extract thresholds for a single sensor.
    Returns None contact/grasp values for sensors that never made contact.
    """
    win = max(1, int(SMOOTH_WINDOW_S * fs))
    sig = smooth(raw, win)

    baseline_end = int(BASELINE_DURATION_S * fs)
    baseline_end = min(baseline_end, len(sig) // 4)
    baseline_vals = sig[:baseline_end]
    baseline = float(np.mean(baseline_vals))
    baseline_std = float(np.std(baseline_vals))

    contact_thresh = baseline + max(CONTACT_MIN_ABS, CONTACT_STD_MULT * baseline_std)
    sustain_samples = max(1, int(CONTACT_SUSTAIN_S * fs))

    # Find first sustained contact
    above = sig > contact_thresh
    contact_sample = None
    for i in range(len(above) - sustain_samples):
        if np.all(above[i : i + sustain_samples]):
            contact_sample = i
            break

    if contact_sample is None:
        return {
            "baseline": round(baseline, 1),
            "contact_force": None,
            "grasp_force": None,
            "contact_sample": None,
            "hold_start_sample": None,
            "release_start_sample": None,
        }

    # Find peak after contact
    peak_sample = int(np.argmax(sig[contact_sample:])) + contact_sample
    peak_val = float(sig[peak_sample])

    # Find release: first sample after peak where signal drops below fraction of peak
    release_thresh = RELEASE_FRACTION * peak_val
    release_sample = len(sig) - 1  # default: never released in recording
    for i in range(peak_sample, len(sig)):
        if sig[i] < release_thresh:
            release_sample = i
            break

    # Grasp window: latter half of hold period (stable plateau before release)
    hold_len = release_sample - contact_sample
    hold_start = contact_sample + int(hold_len * (1 - GRASP_WINDOW_FRACTION))
    hold_end = release_sample
    if hold_start >= hold_end:
        hold_start = contact_sample

    grasp_force = float(np.mean(raw[hold_start:hold_end]))
    contact_force = float(sig[contact_sample])

    return {
        "baseline": round(baseline, 1),
        "contact_force": round(contact_force, 1),
        "grasp_force": round(grasp_force, 1),
        "contact_sample": int(contact_sample),
        "hold_start_sample": int(hold_start),
        "release_start_sample": int(release_sample),
    }


def plot_results(df: pd.DataFrame, results: dict, recording: str, output_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    t = df["t"].values
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"FSR phase analysis — {recording}", fontsize=13)

    for i, (sensor, finger) in enumerate(zip(SENSOR_NAMES, FINGER_NAMES)):
        ax = axes[i]
        raw = df[sensor].values
        ax.plot(t, raw, color="steelblue", lw=0.8, label="raw")
        ax.set_ylabel(f"{finger}\n(ADC)", fontsize=8)
        ax.grid(True, alpha=0.3)

        r = results[sensor]
        ax.axhline(r["baseline"], color="gray", ls="--", lw=0.8, label="baseline")
        if r["contact_force"] is not None:
            ax.axhline(r["contact_force"], color="orange", ls="--", lw=0.8, label="contact")
            ax.axvline(t[r["contact_sample"]], color="orange", lw=1.2, alpha=0.7)
        if r["grasp_force"] is not None:
            ax.axhline(r["grasp_force"], color="red", ls="--", lw=0.8, label="grasp target")
            ax.axvspan(
                t[r["hold_start_sample"]],
                t[min(r["release_start_sample"], len(t) - 1)],
                alpha=0.15, color="green", label="grasp window",
            )
            if r["release_start_sample"] < len(t):
                ax.axvline(t[r["release_start_sample"]], color="purple", lw=1.2, alpha=0.7)

        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{recording}_fsr_plot.png")
    fig.savefig(png_path, dpi=120)
    print(f"Plot saved to {png_path}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Analyze FSR recording and extract thresholds.")
    parser.add_argument("fsr_log", help="Path to FSR log CSV")
    parser.add_argument("--recording", required=True,
                        help="Name matching a key in csv_map (e.g. bottle_body)")
    parser.add_argument("--output-dir",
                        default="ros_ws/src/nano_hand/fsr_csv",
                        help="Directory to write the output JSON and plot PNG")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    args = parser.parse_args()

    if not os.path.exists(args.fsr_log):
        sys.exit(f"ERROR: FSR log not found: {args.fsr_log}")

    print(f"Loading {args.fsr_log}...")
    df = load_fsr(args.fsr_log)
    n = len(df)
    duration = df["t"].iloc[-1]
    dt = float(np.median(np.diff(df["t"].values)))
    fs = 1.0 / dt if dt > 0 else 100.0
    print(f"  {n} samples, {duration:.1f}s, ~{fs:.0f} Hz")

    results = {}
    print(f"\n{'Sensor':<8} {'Finger':<8} {'Baseline':>10} {'Contact':>10} {'Grasp target':>14}")
    print("-" * 56)
    for sensor, finger in zip(SENSOR_NAMES, FINGER_NAMES):
        r = analyze_sensor(df["t"].values, df[sensor].values.astype(float), fs)
        results[sensor] = r
        contact_str = f"{r['contact_force']:.0f}" if r["contact_force"] is not None else "—"
        grasp_str = f"{r['grasp_force']:.0f}" if r["grasp_force"] is not None else "—"
        print(f"{sensor:<8} {finger:<8} {r['baseline']:>10.0f} {contact_str:>10} {grasp_str:>14}")

    output = {
        "recording": args.recording,
        "fsr_log": os.path.basename(args.fsr_log),
        "sensors": {
            s: {
                "baseline": r["baseline"],
                "contact_force": r["contact_force"],
                "grasp_force": r["grasp_force"],
            }
            for s, r in results.items()
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.recording}_fsr.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved thresholds to {out_path}")

    if not args.no_plot:
        plot_results(df, results, args.recording, args.output_dir)


if __name__ == "__main__":
    main()
