#!/usr/bin/env python3
"""
Simple launcher for the shared experiment runner.

Modes:
- Tk GUI (default) if X is available and Tk initializes.
- Text fallback (auto-used when Tk fails or when --text is passed).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
import os
import traceback

# Locate the shared runner
ROOT = Path(__file__).resolve().parents[2]  # scenario root, e.g. ADD/
RUNNER = ROOT / "_Shared" / "scripts" / "run_experiments.py"

ACTIONS = [
    "vkitti-extract",
    "vkitti-prepare",
    "vkitti-yaml",
    "vkitti-yolo-train",
    "vkitti-yolo-eval",
    "vkitti-frcnn-train",
    "vkitti2-extract",
    "vkitti2-prepare",
    "vkitti2-yaml",
    "vkitti2-yolo-train",
    "vkitti2-yolo-eval",
    "extract-bdd",
    "sanity-bdd",
    "convert-bdd",
    "make-splits",
    "materialize",
    "yolo-train",
    "yolo-eval",
    "frcnn-train",
    "frcnn-eval",
]

# Common dataset roots (more can be added later)
DATASET_OPTIONS = [
    "",  # blank = use defaults inside runner
    str(ROOT / "_Shared" / "data" / "ad" / "vkitti"),
    str(ROOT / "_Shared" / "data" / "ad" / "vkitti_yolo_splits"),
    str(ROOT / "_Shared" / "data" / "ad" / "vkitti_yolo_splits" / "vkitti_det.yaml"),
    str(ROOT / "_Shared" / "data" / "ad" / "vkitti2"),
    str(ROOT / "_Shared" / "data" / "ad" / "vkitti2_yolo_splits"),
    str(ROOT / "_Shared" / "data" / "ad" / "vkitti2_yolo_splits" / "vkitti2_det.yaml"),
    str(ROOT / "_Shared" / "data" / "ad" / "bdd_yolo_splits"),
    str(ROOT / "_Shared" / "data" / "ad" / "bdd100k_raw" / "bdd100k"),
]


def text_mode():
    print("Text mode launcher (Tk unavailable).")
    print("Available actions:")
    for i, act in enumerate(ACTIONS):
        print(f"  [{i}] {act}")
    try:
        idx = int(input("Select action by number: ").strip())
        action = ACTIONS[idx]
    except Exception:
        print("Invalid selection.")
        return

    seed = input("Seed (default 0): ").strip() or "0"
    n_images = input("n_images / subset (default 9000): ").strip() or "9000"
    device = input("Device (cuda/cpu, default cuda): ").strip() or "cuda"
    dataset_root = input("Dataset root (optional, blank=default): ").strip()

    cmd = [sys.executable, str(RUNNER), "--action", action, "--seed", seed]
    if action == "make-splits":
        cmd += ["--n-images", n_images]
    if action == "vkitti-prepare":
        cmd += ["--subset-size", n_images]
    if action == "vkitti-yaml":
        if dataset_root.endswith(".yaml"):
            cmd += ["--out-yaml", dataset_root]
            dataset_root = ""
        if dataset_root:
            cmd += ["--dataset-root", dataset_root]
        if action in {"vkitti-extract", "vkitti-prepare", "vkitti-yolo-train", "vkitti-yolo-eval", "vkitti-frcnn-train",
                      "vkitti2-extract", "vkitti2-prepare", "vkitti2-yolo-train", "vkitti2-yolo-eval"} and dataset_root:
            cmd += ["--dataset-root", dataset_root]
    if action in {"yolo-train", "yolo-eval", "frcnn-train", "frcnn-eval",
                  "vkitti-yolo-train", "vkitti-yolo-eval", "vkitti-frcnn-train",
                  "vkitti2-yolo-train", "vkitti2-yolo-eval"}:
        cmd += ["--device", device]

    print("\n$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
    rc = proc.wait()
    print(f"\n[exit code {rc}]")


class RunnerUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Experiment Launcher")
        self.geometry("640x480")

        # Controls frame
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="x")

        ttk.Label(frm, text="Action").grid(row=0, column=0, sticky="w")
        self.action_var = tk.StringVar(value=ACTIONS[0])
        self.action_combo = ttk.Combobox(frm, textvariable=self.action_var, values=ACTIONS, state="readonly", width=20)
        self.action_combo.grid(row=0, column=1, sticky="w")
        self.action_combo.bind("<<ComboboxSelected>>", self._on_action_change)

        ttk.Label(frm, text="Seed").grid(row=1, column=0, sticky="w")
        self.seed_var = tk.StringVar(value="0")
        ttk.Entry(frm, textvariable=self.seed_var, width=10).grid(row=1, column=1, sticky="w")

        ttk.Label(frm, text="n_images / subset").grid(row=2, column=0, sticky="w")
        self.n_images_var = tk.StringVar(value="9000")
        self.n_images_entry = ttk.Entry(frm, textvariable=self.n_images_var, width=10)
        self.n_images_entry.grid(row=2, column=1, sticky="w")

        ttk.Label(frm, text="Device").grid(row=3, column=0, sticky="w")
        self.device_var = tk.StringVar(value="cuda")
        self.device_combo = ttk.Combobox(frm, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly", width=10)
        self.device_combo.grid(row=3, column=1, sticky="w")

        ttk.Label(frm, text="Dataset root (optional)").grid(row=4, column=0, sticky="w")
        self.dataset_var = tk.StringVar(value=DATASET_OPTIONS[0])
        self.dataset_combo = ttk.Combobox(
            frm,
            textvariable=self.dataset_var,
            values=DATASET_OPTIONS,
            state="normal",  # allow typing custom paths
            width=50,
        )
        self.dataset_combo.grid(row=4, column=1, columnspan=2, sticky="w")

        self.run_btn = ttk.Button(frm, text="Run", command=self.run_action)
        self.run_btn.grid(row=0, column=3, padx=10)
        self.stop_btn = ttk.Button(frm, text="Stop", command=self.stop_action, state="disabled")
        self.stop_btn.grid(row=1, column=3, padx=10)

        # Output text
        self.text = tk.Text(self, wrap="word")
        self.text.pack(fill="both", expand=True, padx=10, pady=5)
        clear_btn = ttk.Button(self, text="Clear", command=self.clear_text)
        clear_btn.pack(pady=2)

        # State
        self.process: subprocess.Popen | None = None
        self._on_action_change()

    def _on_action_change(self, event=None):
        act = self.action_var.get()
        if act in {"make-splits", "vkitti-prepare"}:
            self.n_images_entry.configure(state="normal")
        else:
            self.n_images_entry.configure(state="disabled")

    def append(self, msg: str) -> None:
        self.text.insert("end", msg)
        self.text.see("end")

    def run_action(self) -> None:
        if self.process:
            return
        action = self.action_var.get()
        seed = self.seed_var.get().strip() or "0"
        n_images = self.n_images_var.get().strip() or "9000"
        device = self.device_var.get().strip() or "cuda"
        dataset_root = self.dataset_var.get().strip()
        out_yaml = ""  # only used for vkitti-yaml

        cmd = [sys.executable, str(RUNNER), "--action", action, "--seed", seed]
        if action == "make-splits":
            cmd += ["--n-images", n_images]
        if action == "vkitti-prepare":
            cmd += ["--subset-size", n_images]
        if action == "vkitti-yaml":
            # if user types a yaml path here, treat it as out-yaml instead of dataset-root
            if dataset_root.endswith(".yaml"):
                out_yaml = dataset_root
                cmd += ["--out-yaml", out_yaml]
                dataset_root = ""  # don't also pass as dataset-root
            if dataset_root:
                cmd += ["--dataset-root", dataset_root]
        if action in {"vkitti-extract", "vkitti-prepare"} and dataset_root:
            cmd += ["--dataset-root", dataset_root]
        cmd += ["--device", device]

        self.append(f"\n$ {' '.join(cmd)}\n")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        def target():
            try:
                self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.append(line)
            finally:
                rc = self.process.wait() if self.process else -1
                self.append(f"\n[exit code {rc}]\n")
                self.process = None
                self.run_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")

        threading.Thread(target=target, daemon=True).start()

    def stop_action(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.append("\n[terminated]\n")
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.process = None

    def clear_text(self) -> None:
        self.text.delete("1.0", "end")


def main() -> None:
    # Text mode forced
    if "--text" in sys.argv:
        sys.argv.remove("--text")
        return text_mode()

    # Try GUI; fallback to text on any X/Tk error
    try:
        app = RunnerUI()
        app.mainloop()
    except Exception as exc:
        traceback.print_exc()
        print("\n[Warn] GUI failed, falling back to text mode. You can force text mode with --text.\n")
        text_mode()


if __name__ == "__main__":
    main()
