"""
run_eval_matrix.py

Automate a LIBERO evaluation matrix (clean + robustness) and summarize results.
"""

from __future__ import annotations

import argparse
import json
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"on", "true", "1", "yes"}:
            return True
        if v in {"off", "false", "0", "no"}:
            return False
    raise ValueError(f"Unsupported boolean value: {value}")


def _find_result_json(log_dir: Path, note: str, after_ts: Optional[float]) -> Optional[Path]:
    pattern = f"*--{note}*.json"
    candidates = [p for p in log_dir.glob(pattern)]
    if after_ts is not None:
        candidates = [p for p in candidates if p.stat().st_mtime >= after_ts - 1.0]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def _format_mean_std(m: Optional[float], s: Optional[float]) -> str:
    if m is None:
        return "n/a"
    if s is None:
        return f"{m:.4f}"
    return f"{m:.4f} +/- {s:.4f}"


def build_runs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    suite = cfg["suite"]
    seeds = cfg["seeds"]
    model_family = cfg.get("model_family", "openvla")
    center_crop = bool(cfg.get("center_crop", True))
    num_trials = int(cfg.get("num_trials_per_task", 50))
    local_log_dir = cfg.get("local_log_dir", "./experiments/logs")
    run_id_prefix = cfg.get("run_id_prefix", "matrix")
    robustness_eval = bool(cfg.get("robustness_eval", True))
    robustness = cfg.get("robustness", {})

    checkpoints = cfg.get("pretrained_checkpoint", {})
    baseline_ckpt = checkpoints.get("baseline_ar")
    diffusion_ckpt_on = checkpoints.get("diffusion", {}).get("masking_on")
    diffusion_ckpt_off = checkpoints.get("diffusion", {}).get("masking_off")

    if baseline_ckpt is None:
        raise ValueError("pretrained_checkpoint.baseline_ar must be set")
    if diffusion_ckpt_on is None or diffusion_ckpt_off is None:
        raise ValueError("pretrained_checkpoint.diffusion.masking_on/off must be set")

    diffusion_steps = cfg.get("matrix", {}).get("diffusion_steps", [1, 5, 10])
    mask_schedules = cfg.get("matrix", {}).get("mask_schedules", ["linear", "cosine"])
    token_masking = cfg.get("matrix", {}).get("train_token_masking", ["on", "off"])

    runs: List[Dict[str, Any]] = []

    # Baseline AR
    for seed in seeds:
        note = _slugify(f"{run_id_prefix}-baseline-ar-seed{seed}")
        runs.append(
            dict(
                note=note,
                method="baseline_ar",
                suite=suite,
                seed=seed,
                model_family=model_family,
                center_crop=center_crop,
                num_trials_per_task=num_trials,
                local_log_dir=local_log_dir,
                pretrained_checkpoint=baseline_ckpt,
                decoder_type="ar",
                diffusion_steps=10,
                diffusion_mask_schedule="linear",
                train_token_masking=False,
                robustness_eval=robustness_eval,
                robustness=robustness,
            )
        )

    # Diffusion variants
    for k in diffusion_steps:
        for sched in mask_schedules:
            for tm in token_masking:
                tm_bool = _parse_bool(tm)
                ckpt = diffusion_ckpt_on if tm_bool else diffusion_ckpt_off
                tm_tag = "on" if tm_bool else "off"
                for seed in seeds:
                    note = _slugify(f"{run_id_prefix}-diff-k{k}-{sched}-mask{tm_tag}-seed{seed}")
                    runs.append(
                        dict(
                            note=note,
                            method=f"diffusion_k{k}",
                            suite=suite,
                            seed=seed,
                            model_family=model_family,
                            center_crop=center_crop,
                            num_trials_per_task=num_trials,
                            local_log_dir=local_log_dir,
                            pretrained_checkpoint=ckpt,
                            decoder_type="diffusion",
                            diffusion_steps=int(k),
                            diffusion_mask_schedule=str(sched),
                            train_token_masking=tm_bool,
                            robustness_eval=robustness_eval,
                            robustness=robustness,
                        )
                    )

    return runs


def _write_config_yaml(out_dir: Path, run_cfg: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        model_family=run_cfg["model_family"],
        pretrained_checkpoint=run_cfg["pretrained_checkpoint"],
        task_suite_name=run_cfg["suite"],
        center_crop=run_cfg["center_crop"],
        num_trials_per_task=run_cfg["num_trials_per_task"],
        local_log_dir=run_cfg["local_log_dir"],
        seed=run_cfg["seed"],
        decoder_type=run_cfg["decoder_type"],
        diffusion_steps=run_cfg["diffusion_steps"],
        diffusion_mask_schedule=run_cfg["diffusion_mask_schedule"],
        train_token_masking=run_cfg["train_token_masking"],
        robustness_eval=run_cfg["robustness_eval"],
        run_id_note=run_cfg["note"],
    )
    robustness = run_cfg.get("robustness", {})
    cfg.update(
        dict(
            observation_noise_sigma=float(robustness.get("observation_noise_sigma", 0.0)),
            brightness_jitter=float(robustness.get("brightness_jitter", 0.0)),
            contrast_jitter=float(robustness.get("contrast_jitter", 0.0)),
            brightness_jitter_prob=float(robustness.get("brightness_jitter_prob", 1.0)),
            instruction_dropout_prob=float(robustness.get("instruction_dropout_prob", 0.0)),
            instruction_dropout_mode=str(robustness.get("instruction_dropout_mode", "drop")),
            instruction_dropout_mask_token=str(robustness.get("instruction_dropout_mask_token", "[MASK]")),
        )
    )
    yaml_path = out_dir / f"{run_cfg['note']}.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return yaml_path


def _run_eval(run_cfg: Dict[str, Any], repo_root: Path, config_dir: Path, runs_log: Path) -> Optional[Path]:
    log_dir = Path(run_cfg["local_log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = _write_config_yaml(config_dir, run_cfg)
    cmd = [
        sys.executable,
        "experiments/robot/libero/run_libero_eval.py",
        "--config_yaml",
        str(yaml_path),
    ]
    start_ts = time.time()
    result_json = None
    try:
        subprocess.run(cmd, check=True, cwd=repo_root)
        result_json = _find_result_json(log_dir, run_cfg["note"], after_ts=start_ts)
    finally:
        record = dict(
            note=run_cfg["note"],
            command=" ".join(cmd),
            config_yaml=str(yaml_path),
            result_json=str(result_json) if result_json else None,
            seed=run_cfg["seed"],
            decoder_type=run_cfg["decoder_type"],
            diffusion_steps=run_cfg["diffusion_steps"],
            diffusion_mask_schedule=run_cfg["diffusion_mask_schedule"],
            train_token_masking=run_cfg["train_token_masking"],
            task_suite=run_cfg["suite"],
        )
        with open(runs_log, "a") as f:
            f.write(json.dumps(record) + "\n")
    return result_json


def summarize_results(
    log_dir: Path,
    run_id_prefix: str,
    suite: Optional[str],
    out_csv: Path,
    out_md: Path,
    runs_log: Optional[Path],
    robustness: Dict[str, Any],
) -> None:
    results: List[Dict[str, Any]] = []
    for json_path in log_dir.glob("*.json"):
        with open(json_path, "r") as f:
            data = json.load(f)
        run_id = data.get("run_id", "")
        if run_id_prefix and f"--{run_id_prefix}-" not in run_id:
            continue
        if suite and data.get("task_suite") != suite:
            continue
        results.append(data)

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in results:
        decoder = r.get("decoder_type", "ar")
        steps = r.get("diffusion_steps", 10) if decoder == "diffusion" else "n/a"
        sched = r.get("diffusion_mask_schedule", "linear") if decoder == "diffusion" else "n/a"
        train_mask = r.get("train_token_masking", False) if decoder == "diffusion" else "n/a"
        key = (
            decoder,
            steps,
            sched,
            train_mask,
            r.get("task_suite"),
            r.get("pretrained_checkpoint"),
        )
        grouped.setdefault(key, []).append(r)

    rows: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        decoder, steps, sched, train_mask, task_suite, ckpt = key
        clean_vals = []
        robust_vals = []
        drop_vals = []
        seeds = []
        for it in items:
            seeds.append(it.get("seed"))
            clean_vals.append(it.get("clean_SR", it.get("success_rate_total")))
            if it.get("robust_SR") is not None:
                robust_vals.append(it.get("robust_SR"))
            if it.get("drop") is not None:
                drop_vals.append(it.get("drop"))
        clean_m, clean_s = _mean_std([v for v in clean_vals if v is not None])
        robust_m, robust_s = _mean_std([v for v in robust_vals if v is not None])
        drop_m, drop_s = _mean_std([v for v in drop_vals if v is not None])
        rows.append(
            dict(
                method="baseline_ar" if decoder == "ar" else f"diffusion_k{steps}",
                decoder_type=decoder,
                diffusion_steps=steps,
                mask_schedule=sched,
                train_token_masking=train_mask,
                task_suite=task_suite,
                pretrained_checkpoint=ckpt,
                clean_sr_mean=clean_m,
                clean_sr_std=clean_s,
                robust_sr_mean=robust_m,
                robust_sr_std=robust_s,
                drop_mean=drop_m,
                drop_std=drop_s,
                seeds=",".join(str(s) for s in sorted(set(seeds))),
                num_seeds=len(set(seeds)),
            )
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        header = [
            "method",
            "decoder_type",
            "diffusion_steps",
            "mask_schedule",
            "train_token_masking",
            "task_suite",
            "pretrained_checkpoint",
            "clean_sr_mean",
            "clean_sr_std",
            "robust_sr_mean",
            "robust_sr_std",
            "drop_mean",
            "drop_std",
            "seeds",
            "num_seeds",
        ]
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r[h] if r[h] is not None else "" for h in header])

    # Markdown report
    lines: List[str] = []
    lines.append("# LIBERO Eval Matrix Report")
    if suite:
        lines.append(f"Suite: {suite}")
    lines.append("")
    lines.append("**Robustness Config**")
    if robustness:
        for k in sorted(robustness.keys()):
            lines.append(f"- {k}: {robustness[k]}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("**Summary Table**")
    lines.append("| Method | Mask schedule | Train token masking | Clean SR (mean +/- std) | Robust SR (mean +/- std) | Drop (mean +/- std) | Seeds |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(rows, key=lambda x: (x["decoder_type"], str(x["diffusion_steps"]), str(x["mask_schedule"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    r["method"],
                    str(r["mask_schedule"]),
                    str(r["train_token_masking"]),
                    _format_mean_std(r["clean_sr_mean"], r["clean_sr_std"]),
                    _format_mean_std(r["robust_sr_mean"], r["robust_sr_std"]),
                    _format_mean_std(r["drop_mean"], r["drop_std"]),
                    r["seeds"],
                ]
            )
            + " |"
        )
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.append("")
    lines.append("**Conclusions**")
    if rows:
        best_clean = max(rows, key=lambda x: (x["clean_sr_mean"] or -1))
        lines.append(
            f"- Best clean SR: {best_clean['method']} (schedule={best_clean['mask_schedule']}, mask={best_clean['train_token_masking']})"
        )
        robust_candidates = [r for r in rows if r["robust_sr_mean"] is not None]
        if robust_candidates:
            best_robust = max(robust_candidates, key=lambda x: (x["robust_sr_mean"] or -1))
            lines.append(
                f"- Best robust SR: {best_robust['method']} (schedule={best_robust['mask_schedule']}, mask={best_robust['train_token_masking']})"
            )
        else:
            lines.append("- No robust SR values found.")
    else:
        lines.append("- No completed runs found for this prefix and suite.")

    lines.append("")
    lines.append("**Reproducibility**")
    lines.append("- Run full matrix: `python experiments/robot/libero/run_eval_matrix.py --config experiments/robot/libero/exp_matrix.yaml --mode run`")
    lines.append("- Summarize existing logs: `python experiments/robot/libero/run_eval_matrix.py --config experiments/robot/libero/exp_matrix.yaml --mode summarize`")
    if runs_log:
        lines.append(f"- Per-run commands: {runs_log.as_posix()}")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/robot/libero/exp_matrix.yaml",
        help="Path to experiment matrix YAML",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "summarize", "run_and_summarize"],
        default="run_and_summarize",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--skip_existing", action="store_true", help="Skip runs with existing results")
    args = parser.parse_args()

    repo_root = Path(__file__).parents[2]
    cfg = _load_yaml(repo_root / args.config)

    log_dir = Path(cfg.get("local_log_dir", "./experiments/logs"))
    run_id_prefix = cfg.get("run_id_prefix", "matrix")
    suite = cfg.get("suite")
    out_dir = Path(cfg.get("output_dir", "./experiments/robot/libero/exp_results"))
    config_dir = out_dir / "generated_configs"
    runs_log = out_dir / "runs.jsonl"
    robustness = cfg.get("robustness", {})

    if args.mode in {"run", "run_and_summarize"}:
        runs = build_runs(cfg)
        for run_cfg in runs:
            existing = _find_result_json(log_dir, run_cfg["note"], after_ts=None)
            if args.skip_existing and existing is not None:
                continue
            if args.dry_run:
                yaml_path = _write_config_yaml(config_dir, run_cfg)
                cmd = [
                    sys.executable,
                    "experiments/robot/libero/run_libero_eval.py",
                    "--config_yaml",
                    str(yaml_path),
                ]
                print(" ".join(cmd))
                continue
            _run_eval(run_cfg, repo_root, config_dir, runs_log)

    if args.mode in {"summarize", "run_and_summarize"}:
        out_csv = out_dir / "summary.csv"
        out_md = out_dir / "report.md"
        summarize_results(
            log_dir=log_dir,
            run_id_prefix=run_id_prefix,
            suite=suite,
            out_csv=out_csv,
            out_md=out_md,
            runs_log=runs_log,
            robustness=robustness,
        )


if __name__ == "__main__":
    main()
