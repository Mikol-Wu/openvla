"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
import csv
import json
import time
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union, Dict, Any

import draccus
import yaml
import torch
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.chunking import (
    ActionChunkEnsembler,
    ensure_action_flow,
    get_action_from_chunk,
    should_replan,
)
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action_flow,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class EvalConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 25                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # Decoder / evaluation extras
    decoder_type: str = "diffusion"                  # ["ar", "diffusion"]
    diffusion_steps: int = 5                         # used only when decoder_type=="diffusion"
    diffusion_mask_schedule: str = "cosine"         # ["linear", "cosine"]
    action_chunk_size: Optional[int] = None          # if None, read from checkpoint config and fall back to 1
    chunk_replan_interval: int = 1                   # closed-loop replanning interval in env steps
    use_chunk_ensembling: bool = True                # fuse overlapping chunk predictions when replanning
    chunk_ensemble_decay: float = 0.6                # temporal decay for older chunk votes
    action_tokenizer: str = "default"                # placeholder for future tokenizer switch
    train_token_masking: bool = False                # whether checkpoint used action token masking during training

    #################################################################################################################
    # Robustness eval (optional perturbations)
    #################################################################################################################
    robustness_eval: bool = False                    # Run clean + perturbed evals on the same suite
    observation_noise_sigma: float = 0.0             # Gaussian noise sigma (pixel space, 0-255)
    brightness_jitter: float = 0.0                   # Max brightness delta as fraction of 255 (0-1)
    contrast_jitter: float = 0.0                     # Max contrast delta (0-1), sampled around 1.0
    brightness_jitter_prob: float = 1.0              # Probability to apply brightness/contrast per frame
    instruction_dropout_prob: float = 0.0            # Probability to drop/mask each token
    instruction_dropout_mode: str = "drop"           # ["drop", "mask"]
    instruction_dropout_mask_token: str = "[MASK]"   # Token used when instruction_dropout_mode == "mask"

    # fmt: on

    @staticmethod
    def from_yaml(path: Optional[Union[str, Path]]) -> Optional["EvalConfig"]:
        if path is None:
            return None
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return EvalConfig(**data)

    # fmt: on


def _get_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parents[2])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _warn_env():
    missing = []
    for key in ["MUJOCO_EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES"]:
        if os.environ.get(key) is None:
            missing.append(key)
    if missing:
        print(f"[warn] Missing env vars: {missing}. Set them to select GPU/renderer.")


def _count_params(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _apply_observation_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return img
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _apply_brightness_contrast(
    img: np.ndarray, brightness: float, contrast: float, prob: float, rng: np.random.Generator
) -> np.ndarray:
    if (brightness <= 0 and contrast <= 0) or prob <= 0:
        return img
    if rng.random() > prob:
        return img
    out = img.astype(np.float32)
    if contrast > 0:
        contrast_factor = rng.uniform(1.0 - contrast, 1.0 + contrast)
        out = (out - 127.5) * contrast_factor + 127.5
    if brightness > 0:
        brightness_delta = rng.uniform(-brightness, brightness) * 255.0
        out = out + brightness_delta
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _apply_instruction_dropout(
    text: str, prob: float, mode: str, mask_token: str, rng: np.random.Generator
) -> str:
    if prob <= 0:
        return text
    tokens = text.split()
    if not tokens:
        return text
    if mode not in {"drop", "mask"}:
        raise ValueError(f"Unsupported instruction_dropout_mode: {mode}")
    if mode == "mask":
        dropped = 0
        out_tokens = []
        for tok in tokens:
            if rng.random() < prob:
                out_tokens.append(mask_token)
                dropped += 1
            else:
                out_tokens.append(tok)
        if dropped == len(tokens):
            idx = int(rng.integers(len(tokens)))
            out_tokens[idx] = tokens[idx]
        return " ".join(out_tokens)

    kept = [tok for tok in tokens if rng.random() >= prob]
    if not kept:
        kept = [tokens[int(rng.integers(len(tokens)))]]
    return " ".join(kept)


@draccus.wrap()
def eval_libero(cfg: EvalConfig, config_yaml: Optional[str] = None) -> None:
    if config_yaml:
        loaded = EvalConfig.from_yaml(config_yaml)
        if loaded is not None:
            cfg = loaded
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    _warn_env()

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    if cfg.action_chunk_size is None:
        cfg.action_chunk_size = int(getattr(getattr(model, "config", None), "action_chunk_size", 1) or 1)
    if cfg.action_chunk_size < 1:
        raise ValueError(f"action_chunk_size must be >= 1, got {cfg.action_chunk_size}")
    if cfg.chunk_replan_interval < 1:
        raise ValueError(f"chunk_replan_interval must be >= 1, got {cfg.chunk_replan_interval}")

    # Initialize local logging
    run_id = (
        f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{cfg.decoder_type}"
        f"-h{cfg.action_chunk_size}-rp{cfg.chunk_replan_interval}-{DATE_TIME}"
    )
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    if cfg.robustness_eval:
        run_id += "--robust"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Log model size
    param_stats = _count_params(model)
    log_file.write(f"Model params (total/trainable): {param_stats['total']}/{param_stats['trainable']}\n")
    print(f"Model params (total/trainable): {param_stats['total']}/{param_stats['trainable']}")

    control_mode = (
        f"Control: chunk={cfg.action_chunk_size}, replan={cfg.chunk_replan_interval}, "
        f"ensemble={cfg.use_chunk_ensembling and cfg.action_chunk_size > 1}, decay={cfg.chunk_ensemble_decay}"
    )
    print(control_mode)
    log_file.write(control_mode + "\n")

    # Start evaluation
    def _run_eval_pass(pass_name: str, apply_perturbations: bool) -> Dict[str, Any]:
        print(f"\n=== {pass_name} ===")
        log_file.write(f"\n=== {pass_name} ===\n")
        log_file.flush()

        rng = np.random.default_rng(cfg.seed) if apply_perturbations else None
        total_episodes, total_successes = 0, 0
        per_task_success: Dict[str, float] = {}
        per_step_times: list[float] = []
        per_episode_times: list[float] = []

        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                task_description_eval = task_description
                if apply_perturbations and cfg.instruction_dropout_prob > 0:
                    task_description_eval = _apply_instruction_dropout(
                        task_description,
                        cfg.instruction_dropout_prob,
                        cfg.instruction_dropout_mode,
                        cfg.instruction_dropout_mask_token,
                        rng,
                    )

                print(f"\nTask: {task_description}")
                log_file.write(f"\nTask: {task_description}\n")
                if apply_perturbations and task_description_eval != task_description:
                    print(f"Task (perturbed): {task_description_eval}")
                    log_file.write(f"Task (perturbed): {task_description_eval}\n")

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                policy_t = 0
                replay_images = []
                episode_start = time.perf_counter()
                latest_action_flow = None
                latest_chunk_start = None
                chunk_ensembler = (
                    ActionChunkEnsembler(decay=cfg.chunk_ensemble_decay)
                    if cfg.use_chunk_ensembling and cfg.action_chunk_size > 1
                    else None
                )
                if cfg.task_suite_name == "libero_spatial":
                    max_steps = 220  # longest training demo has 193 steps
                elif cfg.task_suite_name == "libero_object":
                    max_steps = 280  # longest training demo has 254 steps
                elif cfg.task_suite_name == "libero_goal":
                    max_steps = 300  # longest training demo has 270 steps
                elif cfg.task_suite_name == "libero_10":
                    max_steps = 520  # longest training demo has 505 steps
                elif cfg.task_suite_name == "libero_90":
                    max_steps = 400  # longest training demo has 373 steps

                print(f"Starting episode {task_episodes+1}...")
                log_file.write(f"Starting episode {task_episodes+1}...\n")
                done = False
                while t < max_steps + cfg.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue

                        # Get preprocessed image
                        img = get_libero_image(obs, resize_size)

                        # Apply perturbations to image if requested
                        if apply_perturbations and rng is not None:
                            img = _apply_brightness_contrast(
                                img,
                                cfg.brightness_jitter,
                                cfg.contrast_jitter,
                                cfg.brightness_jitter_prob,
                                rng,
                            )
                            img = _apply_observation_noise(img, cfg.observation_noise_sigma, rng)

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        # Prepare observations dict
                        # Note: OpenVLA does not take proprio state as input
                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }

                        # Query model to get the latest action flow and execute the current receding-horizon step
                        t0 = time.perf_counter()
                        needs_replan = latest_action_flow is None or should_replan(policy_t, cfg.chunk_replan_interval)
                        if latest_action_flow is not None and latest_chunk_start is not None:
                            if policy_t - latest_chunk_start >= latest_action_flow.shape[0]:
                                needs_replan = True

                        if needs_replan:
                            latest_action_flow = ensure_action_flow(
                                get_action_flow(
                                    cfg,
                                    model,
                                    observation,
                                    task_description_eval,
                                    processor=processor,
                                )
                            )
                            latest_chunk_start = policy_t
                            if chunk_ensembler is not None:
                                chunk_ensembler.add(policy_t, latest_action_flow)

                        if latest_action_flow is None or latest_chunk_start is None:
                            raise RuntimeError("No action flow available for execution")

                        if chunk_ensembler is not None:
                            action = chunk_ensembler.get_action(policy_t)
                        else:
                            action = get_action_from_chunk(latest_action_flow, latest_chunk_start, policy_t)
                        per_step_times.append(time.perf_counter() - t0)

                        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                        action = normalize_gripper_action(action, binarize=True)

                        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        t += 1
                        policy_t += 1
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break

                    except Exception as e:
                        print(f"Caught exception: {e}")
                        log_file.write(f"Caught exception: {e}\n")
                        break

                task_episodes += 1
                total_episodes += 1
                per_episode_times.append(time.perf_counter() - episode_start)

                # Save a replay video of the episode
                save_rollout_video(
                    replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
                )

                # Log current results
                print(f"Success: {done}")
                print(f"# episodes completed so far: {total_episodes}")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                log_file.write(f"Success: {done}\n")
                log_file.write(f"# episodes completed so far: {total_episodes}\n")
                log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                log_file.flush()

            # Log final results
            task_rate = float(task_successes) / float(task_episodes)
            per_task_success[task_description] = task_rate
            print(f"Current task success rate: {task_rate}")
            print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
            log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
            log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
            log_file.flush()
            if cfg.use_wandb:
                prefix = "robust" if apply_perturbations else "clean"
                wandb.log(
                    {
                        f"{prefix}/success_rate/{task_description}": float(task_successes) / float(task_episodes),
                        f"{prefix}/num_episodes/{task_description}": task_episodes,
                    }
                )

        return {
            "success_rate_total": float(total_successes) / float(total_episodes),
            "per_task_success": per_task_success,
            "num_episodes": total_episodes,
            "avg_step_time_sec": float(np.mean(per_step_times)) if per_step_times else None,
            "avg_episode_time_sec": float(np.mean(per_episode_times)) if per_episode_times else None,
        }

    clean_results = _run_eval_pass("clean", apply_perturbations=False)
    robust_results = None
    clean_sr = clean_results["success_rate_total"]
    robust_sr = None
    drop_sr = None
    if cfg.robustness_eval:
        robust_results = _run_eval_pass("robust", apply_perturbations=True)
        robust_sr = robust_results["success_rate_total"]
        drop_sr = clean_sr - robust_sr
        print(f"\nClean SR: {clean_sr:.4f} | Robust SR: {robust_sr:.4f} | Drop: {drop_sr:.4f}")
        log_file.write(f"\nClean SR: {clean_sr:.4f} | Robust SR: {robust_sr:.4f} | Drop: {drop_sr:.4f}\n")
        log_file.flush()

    # Save local log file
    log_file.close()

    # Persist structured results (json + csv)
    results_dir = Path(cfg.local_log_dir)
    robustness_config = {
        "observation_noise_sigma": cfg.observation_noise_sigma,
        "brightness_jitter": cfg.brightness_jitter,
        "contrast_jitter": cfg.contrast_jitter,
        "brightness_jitter_prob": cfg.brightness_jitter_prob,
        "instruction_dropout_prob": cfg.instruction_dropout_prob,
        "instruction_dropout_mode": cfg.instruction_dropout_mode,
        "instruction_dropout_mask_token": cfg.instruction_dropout_mask_token,
    }
    results = {
        "run_id": run_id,
        "task_suite": cfg.task_suite_name,
        "success_rate_total": clean_sr,
        "clean_success_rate_total": clean_sr,
        "per_task_success": clean_results["per_task_success"],
        "clean_per_task_success": clean_results["per_task_success"],
        "num_episodes": clean_results["num_episodes"],
        "seed": cfg.seed,
        "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
        "commit_hash": _get_commit_hash(),
        "decoder_type": cfg.decoder_type,
        "diffusion_steps": cfg.diffusion_steps,
        "diffusion_mask_schedule": cfg.diffusion_mask_schedule,
        "mask_schedule": cfg.diffusion_mask_schedule,
        "action_chunk_size": cfg.action_chunk_size,
        "chunk_replan_interval": cfg.chunk_replan_interval,
        "use_chunk_ensembling": cfg.use_chunk_ensembling,
        "chunk_ensemble_decay": cfg.chunk_ensemble_decay,
        "policy_mode": "rhaf" if cfg.action_chunk_size > 1 else "single_step",
        "action_tokenizer": cfg.action_tokenizer,
        "train_token_masking": cfg.train_token_masking,
        "robustness_eval": cfg.robustness_eval,
        "robustness_config": robustness_config,
        "model_params_total": param_stats["total"],
        "model_params_trainable": param_stats["trainable"],
        "avg_step_time_sec": clean_results["avg_step_time_sec"],
        "avg_episode_time_sec": clean_results["avg_episode_time_sec"],
    }
    if cfg.robustness_eval and robust_results is not None:
        results.update(
            {
                "clean_SR": clean_sr,
                "robust_SR": robust_sr,
                "drop": drop_sr,
                "clean_success_rate_total": clean_sr,
                "robust_success_rate_total": robust_sr,
                "success_rate_drop": drop_sr,
                "clean_per_task_success": clean_results["per_task_success"],
                "robust_per_task_success": robust_results["per_task_success"],
                "robust_num_episodes": robust_results["num_episodes"],
                "robust_avg_step_time_sec": robust_results["avg_step_time_sec"],
                "robust_avg_episode_time_sec": robust_results["avg_episode_time_sec"],
            }
        )
    json_path = results_dir / f"{run_id}.json"
    csv_path = results_dir / f"{run_id}.csv"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "success_rate"])
        for k, v in clean_results["per_task_success"].items():
            writer.writerow([k, v])
        writer.writerow(["total_success_rate", results["success_rate_total"]])
    if cfg.robustness_eval and robust_results is not None:
        robust_csv_path = results_dir / f"{run_id}_robust.csv"
        with open(robust_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "success_rate"])
            for k, v in robust_results["per_task_success"].items():
                writer.writerow([k, v])
            writer.writerow(["total_success_rate", robust_sr])
        summary_csv_path = results_dir / f"{run_id}_summary.csv"
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["clean_SR", clean_sr])
            writer.writerow(["robust_SR", robust_sr])
            writer.writerow(["drop", drop_sr])

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total_clean": clean_sr,
                "num_episodes/total_clean": clean_results["num_episodes"],
            }
        )
        if cfg.robustness_eval and robust_results is not None:
            wandb.log(
                {
                    "success_rate/total_robust": robust_sr,
                    "num_episodes/total_robust": robust_results["num_episodes"],
                    "success_rate/drop": drop_sr,
                }
            )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
