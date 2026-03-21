"""Utilities for receding-horizon chunked action execution in LIBERO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ChunkPrediction:
    start_step: int
    actions: np.ndarray


@dataclass
class ActionChunkEnsembler:
    decay: float = 0.6
    predictions: List[ChunkPrediction] = field(default_factory=list)

    def add(self, start_step: int, actions: np.ndarray) -> None:
        action_flow = ensure_action_flow(actions)
        self.predictions.append(ChunkPrediction(start_step=start_step, actions=action_flow))
        self.prune(start_step)

    def prune(self, current_step: int) -> None:
        self.predictions = [
            pred for pred in self.predictions if current_step - pred.start_step < pred.actions.shape[0]
        ]

    def get_action(self, current_step: int) -> np.ndarray:
        self.prune(current_step)
        weighted_action = None
        total_weight = 0.0

        for pred in self.predictions:
            offset = current_step - pred.start_step
            if offset < 0 or offset >= pred.actions.shape[0]:
                continue
            weight = float(self.decay ** offset)
            action = pred.actions[offset]
            weighted_action = action * weight if weighted_action is None else weighted_action + action * weight
            total_weight += weight

        if weighted_action is None or total_weight <= 0.0:
            raise ValueError(f"No valid action chunk covers current_step={current_step}")

        return weighted_action / total_weight


def ensure_action_flow(actions: np.ndarray) -> np.ndarray:
    action_flow = np.asarray(actions, dtype=np.float32)
    if action_flow.ndim == 1:
        action_flow = action_flow[None, :]
    if action_flow.ndim != 2:
        raise ValueError(f"Expected action flow with shape [H, D], got {action_flow.shape}")
    return action_flow


def should_replan(current_step: int, replan_interval: int) -> bool:
    if replan_interval < 1:
        raise ValueError(f"replan_interval must be >= 1, got {replan_interval}")
    return current_step % replan_interval == 0


def get_action_from_chunk(action_flow: np.ndarray, chunk_start_step: int, current_step: int) -> np.ndarray:
    flow = ensure_action_flow(action_flow)
    offset = current_step - chunk_start_step
    if offset < 0 or offset >= flow.shape[0]:
        raise ValueError(
            f"Requested step {current_step} is outside chunk starting at {chunk_start_step} with horizon {flow.shape[0]}"
        )
    return flow[offset]
