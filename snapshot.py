#!/usr/bin/env python3


from beartype import beartype
from brax.training.acme.running_statistics import RunningStatisticsState
from brax.training.agents.ppo.losses import PPONetworkParams
from cv2 import VideoWriter, VideoWriter_fourcc as fourcc
from env import Env, PRODUCTION_REFRESH_RATE_HZ
from jax import jit, random as jr
from jaxtyping import jaxtyped
from os import makedirs
from typing import Any, Callable, Dict, Tuple, Union


N_SNAPSHOTS = 10
N_STEPS_PER_SNAPSHOT = 500


EVAL_ENV = Env()
JIT_RESET = jit(EVAL_ENV.reset)
JIT_STEP = jit(EVAL_ENV.step)
RENDER = EVAL_ENV.render


@jaxtyped(typechecker=beartype)
def snapshot(
    current_step: int,
    make_policy: Callable,
    params: Tuple[RunningStatisticsState, Union[PPONetworkParams, Dict[str, Any]]],
) -> None:
    if type(params[1]) == "dict":
        return snapshot_valid_input(current_step, make_policy, params[0], params[1])
    else:
        return snapshot_valid_input(
            current_step, make_policy, params[0], params[1].policy
        )


@jaxtyped(typechecker=beartype)
def snapshot_valid_input(
    current_step: int,
    make_policy: Callable,
    param0: RunningStatisticsState,
    param1: Dict[str, Any],
) -> None:
    print(f"Rendering a snapshot...")
    policy = make_policy((param0, param1))
    snapshot_folder = f"snapshot_step_{current_step}"
    makedirs(snapshot_folder, exist_ok=True)
    keys = jr.split(jr.PRNGKey(42), N_SNAPSHOTS)
    for sample_number, key in enumerate(keys):
        key, *keys = jr.split(key, 1 + N_STEPS_PER_SNAPSHOT)
        state = JIT_RESET(key)
        rollout = []
        for i, key in enumerate(keys):
            result = policy(state.obs, key)
            ctrl, _ = result
            state = JIT_STEP(state, ctrl)
            rollout.append(state.pipeline_state)
            if state.done:
                break
        video = RENDER(rollout)
        writer = VideoWriter(
            f"{snapshot_folder}/sample_{sample_number}.mp4",
            fourcc(*"mp4v"),
            PRODUCTION_REFRESH_RATE_HZ,
            video[0].shape[:-1][::-1],
        )
        for frame in video:
            writer.write(frame[..., ::-1])
        writer.release()
