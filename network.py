#!/usr/bin/env python3


from beartype import beartype
from brax.training.acme.running_statistics import RunningStatisticsState
from brax.training.agents.ppo import networks, train
from brax.training.agents.ppo.losses import PPONetworkParams
from datetime import datetime
from etils import epath
from flax.training import orbax_utils
import functools
from jaxtyping import jaxtyped
import measurements
from orbax import checkpoint as ocp
from randomize import randomize
from typing import Any, Callable, Dict, Tuple, Union


N_TIMESTEPS_IN_TRAINING = 100_000_000  # 100_000
POLICY_HIDDEN_LAYER_SIZES = (128,) * 4
NORMALIZE_OBSERVATIONS = True


BATCH_SIZE = 32
N_MINIBATCHES = 16
N_ENVS = BATCH_SIZE * N_MINIBATCHES
N_CHECKPOINTS = 100
EPISODE_LENGTH_SECONDS = 2.5


factory = functools.partial(
    networks.make_ppo_networks,
    policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
)


CHECKPOINT_PATH = epath.Path("/tmp/quadrupred_joystick/ckpts")
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)


@jaxtyped(typechecker=beartype)
def policy_params(
    current_step: int,
    make_policy: Callable,
    params: Tuple[RunningStatisticsState, Union[PPONetworkParams, Dict[str, Any]]],
) -> None:
    # save checkpoints
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = CHECKPOINT_PATH / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    __import__("snapshot").snapshot(current_step, make_policy, params)


START_TIME = datetime.now()
JIT_TIME = None


@jaxtyped(typechecker=beartype)
def progress(num_steps: int, metrics: Dict[str, Any]) -> None:
    global JIT_TIME

    if JIT_TIME is None:
        JIT_TIME = datetime.now()
        print(f"    finished in {JIT_TIME - START_TIME}")
        print("Training...")

    print("")
    print(
        f"    After {num_steps} timesteps, reward distribution is {metrics['eval/episode_reward']} +/- {metrics['eval/episode_reward_std']}"
    )
    print("    Breakdown:")
    for k, v in metrics.items():
        print(f"        {k}: {v}")


episode_length = EPISODE_LENGTH_SECONDS / measurements.SIMULATION_TIMESTEP_SECONDS
train = functools.partial(
    train.train,
    learning_rate=3e-4,
    num_timesteps=N_TIMESTEPS_IN_TRAINING,
    normalize_observations=NORMALIZE_OBSERVATIONS,
    num_envs=N_ENVS,
    num_evals=N_CHECKPOINTS,
    episode_length=episode_length,
    batch_size=BATCH_SIZE,
    num_minibatches=N_MINIBATCHES,
    network_factory=factory,
    # randomization_fn=randomize,
    policy_params_fn=policy_params,
    progress_fn=progress,
)
