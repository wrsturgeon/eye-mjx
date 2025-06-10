#!/usr/bin/env python3


from beartype import beartype
from brax import math
from brax.base import State as PipelineState
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import generate_mjcf
from jax import numpy as jp, random as jr
from jaxtyping import jaxtyped, Array, Float, UInt
import measurements
import mujoco
import network
from typing import Any, Dict
import units


PRODUCTION_REFRESH_RATE_HZ = 50
OBSERVATION_HISTORY_SIZE = 15


LINEAR_VELOCITY_STDDEV = units.inches(12) / units.seconds(1)
ANGULAR_VELOCITY_STDDEV = units.degrees(180) / units.seconds(1)
HEIGHT_EXPECTATION = measurements.LENGTH_KNEE_TO_FOOT
HEIGHT_STDDEV = measurements.LENGTH_KNEE_TO_FOOT / 2
TILT_STDDEV = units.degrees(15)


SENSOR_NOISE = 0.01


PIPELINE_DEBUG = True


class Env(PipelineEnv):

    @jaxtyped(typechecker=beartype)
    def __init__(self):
        sys = mjcf.load(generate_mjcf.MJCF_XML_PATH_SCENE)

        self._sphere_geom_id = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_GEOM.value, "sphere"
        )
        assert self._sphere_geom_id != -1, "Sphere body not found!"

        # Arguments to `PipelineEnv` from <https://github.com/google/brax/blob/d59e4db582e98da1734c098aed7219271c940bda/brax/envs/base.py#L95C5-L100C67>:
        #   sys: system defining the kinematic tree and other properties
        #   backend: string specifying the physics pipeline
        #   n_frames: the number of times to step the physics pipeline for each
        #     environment step
        #   debug: whether to get debug info from the pipeline init/step

        # In production, we want to run this algorithm
        # `PRODUCTION_REFRESH_RATE_HZ` times per second,
        # so `n_frames` needs to be the number of physics steps per second
        # divided by `PRODUCTION_REFRESH_RATE_HZ`.
        physics_steps_per_second = 1 / sys.opt.timestep
        n_frames = physics_steps_per_second / PRODUCTION_REFRESH_RATE_HZ
        super().__init__(
            sys=sys, backend="mjx", n_frames=n_frames, debug=PIPELINE_DEBUG
        )
        assert int(round(1 / self.dt)) == PRODUCTION_REFRESH_RATE_HZ

    @jaxtyped(typechecker=beartype)
    def reset(self, key: UInt[Array, "2"]) -> State:
        command_key, state_key, obs_key = jr.split(key, 3)
        pipeline_state = self.pipeline_init(q=self.sys.qpos0, qd=jp.zeros(self.sys.nv))
        info = {
            "command": self._sample_command(command_key),
            "history": jp.zeros((OBSERVATION_HISTORY_SIZE, 6)),
            "key": state_key,
            "positions_requested_last_frame": jp.zeros(self.action_size),
            "step": 0,
        }
        sensor_readings = self._read_sensors(pipeline_state, obs_key)
        obs = self._observations_are_more_than_sensor_readings(sensor_readings, info)
        reward, done = jp.zeros(2)
        metrics = {f"loss/{k}": 0.0 for k in self._loss_weights().keys()}
        metrics["reward"] = 0.0
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    @jaxtyped(typechecker=beartype)
    def step(self, state: State, action: Float[Array, "actions"]) -> State:
        command = state.info["command"]
        key, obs_key = jr.split(state.info["key"])

        # Physics step:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        x, xd = pipeline_state.x, pipeline_state.xd

        # Observation/sensing step:
        sensor_readings = self._read_sensors(pipeline_state, obs_key)
        obs = self._observations_are_more_than_sensor_readings(
            sensor_readings, state.info
        )

        height = pipeline_state.geom_xpos[self._sphere_geom_id][2]
        done = jp.zeros(())

        # Loss calculation:
        @jaxtyped(typechecker=beartype)
        def loss_by_name(name: str) -> Float[Array, ""]:
            if name == "linear_velocity":
                ideal = command[:2] / LINEAR_VELOCITY_STDDEV
                actual = (
                    math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))[:2]
                    / LINEAR_VELOCITY_STDDEV
                )
            elif name == "angular_velocity":
                ideal = command[2] / ANGULAR_VELOCITY_STDDEV
                actual = (
                    math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))[2]
                    / ANGULAR_VELOCITY_STDDEV
                )
            elif name == "height":
                ideal = command[3] / HEIGHT_STDDEV
                actual = height / HEIGHT_STDDEV
            elif name == "tilt":
                ideal = command[4:6] / TILT_STDDEV
                actual = (
                    jp.arcsin(math.rotate(jp.array([0.0, 0.0, 1.0]), x.rot[0])[:2])
                    / TILT_STDDEV
                )
            elif name == "submerging":
                ideal = jp.zeros(height.shape)
                actual = (height < measurements.SPHERE_RADIUS)
            else:
                raise ValueError(f"Unrecognized loss name: `{name}`")
            error = actual - ideal
            L1 = jp.mean(jp.abs(error))
            L2 = jp.mean(jp.square(error))
            return 0.5 * L1 + L2

        metrics = {
            f"loss/{k}": v * loss_by_name(k) for k, v in self._loss_weights().items()
        }
        reward = -sum(metrics.values())
        metrics["reward"] = reward

        @jaxtyped(typechecker=beartype)
        def info_by_name(name: str, passthrough: Any) -> Any:
            if name == "history":
                return jp.concatenate([sensor_readings[jp.newaxis], passthrough[:-1]])
            if name == "key":
                return key
            if name == "positions_requested_last_frame":
                return action
            if name == "step":
                return passthrough + 1
            # if (
            #     name == "command"
            #     or name == "first_obs"
            #     or name == "first_pipeline_state"
            # ):
            #     return passthrough
            # raise ValueError(
            #     f"Unrecognized state info key: `{name}` (previously =`{passthrough}`)"
            # )
            return passthrough

        info = {k: info_by_name(k, v) for k, v in state.info.items()}

        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    @jaxtyped(typechecker=beartype)
    def _read_sensors(
        self, pipeline_state: PipelineState, key: UInt[Array, "2"]
    ) -> Float[Array, "flattened"]:
        obs = pipeline_state.sensordata
        return obs + jr.normal(key, obs.shape) * SENSOR_NOISE

    @jaxtyped(typechecker=beartype)
    def _observations_are_more_than_sensor_readings(
        self,
        sensor_readings: Float[Array, "sensors"],
        info: Dict[str, Any],
    ) -> Float[Array, "flattened"]:
        command = info["command"]
        history = info["history"]
        positions_requested_last_frame = info["positions_requested_last_frame"]

        sensor_readings_and_history = jp.concatenate(
            [sensor_readings[jp.newaxis], history]
        )

        with_state = jp.concatenate(
            [
                command,  # YOU HAVE TO FUCKING PASS IT THE COMMAND WHAT THE FUCK ?!?!? (angery)
                positions_requested_last_frame,
                sensor_readings_and_history.flatten(),
            ]
        )
        return with_state

    @jaxtyped(typechecker=beartype)
    def _sample_command(self, key: UInt[Array, "2"]) -> Float[Array, "6"]:
        # lv_x, lv_y, av, h, t_x, t_y
        command = jr.normal(key, (6,))
        command = command.at[0:2].multiply(LINEAR_VELOCITY_STDDEV)
        command = command.at[2].multiply(ANGULAR_VELOCITY_STDDEV)
        command = command.at[3].multiply(HEIGHT_STDDEV)
        command = command.at[3].add(HEIGHT_EXPECTATION)
        command = command.at[3].set(
            jp.where(
                command[3] < measurements.SPHERE_RADIUS,
                measurements.SPHERE_RADIUS,
                command[3],
            )
        )
        command = command.at[4:6].multiply(TILT_STDDEV)
        return command

    @jaxtyped(typechecker=beartype)
    def _loss_weights(self) -> Dict[str, Float[Array, ""]]:
        return {
            "linear_velocity": jp.asarray(1.0),
            "angular_velocity": jp.asarray(0.5),
            "height": jp.asarray(0.125),
            "tilt": jp.asarray(0.125),
            "submerging": jp.asarray(100.0),
        }


if __name__ == "__main__":
    # sys = mjcf.load(generate_mjcf.MJCF_XML_PATH_SCENE)
    # raise ValueError(len(sys.actuator.q_id))
    # raise ValueError(sys.nsensor)
    # raise ValueError("\n" + "\n".join(list(sorted(sys.__dict__.keys()))))
    env = Env()
