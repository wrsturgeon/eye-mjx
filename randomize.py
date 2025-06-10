#!/usr/bin/env python3


from beartype import beartype
from brax.base import System
from jax import random as jr, tree_util as jt, vmap
from jaxtyping import jaxtyped, Array, Float, UInt
from typing import Tuple


@jaxtyped(typechecker=beartype)
def randomize(sys: System, keys: UInt[Array, "*batch 2"]) -> Tuple[System, System]:
    """Randomizes the mjx.Model."""

    @jaxtyped(typechecker=beartype)
    def rand(
        key: UInt[Array, "2"],
    ) -> Tuple[
        Float[Array, "geoms 3"],
        Float[Array, "actuators *batch"],
        Float[Array, "actuators *batch"],
    ]:
        _, key = jr.split(key, 2)
        # friction
        friction = jr.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jr.split(key, 2)
        gain_range = (-5, 5)
        param = (
            jr.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1])
            + sys.actuator_gainprm[:, 0]
        )
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias

    friction, gain, bias = vmap(rand)(keys)

    in_axes = jt.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    sys = sys.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )

    return sys, in_axes


if __name__ == "__main__":
    from brax.io import mjcf
    import mujoco
    import generate_mjcf

    sys = mjcf.load(generate_mjcf.MJCF_XML_PATH_SCENE)
    key = jr.PRNGKey(42)
    keys = jr.split(key, 42)
    randomize(sys, keys)
