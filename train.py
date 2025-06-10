#!/usr/bin/env python3


from brax.io import model
from env import Env
import network


SAVED_POLICY_FILEPATH = "trained.policy"


print("JIT-compiling the training loop...")
make_inference_fn, params, _ = network.train(environment=Env(), eval_env=Env())


print("Saving parameters...")
model.save_params(SAVED_POLICY_FILEPATH, params)
