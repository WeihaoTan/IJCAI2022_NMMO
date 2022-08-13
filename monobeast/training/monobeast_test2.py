# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing
from typing import Dict, List

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

from pathlib import Path

import torch
from ijcai2022nmmo import CompetitionConfig, TeamBasedEnv
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

#from torchbeast.core import file_writer, prof, vtrace
from torchbeast.neural_mmo.monobeast_wrapper import \
    MonobeastWrapper as Environment
from torchbeast.neural_mmo.net import NMMONet
from torchbeast.neural_mmo.train_wrapper_changed import TrainWrapper


parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")
flags = parser.parse_args()
flags.num_agents = int(CompetitionConfig.NENT / CompetitionConfig.NPOP)

def create_env(flags):
    cfg = CompetitionConfig()
    cfg.NMAPS = 400
    return TrainWrapper(TeamBasedEnv(config=cfg))


def batch(env_output: Dict, filter_keys: List[str]):
    """Transform agent-wise env_output to bach format."""
    filter_keys = list(filter_keys)
    obs_batch = {key: [] for key in filter_keys}
    agent_ids = []
    for agent_id, out in env_output.items():
        agent_ids.append(agent_id)
        for key, val in out.items():
            if key in filter_keys:
                obs_batch[key].append(val)
    for key, val in obs_batch.items():
        obs_batch[key] = torch.cat(val, dim=1)

    return obs_batch, agent_ids


def net(input_dict, state=(), training=False):
    print("input_dict", input_dict.keys())

    # [T, B, ...]
    assert "va" in input_dict
    terrain, camp, entity = input_dict["terrain"], input_dict[
        "camp"], input_dict["entity"]
    terrain = F.one_hot(terrain, num_classes=6).permute(0, 1, 4, 2, 3)
    camp = F.one_hot(camp, num_classes=4).permute(0, 1, 4, 2, 3)


    print("terrain", terrain.shape)
    print("camp", camp.shape)
    print("entity", entity.shape)


    # print(terrain.shape, camp.shape, entity.shape)
    x = torch.cat([terrain, camp, entity], dim=2)

    print("x", x.shape)

    T, B, C, H, W = x.shape
    # # assert C == 17 and H == W == 15
    # x = torch.flatten(x, 0, 1)
    # x = self.cnn(x)
    # x = torch.flatten(x, start_dim=1)
    # x = F.relu(self.core(x))

    # logits = self.policy(x)
    # baseline = self.baseline(x)

    # va = input_dict.get("va", None)
    # if va is not None:
    #     va = torch.flatten(va, 0, 1)

    # dist = MaskedPolicy(logits, valid_actions=va)
    # if not training:
    #     action = dist.sample()
    #     action = action.view(T, B)
    # else:
    #     action = None
    # policy_logits = dist.logits.view(T, B, -1)
    # baseline = baseline.view(T, B)
    # output = dict(policy_logits=policy_logits, baseline=baseline)
    # if action is not None:
    #     output["action"] = action
    # return (output, tuple())



actor_index = 10

gym_env = create_env(flags)
seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
gym_env.seed(seed)
env = Environment(gym_env)
env_output = env.initial()

for i in range(5):
    env_output = env.step({})


env_output_batch, agent_ids = batch(
    env_output, filter_keys=gym_env.observation_space.keys())
# agent_output_batch, unused_state = model(env_output_batch, agent_state)
# agent_output = unbatch(agent_output_batch, agent_ids)

# print(env_output.keys())

# print(env_output[0].keys())

# # print("terrain")
# # print(env_output[0]['terrain'])


# for key in env_output[4].keys():
#     print(key)
#     print(env_output[4][key])
#     print(env_output[4][key].shape)

# print(env_output_batch.keys())

# # for key in env_output_batch.keys():
# #     print(key)
# #     print(env_output_batch[key])
# #     print(env_output_batch[key].shape)


# net(input_dict=env_output_batch)

#print(env_output[4]['entity'].shape)

print('terrain')
print(env_output[4]['terrain'][0][0])

print('camp')
print(env_output[4]['camp'][0][0])

for i in range(7):
    print(i)
    print(env_output[4]['entity'][0][0][i])


#  0      1      2      3      4       5
# lava, water, grass, scrub, forest, stone

#  0      1       2       3
# none,  npc, teammate, opponent

#   0      1         2        3     4      5          6      
# level, damage, timealive, food, water, health, is_freezed