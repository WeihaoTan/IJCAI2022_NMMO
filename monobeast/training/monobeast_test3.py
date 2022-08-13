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
from ijcai2022nmmo import CompetitionConfig
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

#from torchbeast.core import file_writer, prof, vtrace
from torchbeast.neural_mmo.monobeast_wrapper import \
    MonobeastWrapper as Environment
from torchbeast.neural_mmo.net import NMMONet
from torchbeast.neural_mmo.train_wrapper_changed import TrainWrapper
from torchbeast.neural_mmo.team_based_env import TeamBasedEnv
from torchbeast.neural_mmo.net import NMMONet


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

actor_index = 10

# gym_env = create_env(flags)
# seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
# gym_env.seed(seed)
# env = Environment(gym_env)
# env_output = env.initial()

# env_output = env.step({})
# print("success")


gym_env = create_env(flags)
seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
gym_env.seed(seed)
env = Environment(gym_env)
Net = NMMONet
model = Net(gym_env.observation_space, gym_env.action_space.n, False)
env_output = env.initial()
agent_state = model.initial_state(batch_size=1)
env_output_batch, agent_ids = batch(
    env_output, filter_keys=gym_env.observation_space.keys())
agent_output_batch, h_states = model(env_output_batch)
print(agent_output_batch)
#agent_output = unbatch(agent_output_batch, agent_ids)


# obs = gym_env.testReset()

# history_info = gym_env.get_history_info2()
# print(history_info['player'])
# print(history_info['team'])



# print(env_output.keys())

# print(env_output[0].keys())

# # print("terrain")
# # print(env_output[0]['terrain'])


# for key in env_output[2].keys():
#     print(key)
#     print(env_output[2][key])
#     print(env_output[2][key].shape)




# for agent_id in gym_env.realm.players.entities.keys():
#     print(agent_id, gym_env.realm.players.entities[agent_id].food)





# achievement_rewards = player.diary.update(self.realm, player)





#env_output_batch, agent_ids = batch(env_output, filter_keys=gym_env.observation_space.keys())



    # def act(self, observations):
    #     raw_observations = observations
    #     observations = self.feature_parser.parse(observations)
    #     observations = tree.map_structure(
    #         lambda x: torch.from_numpy(x).view(1, 1, *x.shape), observations)
    #     obs_batch, ids = batch(observations,
    #                            self.feature_parser.feature_spec.keys())
    #     output, _ = self.model(obs_batch)
    #     output = unbatch(output, ids)
    #     actions = {i: output[i]["action"].item() for i in output}
    #     actions = TrainWrapper.transform_action(actions, raw_observations,
    #                                             self.auxiliary_script)
    #     return actions