import numpy as np
# import torch


# # raw_pop = 5
# # print(np.sign(raw_pop) + 2)


# # raw_pop = -5
# # print(np.sign(raw_pop) + 2)


# print({} == {})


# import nmmo
# from ijcai2022nmmo import CompetitionConfig

# env = nmmo.Env(CompetitionConfig())
# obs = env.reset()
# # print(obs.keys())

# obs, reward, done, info  = env.step({})

# # print("obs", obs)
# # print("reward", reward)
# # print("done", done)
# # print("info", info)


# # center = env.config.TERRAIN_CENTER
# # border = env.config.TERRAIN_BORDER

# # print(center, border)

# #print(env.realm.)
# # print(env.realm.players.items)


# for agent_id in env.realm.players.entities.keys():
#     #print(agent_id, env.realm.players.entities[agent_id].resources.food.val)
#     print(agent_id, env.realm.players.entities[agent_id].history.playerKills)

    













# for i in range(128):
#     print(env.realm.players[i + 1].skills.packet())

#print(env.realm.players.entities[1])



# for entID in list(env.realm.entities):
#     player = self.entities[entID]
#     if not player.alive:
#     r, c  = player.base.pos
#     entID = player.entID
#     self.dead[entID] = player




# from pdb import set_trace as T
# import numpy as np

# class ExperienceCalculator:
#    def __init__(self):
#       self.exp = [0]
#       self.tabulateExp()

#    def tabulateExp(self, numLevels=99):
#       for i in range(2, numLevels+1):
#          increment = np.floor(i-1 + 300*(2**((i-1)/7.0)))/4.0
#          self.exp += [self.exp[-1] + increment]

#       self.exp = np.floor(np.array(self.exp))

#    def expAtLevel(self, level):
#       return self.exp[level - 1]

#    def levelAtExp(self, exp):
#       if exp >= self.exp[-1]:
#          return len(self.exp)
#       return np.argmin(exp >= self.exp)


# e = ExperienceCalculator()

# print(e.expAtLevel(10))
# print(e.expAtLevel(50))




# class FeatureParser:
#     map_size = 15

#     def parse(self, obs):
#         print(self.map_size)

# a = np.arange(10)
# a[1:5] = a[1:5] + 1

# print(a)
# import torch
# x = torch.tensor([[1, 2, 3], [1, 2, 3]])
# print(x.shape)
# x = x.unsqueeze(1)
# print(x.shape)
# x = x.repeat(1, 8, 1)
# print(x.shape, x)

# print(x.unsqueeze(1).repeat(1, 8, 1))

# for i in range(0, 8):
#     print(i)

# TT_ID = 0
# print({i: 0 for i in range(TT_ID, TT_ID + 8)})

# a = {0: 1, 7:100}
# # if 0 in a:
# #     print("1")
# # else:
# #     print(2)

# for x,y in a.items():
#     print(x, y)

raw_entity = np.arange(10)
for tid, line in enumerate(raw_entity):
    print(tid, line)

# terrain = np.zeros((15, 15), dtype=np.int64)
# NEIGHBOR = [(6, 7), (8, 7), (7, 8), (7, 6)]

# # for i in NEIGHBOR:
# #     print(i)
# #     print(terrain[i])

# # print(dir for dir in NEIGHBOR)
# # a = [dir for dir in NEIGHBOR]
# # print(a)
# a = [terrain[dir] for dir in NEIGHBOR]
# if 0 in a:
#     print(a)


