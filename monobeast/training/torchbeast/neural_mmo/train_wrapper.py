from distutils.log import info
import nmmo
import numpy as np
from gym import Wrapper, spaces
# from ijcai2022nmmo import TeamBasedEnv
from torchbeast.neural_mmo.team_based_env import TeamBasedEnv
from ijcai2022nmmo.scripted import CombatTeam, ForageTeam, RandomTeam
from ijcai2022nmmo.scripted.baselines import Scripted
from ijcai2022nmmo.scripted.scripted_team import ScriptedTeam
from toml import TomlDecodeError
from nmmo.io import action

#new
class ExperienceCalculator:
   def __init__(self):
      self.exp = [0]
      self.tabulateExp()

   def tabulateExp(self, numLevels=99):
      for i in range(2, numLevels+1):
         increment = np.floor(i-1 + 300*(2**((i-1)/7.0)))/4.0
         self.exp += [self.exp[-1] + increment]

      self.exp = np.floor(np.array(self.exp))

   def expAtLevel(self, level):
      return self.exp[level - 1]

   def levelAtExp(self, exp):
      if exp >= self.exp[-1]:
         return len(self.exp)
      return np.argmin(exp >= self.exp)

class FeatureParser:
    map_size = 15
    n_actions = 5
    NEIGHBOR = [(6, 7), (8, 7), (7, 8), (7, 6)]  # north, south, east, west
    OBSTACLE = (0, 1, 5)  # lava, water, stone
    feature_spec = {
        "terrain": spaces.Box(low=0, high=6, shape=(15, 15), dtype=np.int64),
        "camp": spaces.Box(low=0, high=5, shape=(15, 15), dtype=np.int64),
        "entity": spaces.Box(low=0,
                             high=10,
                             shape=(7, 15, 15),
                             dtype=np.float32),
        "va": spaces.Box(low=0, high=2, shape=(5, ), dtype=np.int64),

        #new
        "player_info": spaces.Box(low=0, high=2, shape=(26, ), dtype=np.float32),
        "cur_task": spaces.Box(low=0, high=2, shape=(5, ), dtype=np.float32),
        "global_terrain": spaces.Box(low=0, high=6, shape=(160, 160), dtype=np.int64),
        "global_camp": spaces.Box(low=0, high=6, shape=(160, 160), dtype=np.int64),
    }

    #new
    task_info = {}
    prev_global_terrain = np.zeros((160, 160), dtype=np.int64)
    expCalculator = ExperienceCalculator()

    #health_history = np.zeros((8, 1028), dtype=np.float32)
    health_history = {}
    attack_target = {}

    def parse(self, obs):
        ret = {}
        attack_action = {}
        
        global_terrain = np.zeros((160, 160), dtype=np.int64)
        global_camp = np.zeros((160, 160), dtype=np.int64)
        cur_task = np.zeros(5, dtype=np.float32)
        #global_camp = np.zeros((self.map_size, self.map_size), dtype=np.float32)

        for agent_id in obs:
            terrain = np.zeros((self.map_size, self.map_size), dtype=np.int64)
            camp = np.zeros((self.map_size, self.map_size), dtype=np.int64)
            entity = np.zeros((7, self.map_size, self.map_size),dtype=np.float32)
            va = np.ones(self.n_actions, dtype=np.int64)

            #new
            player_info = np.zeros(26, dtype=np.float32)

            # terrain feature
            tile = obs[agent_id]["Tile"]["Continuous"]
            LT_R, LT_C = int(tile[0, 2]), int(tile[0, 3])
            for line in tile:
                terrain[int(line[2] - LT_R),int(line[3] - LT_C)] = int(line[1])

                #new
                global_terrain[int(line[2]), int(line[3])] = int(line[1])

            # npc and player
            raw_entity = obs[agent_id]["Entity"]["Continuous"]
            P = int(raw_entity[0, 4])
            _r, _c = int(raw_entity[0, 5]), int(raw_entity[0, 6])
            assert int(_r - LT_R) == int(_c - LT_C) == 7, f"({int(_r - LT_R)}, {int(_c - LT_C)})"

            #new
            #task exploration
            #Timealive
            timealive = int(raw_entity[0][8])
            cur_task[0] = timealive / 1024

            if timealive == 0:
                self.task_info[agent_id] = {}
                self.task_info[agent_id]['init_pos'] = raw_entity[0, 5:7]
                self.task_info[agent_id]['max_exploration'] = 0
                self.task_info[agent_id]['equipment'] = 0
                self.task_info[agent_id]['playerkill'] = 0

                # #max explore
                # cur_task[1] = 0
                # #max_level
                # cur_task[2] = 10.0 / 50.0
                # #max equipment
                # cur_task[3] = 0
                # #max playerkill
                # cur_task[4] = 0

                self.attack_target[agent_id] = {'id': 0}

            else:
                exploration = self.linf(self.task_info[agent_id]['init_pos'], raw_entity[0, 5:7])
                self.task_info[agent_id]['max_exploration'] = max(exploration, self.task_info[agent_id]['max_exploration'])
                #max_exp
                cur_task[1] = max(cur_task[1], self.task_info[agent_id]['max_exploration'] / 128.0)

            #task forage
            if timealive == 0:
                self.task_info[agent_id]['pre_food'] = raw_entity[0, 9]
                self.task_info[agent_id]['pre_water'] = raw_entity[0, 10]
                self.task_info[agent_id]['food_exp'] = self.expCalculator.expAtLevel(10)
                self.task_info[agent_id]['water_exp'] = self.expCalculator.expAtLevel(10)
                self.task_info[agent_id]['food_level'] = 10
                self.task_info[agent_id]['water_level'] = 10
            else:
                if timealive > 0 :
                    if self.prev_global_terrain[_r, _c] == 4 or global_terrain[_r, _c] == 4: 
                        self.task_info[agent_id]['food_exp'] += 10 * self.task_info[agent_id]['food_level']
                        self.task_info[agent_id]['food_level'] = self.expCalculator.levelAtExp(self.task_info[agent_id]['food_exp'])
                if 1 in [terrain[dir] for dir in self.NEIGHBOR]:
                    self.task_info[agent_id]['water_exp'] += 10 * self.task_info[agent_id]['water_level']
                    self.task_info[agent_id]['water_level'] = self.expCalculator.levelAtExp(self.task_info[agent_id]['water_exp'])
                #max_level
                cur_task[2] = max(cur_task[2],  (self.task_info[agent_id]['food_level'] + self.task_info[agent_id]['water_level']) / 100.0)

            
            player_level = raw_entity[0][3]
            player_damage = raw_entity[0][7]
            player_relative_level = player_level - 0.25 * player_damage

            if timealive == 0:
                self.health_history[agent_id] = np.zeros((1028), dtype=np.float32)
                self.health_history[agent_id][timealive:timealive + 4] += raw_entity[0, 11] / 30
            self.health_history[agent_id][timealive + 4] = raw_entity[0, 11] / 30

            #is_attacted
            player_info[0] = 1 if raw_entity[0, 2] != 0 else 0
            # level
            player_info[1] = raw_entity[0, 3] / 30
            # damage
            player_info[2] = raw_entity[0, 7] / 30
            #food
            player_info[3] = raw_entity[0, 9] / 50
            #water
            player_info[4] = raw_entity[0, 10] / 50
            #health
            #player_info[5] = raw_entity[0, 11] / 30
            player_info[5:10] = self.health_history[agent_id][timealive:timealive + 5]
            #is_freezed
            player_info[10] = raw_entity[0, 12]
            #x
            player_info[11] = raw_entity[0, 5] / 160
            #y
            player_info[12] = raw_entity[0, 6] / 160
            #init x y
            player_info[13] = self.task_info[agent_id]['init_pos'][0] / 160
            player_info[14] = self.task_info[agent_id]['init_pos'][1] / 160
            #max explore
            player_info[15] = min(self.task_info[agent_id]['max_exploration'] / 128.0, 1.0)
            #food_exp
            player_info[16] = min(self.task_info[agent_id]['food_exp'] /  self.expCalculator.expAtLevel(50), 2)        
            #food_level
            player_info[17] = self.task_info[agent_id]['food_level'] /  50
            #water_exp
            player_info[18] = min(self.task_info[agent_id]['water_exp'] /  self.expCalculator.expAtLevel(50), 2)        
            #water_level
            player_info[19] = self.task_info[agent_id]['water_level'] /  50
            #attacking x
            player_info[20] = 0
            #attacking y
            player_info[21] = 0
            #attacking hp
            player_info[22] = 0
            #attacking level
            player_info[23] = 0

            kill = True

            for line in raw_entity:
                if line[0] != 1:
                    continue
                raw_pop, raw_r, raw_c = int(line[4]), int(line[5]), int(line[6])
                r, c = int(raw_r - LT_R), int(raw_c - LT_C)
                # # new
                # if raw_pop == P:
                #     camp[r, c] = 4
                #     self.global_camp[raw_r, raw_c] = 4
                # #NPC: Aggressive, PassiveAggressive, Passive: 1, 2, 3
                # elif raw_pop < 0:
                #     camp[r, c] = raw_pop + 4
                #     self.global_camp[raw_r, raw_c] = raw_pop + 4
                # elif raw_pop > 0:
                #     camp[r, c] = 5
                #     self.global_camp[raw_r, raw_c] = raw_pop + 5
                camp[r, c] = 2 if raw_pop == P else np.sign(raw_pop) + 2
                global_camp[raw_r, raw_c] = 2 if raw_pop == P else np.sign(raw_pop) + 2

                #is_attacted
                entity[0, r, c] = 1 if line[2] != 0 else 0
                # relative level: (level - 0.25 * damage) / (player.level - 0.25 * player.damage)
                entity[1, r, c] = min((line[3] - 0.25 * line[7]) / player_relative_level, 2)
                # damage
                entity[2, r, c] = line[7] / 30
                #food
                entity[3, r, c] = line[9] / 50
                #water
                entity[4, r, c] = line[10] / 50
                #health
                entity[5, r, c] = line[11] / 30
                #is_freezed
                entity[6, r, c] = line[12]

                #kill
                if self.attack_target[agent_id]['id'] == line[1]:
                    kill = False
            
            if kill:
                if self.attack_target[agent_id]['id'] > 0:
                    self.task_info[agent_id]['playerkill'] += 1
                elif self.attack_target[agent_id]['id'] < 0:
                    self.task_info[agent_id]['equipment'] += 1

            #equipment
            player_info[24] =  min(self.task_info[agent_id]['equipment'] /  20, 2)
            #playerkill
            player_info[25] = min(self.task_info[agent_id]['playerkill'] / 6, 2)

            if timealive > 0:
                cur_task[3] = min(max(cur_task[3], self.task_info[agent_id]['equipment'] / 20), 2)
                cur_task[4] = min(max(cur_task[4], self.task_info[agent_id]['playerkill'] / 6), 2)

            # valid action
            for i, (r, c) in enumerate(self.NEIGHBOR):
                if terrain[r, c] in self.OBSTACLE:
                    va[i + 1] = 0

            ##########################################################################################
            #attack action
            action = {}
            candidate_target = {}

            attacker_ID = raw_entity[0][2]
            find_attacker = False

            for tid, line in enumerate(raw_entity):
                if line[0] != 1:
                    continue
                raw_pop, raw_r, raw_c = int(line[4]), int(line[5]), int(line[6])
                r, c = abs(int(raw_r - _r)), abs(int(raw_c - _c))
                if self.attack_target[agent_id]['id'] == line[1]:
                    prev_attack_target_pos = (r, c)
                    prev_attack_target_level = line[3] - 0.25 * line[7]
                    prev_attack_target_pop = raw_pop
                    prev_attack_target_rpos = (raw_r, raw_c)
                    prev_attack_target_hp = line[11]
                    prev_attack_target_tid = tid
                if attacker_ID == line[1]:
                    attacker_pos = (r, c)
                    attacker_level = line[3] - 0.25 * line[7]
                    attacker_pop = raw_pop
                    attacker_rpos = (raw_r, raw_c)
                    attacker_hp = line[11]
                    attacker_tid = tid
                    find_attacker = True


            if attacker_ID and find_attacker:
                if attacker_ID == self.attack_target[agent_id]['id'] or self.attack_target[agent_id]['id'] == 0:
                    if attacker_pos[0] <= 4 and attacker_pos[1] <= 4:
                        candidate_target['id'] = attacker_ID
                        candidate_target['pos'] = attacker_pos
                        candidate_target['pop'] = attacker_pop
                        candidate_target['rpos'] = attacker_rpos
                        candidate_target['hp'] = attacker_hp
                        candidate_target['level'] = attacker_level
                        candidate_target['tid'] = attacker_tid
                else:
                    if prev_attack_target_level > attacker_level and attacker_pos[0] <= 4 and attacker_pos[1] <= 4:
                        candidate_target['id'] = attacker_ID
                        candidate_target['pos'] = attacker_pos
                        candidate_target['pop'] = attacker_pop
                        candidate_target['rpos'] = attacker_rpos
                        candidate_target['hp'] = attacker_hp
                        candidate_target['level'] = attacker_level
                        candidate_target['tid'] = attacker_tid
                    elif prev_attack_target_level <= attacker_level and prev_attack_target_pos[0] <= 4 and prev_attack_target_pos[1] <= 4:
                        candidate_target['id'] = self.attack_target[agent_id]['id']
                        candidate_target['pos'] = prev_attack_target_pos
                        candidate_target['pop'] = prev_attack_target_pop
                        candidate_target['rpos'] = prev_attack_target_rpos
                        candidate_target['hp'] = prev_attack_target_hp
                        candidate_target['level'] = prev_attack_target_level
                        candidate_target['tid'] = prev_attack_target_tid

            if not candidate_target:       
                #             -3, -2, -1,   >1
                aux_level = [-100, 0, -5, -1000]
                for tid, line in enumerate(raw_entity):
                    if line[0] != 1:
                        continue
                    raw_pop, raw_r, raw_c = int(line[4]), int(line[5]), int(line[6])
                    r, c = abs(int(raw_r - _r)), abs(int(raw_c - _c))

                    if r <= 4 and c <= 4 and raw_pop != P:
                        candidate_pop = 3 if raw_pop > 0 else raw_pop + 3
                        candidate_level = line[3] - 0.25 * line[7] + aux_level[candidate_pop]
                        if candidate_target:
                            if raw_pop > 0:
                                if abs(line[2]) > candidate_target['is_attacked'] or candidate_target['level'] > candidate_level:
                                    candidate_target['id'] = line[1]
                                    candidate_target['level'] = candidate_level
                                    candidate_target['is_attacked'] = line[2]
                                    candidate_target['pop'] = raw_pop
                                    candidate_target['pos'] = (r, c)
                                    candidate_target['rpos'] = (raw_r, raw_c)
                                    candidate_target['hp'] = line[11]
                                    candidate_target['tid'] = tid
                            elif raw_pop == -1 or raw_pop == -3:
                                if candidate_target['level'] > candidate_level:
                                    candidate_target['id'] = line[1]
                                    candidate_target['level'] = candidate_level
                                    candidate_target['is_attacked'] = line[2]
                                    candidate_target['pop'] = raw_pop
                                    candidate_target['pos'] = (r, c)
                                    candidate_target['rpos'] = (raw_r, raw_c)
                                    candidate_target['hp'] = line[11]
                                    candidate_target['tid'] = tid
                            elif raw_pop == -2:
                                if candidate_target['level'] > candidate_level and player_relative_level + aux_level[candidate_pop] >= candidate_level - 3: 
                                    candidate_target['id'] = line[1]
                                    candidate_target['level'] = candidate_level
                                    candidate_target['is_attacked'] = line[2]
                                    candidate_target['pop'] = raw_pop
                                    candidate_target['pos'] = (r, c)
                                    candidate_target['rpos'] = (raw_r, raw_c)
                                    candidate_target['hp'] = line[11]
                                    candidate_target['tid'] = tid
                        else:
                                candidate_target['id'] = line[1]
                                candidate_target['level'] = candidate_level
                                candidate_target['is_attacked'] = line[2]     
                                candidate_target['pop'] = raw_pop 
                                candidate_target['pos'] = (r, c)      
                                candidate_target['rpos'] = (raw_r, raw_c) 
                                candidate_target['hp'] = line[11] 
                                candidate_target['tid'] = tid
                if candidate_target:   
                    c_p = 3 if candidate_target['pop'] > 0 else candidate_target['pop'] + 3
                    candidate_target['level'] -= aux_level[c_p]             
    
            if candidate_target:
                #attacking x
                player_info[20] = candidate_target['rpos'][0] / 128.0
                #attacking y
                player_info[21] = candidate_target['rpos'][1] / 128.0
                #attacking hp
                player_info[22] = candidate_target['hp'] / 30
                #attacking level
                player_info[23] = min(candidate_target['level'] / player_relative_level, 2)

                if timealive <= 50 and candidate_target['pop'] > 0:
                    if candidate_target['pos'][0] <= 1 and candidate_target['pos'][1] <= 1:
                        action = {
                            nmmo.action.Style: nmmo.action.Melee.index,
                            nmmo.action.Target: int(candidate_target['tid'])
                        }
                    elif candidate_target['pos'][0] <= 3 and candidate_target['pos'][1] <= 3:
                        action = {
                            nmmo.action.Style: nmmo.action.Range.index,
                            nmmo.action.Target: int(candidate_target['tid'])
                        }
                    else:
                        action = {
                            nmmo.action.Style: nmmo.action.Mage.index,
                            nmmo.action.Target: int(candidate_target['tid'])
                        }                        
                else:
                        action = {
                            nmmo.action.Style: nmmo.action.Mage.index,
                            nmmo.action.Target: int(candidate_target['tid'])
                        }   

            #is attacking pos
            if candidate_target:
                self.attack_target[agent_id] = candidate_target
            else:
                self.attack_target[agent_id] = {'id': 0}
            
            attack_action[agent_id] = action
            self.prev_global_terrain = global_terrain

            ret[agent_id] = {
                "terrain": terrain,
                "camp": camp,
                "entity": entity,
                "va": va,
                "player_info": player_info,
            }


        for agent_id in obs:
            ret[agent_id].update({
                "cur_task": cur_task,
                "global_terrain": global_terrain,
                "global_camp": global_camp
            })

        return ret, attack_action

    #new
    def linf(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return max(abs(r1 - r2), abs(c1 - c2))

class RewardParser:

    def parse(self, prev_history_info, history_info, TT_ID):
        info = {}

        reward = {i: 0 for i in range(TT_ID, TT_ID + 8)}

        for i in prev_history_info['player'][TT_ID]:
            if i in history_info['player'][TT_ID]:
                reward[i] = self.cal_reward(prev_history_info['player'][TT_ID][i], history_info['player'][TT_ID][i], prev_history_info['team'][TT_ID], history_info['team'][TT_ID])
            else:
                reward[i] = -5

        # reward = {
        #     i: (self.cal_reward(prev_history_info['player'][TT_ID][i], history_info['player'][TT_ID][i], prev_history_info['team'][TT_ID], history_info['team'][TT_ID])) for i in history_info['player'][TT_ID]
        # }

        return reward

    def cal_reward(self, prev_player_info, player_info, prev_team_info, team_info):
        reward = 0

        hp_coe = 0.5

        reward += (player_info['health'] - prev_player_info['health']) / player_info['health_max'] * hp_coe

        if team_info['max_foraging'] < 50:
            if player_info['water'] >= prev_player_info['water']:
                reward += 0.01
            if player_info['food'] >= prev_player_info['food']:
                reward += 0.01
    
        if player_info['water'] <= 3 and player_info['health'] <= 3: 
            reward -= 0.1
        if player_info['food'] <= 3 and player_info['health'] <= 3: 
            reward -= 0.1

        if team_info['max_playerKills'] < 6:
            reward += (player_info['playerKills'] - prev_player_info['playerKills']) * 1

        if team_info['max_equipment'] < 20:
            reward += (player_info['equipment'] - prev_player_info['equipment']) * 0.05

        if team_info['max_exploration'] < 127:
            reward += (player_info['exploration'] - prev_player_info['exploration']) * 0.05

        # if prev_player_info['health'] > 0 and player_info['health'] == 0:
        #     reward -= 5

        killTask = [1, 3, 6]
        equipmentTask = [1, 10, 20]
        exploreTask = [32, 64, 127]
        forageTask = [20, 35, 50]

        taskReward = [1, 3, 5]

        for i in range(3):
            if team_info['max_playerKills'] == killTask[i] and prev_team_info['max_playerKills'] < killTask[i]:
                reward += taskReward[i]
            if team_info['max_equipment'] == equipmentTask[i] and prev_team_info['max_equipment'] < equipmentTask[i]:
                reward += taskReward[i]
            if team_info['max_exploration'] == exploreTask[i] and prev_team_info['max_exploration'] < exploreTask[i]:
                reward += taskReward[i]
            if team_info['max_foraging'] == forageTask[i] and prev_team_info['max_foraging'] < forageTask[i]:
                reward += taskReward[i]

        return reward

class TrainWrapper(Wrapper):
    max_step = 1024
    TT_ID = 0  # training team index
    use_auxiliary_script = False

    def __init__(self, env: TeamBasedEnv) -> None:
        super().__init__(env)
        self.feature_parser = FeatureParser()
        self.reward_parser = RewardParser()
        self.observation_space = spaces.Dict(self.feature_parser.feature_spec)
        self.action_space = spaces.Discrete(5)
        self._dummy_feature = {
            key: np.zeros(shape=val.shape, dtype=val.dtype)
            for key, val in self.observation_space.items()
        }
        self.attack_actions = {}

    def reset(self):
        raw_obs = super().reset()
        obs = raw_obs[self.TT_ID]
        obs, attack_actions = self.feature_parser.parse(obs)

        self.attack_actions = attack_actions

        #self.reset_auxiliary_script(self.config)
        self.reset_scripted_team(self.config)
        self.agents = list(obs.keys())
        #self.agents.remove(1000)
        
        self._prev_history_info = self.get_history_info()

        self._prev_raw_obs = raw_obs
        self._step = 0
        
        return obs

    def step(self, actions):
        decisions = self.get_scripted_team_decision(self._prev_raw_obs)
        decisions[self.TT_ID] = self.transform_action(actions, self.attack_actions, observations=self._prev_raw_obs[self.TT_ID])

        raw_obs, _, raw_done, raw_info = super().step(decisions)
        if self.TT_ID in raw_obs:
            obs = raw_obs[self.TT_ID]
            done = raw_done[self.TT_ID]
            info = raw_info[self.TT_ID]

            obs, attack_actions = self.feature_parser.parse(obs)
            history_info = self.get_history_info()
            reward = self.reward_parser.parse(self._prev_history_info, history_info, self.TT_ID)
            self._prev_history_info = history_info
        else:
            obs, reward, done, info = {}, {}, {}, {}
            attack_actions = {}

        self.attack_actions = attack_actions

        for agent_id in self.agents:
            if agent_id not in obs:
                obs[agent_id] = self._dummy_feature
                reward[agent_id] = 0
                done[agent_id] = True

        self._prev_raw_obs = raw_obs
        self._step += 1

        if self._step >= self.max_step:
            done = {key: True for key in done.keys()}
        return obs, reward, done, info

    def reset_scripted_team(self, config):
        if getattr(self, "_scripted_team", None) is not None:
            for team in self._scripted_team.values():
                team.reset()
            return
        self._scripted_team = {}
        assert config.NPOP == 16
        for i in range(config.NPOP):
            if i == self.TT_ID:
                continue
            if self.TT_ID < i <= self.TT_ID + 7:
                self._scripted_team[i] = RandomTeam(f"random-{i}", config)
            elif self.TT_ID + 7 < i <= self.TT_ID + 12:
                self._scripted_team[i] = ForageTeam(f"forage-{i}", config)
            elif self.TT_ID + 12 < i <= self.TT_ID + 15:
                self._scripted_team[i] = CombatTeam(f"combat-{i}", config)

    def get_scripted_team_decision(self, observations):
        decisions = {}
        tt_id = self.TT_ID
        for team_id, obs in observations.items():
            if team_id == tt_id:
                continue
            decisions[team_id] = self._scripted_team[team_id].act(obs)
        #print("get_scripted_team_decision", decisions)
        return decisions

    # @staticmethod
    # def transform_action(move_actions, attack_actions, observations):
    #     """neural network move + scripted attack"""
    #     decisions = {}

    #     # move decisions
    #     for agent_id, val in move_actions.items():
    #         if observations is not None and agent_id not in observations:
    #             continue
    #         if val == 0:
    #             decisions[agent_id] = {}
    #         elif 1 <= val <= 4:
    #             decisions[agent_id] = {
    #                 nmmo.action.Move: {
    #                     nmmo.action.Direction: val - 1
    #                 }
    #             }
    #         else:
    #             raise ValueError(f"invalid action: {val}")
    #         decisions[agent_id] = attack_actions[agent_id]
    #     return decisions

    @staticmethod
    def transform_action(actions, attack_actions, observations):
        """neural network move + scripted attack"""
        decisions = {}

        # move decisions
        for agent_id, val in actions.items():
            if observations is not None and agent_id not in observations:
                continue
            if val == 0:
                decisions[agent_id] = {}
            elif 1 <= val <= 4:
                decisions[agent_id] = {
                    nmmo.action.Move: {
                        nmmo.action.Direction: val - 1
                    }
                }
            else:
                raise ValueError(f"invalid action: {val}")
            if attack_actions[agent_id]:
                decisions[agent_id][nmmo.action.Attack] = attack_actions[agent_id]

            #decisions[agent_id].update(nmmo.action.Attack:attack_actions[agent_id])

        # for agent_id, d in decisions.items():
        #     d.update(attack_actions[agent_id])
        #     decisions[agent_id] = d
        #print("testdecisions", decisions)
        return decisions

    # @staticmethod
    # def transform_action(actions, attack_actions):
    #     """neural network move + scripted attack"""
    #     decisions = {}

    #     # move decisions
    #     for agent_id, val in actions.items():
    #         if val == 0:
    #             decisions[agent_id] = {nmmo.action.Attack:attack_actions[agent_id]}
    #         elif 1 <= val <= 4:
    #             decisions[agent_id] = {
    #                 nmmo.action.Move: {
    #                     nmmo.action.Direction: val - 1
    #                 },
    #                 nmmo.action.Attack:attack_actions[agent_id]
    #             }
    #         else:
    #             raise ValueError(f"invalid action: {val}")
    #         # decisions[agent_id][nmmo.action.Attack] = attack_actions[agent_id]

    #         #decisions[agent_id].update(nmmo.action.Attack:attack_actions[agent_id])

    #     # for agent_id, d in decisions.items():
    #     #     d.update(attack_actions[agent_id])
    #     #     decisions[agent_id] = d
    #     print("decisions", decisions)
    #     return decisions



# class Attack(Scripted):
#     '''attack'''
#     name = 'Attack_'

#     def __call__(self, obs):
#         super().__call__(obs)

#         self.scan_agents()
#         self.target_weak()
#         self.style = nmmo.action.Range
#         self.attack()
#         return self.actions


# class AttackTeam(ScriptedTeam):
#     agent_klass = Attack
