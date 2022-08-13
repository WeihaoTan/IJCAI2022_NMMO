import nmmo
from nmmo import entity
from typing import Dict, List, Tuple, Any

from ijcai2022nmmo.env.metrics import Metrics
from ijcai2022nmmo.env.stat import Stat


class TeamBasedEnv(object):
    players: Dict[int, entity.Player] = {}
    player_team_map: Dict[int, int] = {}
    team_players_map: Dict[int, List[int]] = {}

    def __init__(self, config: nmmo.config.Config) -> None:
        self._env = nmmo.Env(config)

    def __getattr__(self, __name: str) -> Any:
        if __name not in self.__dict__:
            return getattr(self._env, __name)

    def reset(self) -> Dict[int, Dict[int, dict]]:
        self.players.clear()
        self.player_team_map.clear()
        self.team_players_map.clear()

        observations = self._env.reset(None, True)
        for player in self._env.realm.players.entities.values():
            player: entity.Player
            self.players[player.entID] = player
            self.player_team_map[player.entID] = player.pop
            if player.pop not in self.team_players_map:
                self.team_players_map[player.pop] = []
            self.team_players_map[player.pop].append(player.entID)

        return self._split_by_team(observations)

    def step(
        self,
        actions_by_team: Dict[int, Dict[int, dict]],
    ) -> Tuple[Dict[int, Dict[int, dict]], Dict[int, Dict[int, int]], Dict[
            int, Dict[int, bool]], Dict[int, Dict[int, dict]]]:
        # merge actions
        actions = {}
        for team_idx, team_actions in actions_by_team.items():
            player_ids = self.team_players_map[team_idx]
            for i, action in team_actions.items():
                # avoid invalid player id
                if i >= 0 and i < len(player_ids):
                    actions[player_ids[i]] = action

        observations, rewards, dones, infos = self._env.step(actions)

        # delete the observations of the done players
        for player_id, done in dones.items():
            if done and player_id in observations:
                del observations[player_id]

        return (
            self._split_by_team(observations),
            self._split_by_team(rewards),
            self._split_by_team(dones),
            self._split_by_team(infos),
        )

    def metrices_by_team(self) -> Dict[int, Dict[int, Metrics]]:
        metrices: Dict[int, Metrics] = {}
        for player in self.players.values():
            metrices[player.entID] = Metrics.collect(self, player)
        return self._split_by_team(metrices)

    def stat_by_team(self) -> Dict[int, Stat]:
        stat_by_team = {}
        for team_idx, metrices in self.metrices_by_team().items():
            stat_by_team[team_idx] = Stat.from_metrices(metrices.values())
        return stat_by_team

    def _split_by_team(self, xs: Dict[int, Any]) -> Dict[int, Dict[int, Any]]:
        xs_by_team = {}
        for player_id, x in xs.items():
            team_idx = self.player_team_map[player_id]
            if team_idx not in xs_by_team:
                xs_by_team[team_idx] = {}
            xs_by_team[team_idx][self.team_players_map[team_idx].index(
                player_id)] = x
        return xs_by_team


    def get_history_info(self):
        player_info = {}
        team_info = {}

        for player in self._env.realm.players.entities.values():
            player_info[player.entID] = {}
            player_info[player.entID]['health'] = player.resources.health.val 
            player_info[player.entID]['health_max'] = player.resources.health.max
            player_info[player.entID]['water'] = player.resources.water.val
            player_info[player.entID]['food'] = player.resources.food.val
            player_info[player.entID]['playerKills'] = player.history.playerKills
            player_info[player.entID]['equipment'] = player.loadout.defense
            player_info[player.entID]['exploration'] = player.history.exploration
            player_info[player.entID]['foraging'] = (player.skills.fishing.level + player.skills.hunting.level) / 2.0
            # player_info[player.entID]['fishing'] = player.skills.fishing.level
            # player_info[player.entID]['hunting'] = player.skills.hunting.level

        teams = self._split_by_team(player_info)

        for teamID, teamInfo in teams.items():
            max_playerKills = 0
            max_equipment = 0
            max_exploration = 0
            max_foraging = 0
            for playerID, playerInfo in teamInfo.items():
                max_playerKills = max(max_playerKills, playerInfo['playerKills'])
                max_equipment = max(max_equipment, playerInfo['equipment'])
                max_exploration = max(max_exploration, playerInfo['exploration'])
                max_foraging = max(max_foraging, playerInfo['foraging'])
        
            team_info[teamID] = {'max_playerKills':max_playerKills, 'max_equipment':max_equipment, 'max_exploration':max_exploration, 'max_foraging':max_foraging}
        
        history_info = {'player': teams, 'team': team_info}

        return history_info