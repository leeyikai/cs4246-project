import torch
import time

class AgarObservation():
    def __init__(self, obs):
        self.obs = obs
        self.tensor_obs = None

    def to(self, device):
        parsed_env_obs = []
        for player_obs in self.obs:
            playercells = [torch.from_numpy(player).to(device) for player in player_obs['player']] if player_obs['player'] is not None else None
            foodcells = torch.from_numpy(player_obs['food']).to(device) if player_obs['food'] is not None else None
            viruscells = torch.from_numpy(player_obs['virus']).to(device) if player_obs['virus'] is not None else None
            ejectedcells = torch.from_numpy(player_obs['ejected']).to(device) if player_obs['ejected'] is not None else None
            parsed_env_obs.append({'player': playercells, 'food': foodcells, 'virus': viruscells, 'ejected': ejectedcells})

        self.tensor_obs = parsed_env_obs

        return self
    def get_item(self):
        item = [0,0,0,0,0]
        
        if self.tensor_obs is None:
            return item
        else:
            for player_obs in self.obs:
                player = player_obs['player']
                food = player_obs['food']
                virus = player_obs['virus']
                ejected = player_obs['ejected']
                item.append({'player': player, 'food': food, 'virus': virus, 'ejected': ejected})
            print(item)

            return item