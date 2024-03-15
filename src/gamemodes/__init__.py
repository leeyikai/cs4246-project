from gamemodes.FFA import FFA
from gamemodes.Teams import Teams

def Get_Game_Mode(mode):
    if mode == 0:
        return FFA()

    elif mode == 1:
        return Teams()