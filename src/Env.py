# Author: Boyuan Chen
# Berkeley Artifical Intelligence Research

# The project is largely based on m-byte918's javascript implementation of the game with a lot of bug fixes and optimization for python
# Original Ogar-Edited project https://github.com/m-byte918/MultiOgar-Edited


import gym
from GameServer import GameServer
from modules import AgarObservation
from players import Player, Bot
import numpy as np
import rendering as rendering
import math
import time


class AgarEnv(gym.Env):
    def __init__(self, num_agents = 1, num_bots = 9, gamemode = 0):
        super(AgarEnv, self).__init__()
        self.viewer = None
        self.num_players = num_agents + num_bots
        self.num_agents = num_agents
        self.num_bots = num_bots
        self.gamemode = gamemode

        # factors for reward
        self.mass_reward_eps = 0.001  # make the max mass reward < 100 (max mass = 22500)
        self.kill_reward_eps = 200
        self.killed_reward_eps = 1000

    def step(self, actions, reward):
        for agent in self.agents:
            agent.step(actions)
        # for action, agent in zip(actions, self.agents):
        #     print(self.agents)
        #     agent.step(action)
        for bot in self.bots:
            bot.step()
        done = False
        self.server.Update()
        observations = self.parse_obs(self.agents[0])
        observations = self.split_observation(observations)

        rewards = np.array([self.parse_reward(agent) for agent in self.agents])
        #if rewards - reward > 0:
            # print("Grew in size")
            #rewards += 5
        if self.agents[0].isRemoved == True:
            done = True
        info = {}
        return observations, rewards, done, info

    def reset(self):
        self.server = GameServer()
        self.server.start(self.gamemode)
        self.agents = [Player(self.server) for _ in range(self.num_agents)]
        self.bots = [Bot(self.server) for _ in range(self.num_bots)]
        self.players = self.agents + self.bots
        self.server.addPlayers(self.players)
        self.viewer = None
        self.server.Update()
        observations = self.parse_obs(self.agents[0])
        return observations

    def parse_obs(self, player):
        obs = [{}, [], [], []]
        for cell in player.viewNodes:
            t, feature = self.cell_obs(cell, player)
            if t != 0:  # if type is not player, directly append
                obs[t].append(feature)
            else:
                owner = cell.owner
                if owner in obs[0]:
                    obs[0][owner].append(feature)
                else:
                    obs[0][owner] = [feature]

        playercells = [np.concatenate(v, 0) for k, v in obs[0].items()] # a list of np array. each array represents the state of all cells owned by a player
        foodcells = np.concatenate(obs[1], 0) if obs[1] else None # np array, each row represents a cell
        viruscells = np.concatenate(obs[2], 0) if obs[2] else None
        ejectedcells = np.concatenate(obs[3], 0) if obs[3] else None

        return {'player': playercells, 'food': foodcells, 'virus': viruscells, 'ejected': ejectedcells}

    def cell_obs(self, cell, player):
        if cell.cellType == 0:
            # player features
            boost_x = (cell.boostDistance * cell.boostDirection.x) / self.server.config.splitVelocity  # [-1, 1]
            boost_y = cell.boostDistance * cell.boostDirection.y / self.server.config.splitVelocity  # [-1, 1]
            radius = cell.radius / 400  # need to think about mean though [0, infinite...]  # fixme
            log_radius = np.log(cell.radius / 100)  # need to think about mean though   # fixme
            position_x = (cell.position.x - self.server.config.borderWidth / 2) / self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y - self.server.config.borderHeight / 2) / self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x - self.server.config.serverViewBaseX / 2) / self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y - self.server.config.serverViewBaseY / 2) / self.server.config.serverViewBaseY * 2  # [-1, 1]
            canRemerge = onehot(cell.canRemerge, ndim=2)  # len 2 onehot 0 or 1
            ismycell = onehot(cell.owner == player, ndim=2)# len 2 onehot 0 or 1
            canRemerge = max(canRemerge[0])
            ismycell = max(ismycell[0])
            features_player = np.array([[boost_x, boost_y, radius, log_radius, position_x, position_y, relative_position_x, relative_position_y, canRemerge, ismycell]])

            return cell.cellType, features_player
        elif cell.cellType == 1:
            # food features
            radius = (cell.radius - (self.server.config.foodMaxRadius + self.server.config.foodMinRadius) / 2) / (self.server.config.foodMaxRadius - self.server.config.foodMinRadius) * 2  # fixme
            log_radius = np.log(cell.radius / ((self.server.config.foodMaxRadius + self.server.config.foodMinRadius) / 2))  # fixme
            position_x = (cell.position.x - self.server.config.borderWidth / 2) / self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y - self.server.config.borderHeight / 2) / self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x - self.server.config.serverViewBaseX / 2) / self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y - self.server.config.serverViewBaseY / 2) / self.server.config.serverViewBaseY * 2  # [-1, 1]
            features_food = np.array([[radius, log_radius, position_x, position_y, relative_position_x, relative_position_y]])
            return cell.cellType, features_food

        elif cell.cellType == 2:
            # virus features
            boost_x = (cell.boostDistance * cell.boostDirection.x) / self.server.config.splitVelocity  # [-1, 1]
            boost_y = cell.boostDistance * cell.boostDirection.y / self.server.config.splitVelocity  # [-1, 1]
            radius = (cell.radius - (self.server.config.virusMaxRadius + self.server.config.virusMinRadius) / 2) / (self.server.config.virusMaxRadius - self.server.config.virusMinRadius) * 2  # fixme
            log_radius = np.log(cell.radius / ((self.server.config.virusMaxRadius + self.server.config.virusMinRadius) / 2))  # fixme
            position_x = (cell.position.x - self.server.config.borderWidth / 2) / self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y - self.server.config.borderHeight / 2) / self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x - self.server.config.serverViewBaseX / 2) / self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y - self.server.config.serverViewBaseY / 2) / self.server.config.serverViewBaseY * 2  # [-1, 1]
            features_virus = np.array([[boost_x, boost_y, radius, log_radius, position_x, position_y, relative_position_x, relative_position_y]])
            return cell.cellType, features_virus

        elif cell.cellType == 3:
            # ejected mass features
            boost_x = (cell.boostDistance * cell.boostDirection.x) / self.server.config.splitVelocity  # [-1, 1]
            boost_y = cell.boostDistance * cell.boostDirection.y / self.server.config.splitVelocity  # [-1, 1]
            position_x = (cell.position.x - self.server.config.borderWidth / 2) / self.server.config.borderWidth * 2  # [-1, 1]
            position_y = (cell.position.y - self.server.config.borderHeight / 2) / self.server.config.borderHeight * 2  # [-1, 1]
            relative_position_x = (cell.position.x - player.centerPos.x - self.server.config.serverViewBaseX / 2) / self.server.config.serverViewBaseX * 2  # [-1, 1]
            relative_position_y = (cell.position.y - player.centerPos.y - self.server.config.serverViewBaseY / 2) / self.server.config.serverViewBaseY * 2  # [-1, 1]
            features_food = np.array([[boost_x, boost_y, position_x, position_y, relative_position_x, relative_position_y]])
            return cell.cellType, features_food

    def parse_reward(self, player):
        mass_reward, kill_reward, killed_reward = self.calc_reward(player)
        # reward for being --- big, not dead, eating part of others, killing all of others, not be eaten by someone
        reward = mass_reward * self.mass_reward_eps + \
                 kill_reward * self.kill_reward_eps + \
                 killed_reward * self.killed_reward_eps
        return reward

    def calc_reward(self, player):
        mass_reward = sum([c.mass for c in player.cells])
        kill_reward = player.killreward
        killedreward = player.killedreward
        return mass_reward, kill_reward, killedreward

    def render(self, playeridx, mode = 'human'):
        # time.sleep(0.001)
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.server.config.serverViewBaseX, self.server.config.serverViewBaseY)
            self.render_border()
            self.render_grid()

        bound = self.players[playeridx].get_view_box()
        self.viewer.set_bounds(*bound)
        # self.viewer.set_bounds(-7000, 7000, -7000, 7000)

        self.geoms_to_render = []
        # self.viewNodes = sorted(self.viewNodes, key=lambda x: x.size)
        for node in self.players[playeridx].viewNodes:
            self.add_cell_geom(node)

        self.geoms_to_render = sorted(self.geoms_to_render, key=lambda x: x.order)
        for geom in self.geoms_to_render:
            self.viewer.add_onetime(geom)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_border(self):
        map_left = - self.server.config.borderWidth / 2
        map_right = self.server.config.borderWidth / 2
        map_top = - self.server.config.borderHeight / 2
        map_bottom = self.server.config.borderHeight / 2
        line_top = rendering.Line((map_left, map_top), (map_right, map_top))
        line_top.set_color(0, 0, 0)
        self.viewer.add_geom(line_top)
        line_bottom = rendering.Line((map_left, map_bottom), (map_right, map_bottom))
        line_bottom.set_color(0, 0, 0)
        self.viewer.add_geom(line_bottom)
        line_left = rendering.Line((map_left, map_top), (map_left, map_bottom))
        line_left.set_color(0, 0, 0)
        self.viewer.add_geom(line_left)
        map_right = rendering.Line((map_right, map_top), (map_right, map_bottom))
        map_right.set_color(0, 0, 0)
        self.viewer.add_geom(map_right)

    def render_grid(self):
        map_left = - self.server.config.borderWidth / 2
        map_right = self.server.config.borderWidth / 2
        map_top = - self.server.config.borderHeight / 2
        map_bottom = self.server.config.borderHeight / 2
        for i in range(0, int(map_right), 100):
            line = rendering.Line((i, map_top), (i, map_bottom))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)
            line = rendering.Line((-i, map_top), (-i, map_bottom))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)

        for i in range(0, int(map_bottom), 100):
            line = rendering.Line((map_left, i), (map_right, i))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)
            line = rendering.Line((map_left, -i), (map_right, -i))
            line.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(line)

    def add_cell_geom(self, cell):
        if cell.cellType == 0:
            cellwall = rendering.make_circle(radius=cell.radius)
            cellwall.set_color(cell.color.r * 0.75 / 255.0, cell.color.g * 0.75 / 255.0 , cell.color.b * 0.75 / 255.0)
            xform = rendering.Transform()
            cellwall.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            cellwall.order = cell.radius
            self.geoms_to_render.append(cellwall)

            geom = rendering.make_circle(radius=cell.radius - max(10, cell.radius * 0.1))
            geom.set_color(cell.color.r / 255.0, cell.color.g / 255.0, cell.color.b / 255.0)
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            if cell.owner.maxradius < self.server.config.virusMinRadius:
                geom.order = cell.owner.maxradius + 0.0001
            elif cell.radius < self.server.config.virusMinRadius:
                geom.order = self.server.config.virusMinRadius - 0.0001
            else: #cell.owner.maxradius < self.server.config.virusMaxRadius:
                geom.order = cell.owner.maxradius + 0.0001

            self.geoms_to_render.append(geom)

            # self.viewer.add_onetime(geom)
        elif cell.cellType == 2:
            geom = rendering.make_circle(radius=cell.radius)
            geom.set_color(cell.color.r / 255.0, cell.color.g / 255.0, cell.color.b / 255.0, 0.6)
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            geom.order = cell.radius
            self.geoms_to_render.append(geom)

        else:
            geom = rendering.make_circle(radius=cell.radius)
            geom.set_color(cell.color.r / 255.0, cell.color.g / 255.0, cell.color.b / 255.0)
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(cell.position.x, cell.position.y)
            geom.order = cell.radius
            self.geoms_to_render.append(geom)


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def print_cell_data(self, curr_player):
        print("Boost X = " + str(curr_player[0]))
        print("Boost Y = " + str(curr_player[1]))
        print("Radius = " + str(curr_player[2]))
        print("Log Radius = " + str(curr_player[3]))
        print("Position X = " + str(curr_player[4]))
        print("Position Y = " + str(curr_player[5]))
        print("Relative Position X = " + str(curr_player[6]))
        print("Relative Position Y = " + str(curr_player[7]))
        print("Can Remerge = " + str(curr_player[8]))
        print("Is My Cell = " + str(curr_player[9]))
        print("")

    def print_food_data(self, curr_food):
        print("Radius = " + str(curr_food[0]))
        print("Log Radius = " + str(curr_food[1]))
        print("Position X = " + str(curr_food[2]))
        print("Position Y = " + str(curr_food[3]))
        print("Relative Position X = " + str(curr_food[4]))
        print("Relative Position Y = " + str(curr_food[5]))
        print("")

    def print_virus_data(self, curr_virus):
        print("Boost X = " + str(curr_virus[0]))
        print("Boost Y = " + str(curr_virus[1]))
        print("Radius = " + str(curr_virus[2]))
        print("Log Radius = " + str(curr_virus[3]))
        print("Position X = " + str(curr_virus[4]))
        print("Position Y = " + str(curr_virus[5]))
        print("Relative Position X = " + str(curr_virus[6]))
        print("Relative Position Y = " + str(curr_virus[7]))
        print("")

    def split_observation(self, observation):
        # Returns only the smallest cell that is not your own cell
        # Returns the number of food cells and the closest one to you
        player = observation['player']
        food = observation['food']
        virus = observation['virus']
        ejected = observation['ejected']
        player_size, player_coord_x, player_coord_y = self.get_current_player(player)
        curr_player_coords = (player_coord_x, player_coord_y)
        count_cells, cell_coordinate_x, cell_coordinate_y, size = self.get_closest_cell(player, curr_player_coords)
        food_coordinate_x, food_coordinate_y = self.get_closest_food(food, curr_player_coords)
        virus_coordinate_x, virus_coordinate_y = self.get_closest_virus(virus, curr_player_coords)
        if ejected is not None:

            if player_coord_x is not None and player_coord_y is not None :
                return player_size, player_coord_x, player_coord_y, count_cells, cell_coordinate_x, cell_coordinate_y, size, food_coordinate_x, food_coordinate_y, virus_coordinate_x, virus_coordinate_y,  ejected[2], ejected[3]
            else:
                return player_size, player_coord_x, player_coord_y, count_cells, cell_coordinate_x, cell_coordinate_y, size, food_coordinate_x, food_coordinate_y, virus_coordinate_x, virus_coordinate_y, ejected[2], ejected[3]
        else:
            if player_coord_x is not None and player_coord_y is not None:
                return player_size, player_coord_x, player_coord_y, count_cells, cell_coordinate_x, cell_coordinate_y, size, food_coordinate_x, food_coordinate_y, virus_coordinate_x, virus_coordinate_y, player_coord_x, player_coord_y
            else:
                return player_size, player_coord_x, player_coord_y, count_cells, cell_coordinate_x, cell_coordinate_y, size, food_coordinate_x, food_coordinate_y, virus_coordinate_x, virus_coordinate_y, player_coord_x, player_coord_y

    def get_current_player(self, players):
        for player in players:
            player = player[0]
            if int(player[9]) == 1:
                #return absolute coordinates
                return player[2], player[4], player[5]
        return None, None, None
            
    def get_closest_cell(self, players, curr_player_coords):
        closest_distance = math.inf
        size = 0
        count_cells = 0
        coordinates_x = 0
        coordinates_y = 0
        if len(players) != 0 and players[0] is not None and curr_player_coords[0] is not None and curr_player_coords[1] is not None:
            for cell in players[0]:
                if cell[9] == 0:
                    count_cells += 1
                    distance = np.sqrt((cell[4] - curr_player_coords[0]) ** 2 + (cell[5] - curr_player_coords[1])** 2)
                    if distance < closest_distance:
                        coordinates_x = cell[4]
                        coordinates_y = cell[5]
                        closest_distance = distance
                        size = cell[2]
        return count_cells, coordinates_x,coordinates_y, size
    
    def get_closest_food(self, foods, curr_player_coords):
        closest_food = None
        closest_distance = math.inf
        count_food = 0
        coordinates_x = 0
        coordinates_y = 0
        if foods is not None and curr_player_coords[0] is not None and curr_player_coords[1] is not None:
            for food in foods:
                distance = np.sqrt((food[2] - curr_player_coords[0]) ** 2 + (food[3] - curr_player_coords[1]) ** 2)
                count_food += 1
                if distance < closest_distance:
                    coordinates_x = food[2]
                    coordinates_y = food[3]
                    closest_distance = distance
            return coordinates_x,coordinates_y
        else:
            return 0, 0
    def get_closest_virus(self, viruses, curr_player_coords):
        closest_distance = math.inf
        count_virus = 0
        coordinates_x = 0
        coordinates_y = 0

        if viruses is not None and curr_player_coords[0] is not None and curr_player_coords[1] is not None:
            for virus in viruses:
                distance = np.sqrt((virus[4] - curr_player_coords[0]) ** 2 + (virus[5] - curr_player_coords[1]) ** 2)
                count_virus += 1
                if distance < closest_distance:
                    coordinates_x = virus[4]
                    coordinates_y = virus[5]
                    closest_distance = distance
            return coordinates_x,coordinates_y
        else:
            return 0,0
def onehot(d, ndim):
    v = np.zeros((1, ndim))
    v[0, d] = 1
    return v
