from modules import *
from entity import *
from gamemodes import *
import math
import numpy as np

class Player:
    def __init__(self, gameServer, name = 'dummy'):
        self.gameServer = gameServer
        self.name = name
        self.mouse = Vec2(0, 0)
        self.centerPos = Vec2(0, 0)
        self.cells = []
        self.frozen = False
        self.mergeOverride = False
        self.spectate = False
        self.score = 0
        self.isRemoved = False
        self.spawnmass = 0
        if gameServer:
            gameServer.lastPlayerId += 1
            self.pID = gameServer.lastPlayerId
            gameServer.gameMode.onPlayerInit(self)
            self.joinGame()

    def step(self, action):
        if self.isRemoved:
            return
        # action in format [0] mouse x, [1 mouse y, [2] key space bool, [3] key w bool, [4] no key bool
        self.mouse = Vec2(action[0], action[1])
        # assert np.sum(action[2:]) == 1
        if action[2] == 1:
            self.pressSpace()
        elif action[3] == 1:
            self.pressW()

    def updateTick(self):
        self.updateSpecView()
        scale = max(self.getScale(), self.gameServer.config.serverMinScale)
        halfWidth = (self.gameServer.config.serverViewBaseX + 100) / scale / 2
        halfHeight = (self.gameServer.config.serverViewBaseY + 100) / scale / 2
        self.viewBox = Bound(
            self.centerPos.x - halfWidth,
            self.centerPos.y - halfHeight,
            self.centerPos.x + halfWidth,
            self.centerPos.y + halfHeight)

        self.viewNodes = []
        self.gameServer.quadTree.find(self.viewBox, lambda check: self.viewNodes.push(check))
        self.viewNodes = sorted(self.viewNodes, lambda x: x.nodeId)

    def updateSpecView(self):
        cx = 0
        cy = 0
        for cell in self.cells:
            cx += cell.position.x
            cy += cell.position.y
        self.centerPos = Vec2(cx / len(self.cells), cy / len(self.cells))

    def pressSpace(self):
        if self.gameServer.run:
            if len(self.cells) <= 2:
                self.mergeOverride = False
            if self.mergeOverride or self.frozen:
                return
        self.gameServer.splitCells(self)

    def pressW(self):
        if self.spectate or not self.gameServer.run:
            return
        self.gameServer.ejectMass(self)

    def setCenterPos(self, p):
        p.x = max(p.x, self.gameServer.border.minx)
        p.y = max(p.y, self.gameServer.border.miny)
        p.x = min(p.x, self.gameServer.border.maxx)
        p.y = min(p.y, self.gameServer.border.maxy)
        self.centerPos = p

    def getScale(self):
        scale = 0
        for cell in self.cells:
            scale += cell.size
            self.score += cell.mass
        if scale == 0:
            self.scale = 0.4
        else:
            self.scale = math.pow(min(64 / scale, 1), 0.4)
        return self.scale

    def joinGame(self):
        if self.cells:
            return
        self.gameServer.gameMode.onPlayerSpawn(self.gameServer, self)



