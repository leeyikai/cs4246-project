import math
from Config import Config
import random
from modules import *
from entity import *
from gamemodes import *


# noinspection PyAttributeOutsideInit
class GameServer:
    def __init__(self):
        self.srcFiles = "../src"

        # Startup
        self.run = True
        self.version = '1.6.1'
        self.httpServer = None
        self.lastNodeId = 1
        self.lastPlayerId = 1
        self.players = []
        self.socketCount = 0
        self.largestPlayer = None
        self.nodes = []  # Total nodes
        self.nodesVirus = []  # Virus nodes
        self.nodesFood = []  # Food nodes
        self.nodesEjected = []  # Ejected nodes
        self.nodesPlayer = []  # Player nodes

        self.movingNodes = []  # For move engine
        self.leaderboard = []  # For leaderboard
        self.leaderboardType = -1  # No type

        # Main loop tick
        self.stepDateTime = 0
        self.timeStamp = 0
        self.updateTime = 0
        self.updateTimeAvg = 0
        self.timerLoopBind = None
        self.mainLoopBind = None
        self.tickCounter = 0
        self.disableSpawn = False

        # Config
        self.config = Config()

        self.ipBanList = []
        self.minionTest = []
        self.userList = []
        self.badWords = []

        # Set border, quad-tree
        self.setBorder(self.config.borderWidth, self.config.borderHeight)
        self.quadTree = QuadNode(self.border)

    def start(self, gamemode=1):
        # Set up gamemode(s)
        self.gameMode = Get_Game_Mode(gamemode)
        self.gameMode.onServerInit(self)

    def addPlayers(self, players):
        self.players = players

    def addNode(self, node):
        # Add to quad-tree & node list
        x = node.position.x
        y = node.position.y
        s = node.size
        node.quadItem = QuadItem(node, Bound(x - s, y - s, x + s, y + s))
        self.quadTree.insert(node.quadItem)
        self.nodes.append(node)
        # Special on-add actions
        node.onAdd(self)

    def setBorder(self, width, height):
        hw = width / 2
        hh = height / 2
        self.border = Bound(-hw, -hh, hw, hh)

    @staticmethod
    def getRandomColor():
        colorRGB = [0xff, 0x07, random.randint(0, 256)]
        random.shuffle(colorRGB)
        # return random
        return Color(*colorRGB)

    def removeNode(self, node):
        # Remove from quad-tree
        node.isRemoved = True
        self.quadTree.remove(node.quadItem)
        node.quadItem = None

        # Remove from node lists
        i = self.nodes.index(node)
        if i > -1:
            self.nodes.pop(i)
            i = self.movingNodes.index(node)
        if i > -1:
            self.movingNodes.pop(i)

        # Special on-remove actions
        node.onRemove(self)

    def updatePlayers(self):
        # check dead players
        i = 0
        while i < len(self.players):
            if not self.players[i]:
                i += 1
                continue

            if self.players[i].isRemoved:
                # remove dead player
                self.players.pop(i)
            else:
                i += 1
        # update
        for player in self.players:
            if not player:
                continue
            player.updateTick()

        # for player in self.players:
        #     if not player:
        #         continue
        #     player.playerTracker.sendUpdate()

    def Update(self):
        skipstep = 1
        for i in range(skipstep):
            self.WorldStep()

    def WorldStep(self):
        # Loop main functions
        if self.run:
            # Move moving nodes first
            for cell in self.movingNodes:
                if cell.isRemoved:
                    return
                # Scan and check for ejected mass / virus collisions
                self.boostCell(cell)

                def callback_fun(check):
                    collision = self.checkCellCollision(cell, check)
                    if cell.cellType == 3 and check.cellType == 3 and not self.config.mobilePhysics:
                        self.resolveRigidCollision(collision)
                    else:
                        self.resolveCollision(collision)

                self.quadTree.find(cell.quadItem.bound, callback_fun)
                if not cell.isMoving:
                    self.movingNodes = None

            # Update players and scan for collisions
            eatCollisions = []
            for cell in self.nodesPlayer:
                if cell.isRemoved:
                    return

                # Scan for eat/rigid collisions and resolve them
                def callback_fun(check):
                    collision = self.checkCellCollision(cell, check)
                    if self.checkRigidCollision(collision):
                        self.resolveRigidCollision(collision)
                    elif check != cell:
                        eatCollisions.insert(0, collision)

                self.quadTree.find(cell.quadItem.bound, callback_fun)
                self.movePlayer(cell, cell.owner)
                self.boostCell(cell)
                self.autoSplit(cell, cell.owner)
                # Decay player cells once per second
                if ((self.tickCounter + 3) % 25) == 0:
                    self.updateSizeDecay(cell)
                # Remove external minions if necessary
                if cell.owner.isMinion:
                    cell.owner.socket.close(1000, "Minion")
                    self.removeNode(cell)

            for m in eatCollisions:
                self.resolveCollision(m)

            if (self.tickCounter % self.config.spawnInterval) == 0:
                # Spawn food & viruses
                self.spawnCells()

            self.gameMode.onTick(self)
            self.tickCounter += 1

        if not self.run and self.gameMode.IsTournament:
            self.tickCounter += 1
        self.updatePlayers()

    # update remerge first
    def movePlayer(self, cell, player):
        if not player.socket.isConnected or player.frozen or not player.mouse:
            return  # Do not move

        # get movement from vector
        d = player.mouse.clone().sub(cell.position)
        move = cell.getSpeed(d.sqDist())  # movement speed
        if not move:
            return  # avoid jittering
        cell.position.add(d, move)

        # update remerge
        time = self.config.playerRecombineTime,
        base = max(time, cell.size * 0.2) * 25
        # instant merging conditions
        if not time or player.rec or player.mergeOverride:
            cell._canRemerge = cell.boostDistance < 100
            return  # instant merge

        # regular remerge time
        cell._canRemerge = cell.getAge() >= base

    # decay player cells
    def updateSizeDecay(self, cell):
        rate = self.config.playerDecayRate
        cap = self.config.playerDecayCap

        if not rate or cell.size <= self.config.playerMinSize:
            return

        # remove size from cell at decay rate
        if cap and cell.mass > cap:
            rate *= 10
        decay = 1 - rate * self.gameMode.decayMod
        cell.setSize(math.sqrt(cell.radius * decay))

    def boostCell(self, cell):
        if cell.isMoving and not cell.boostDistance or cell.isRemoved:
            cell.boostDistance = 0
            cell.isMoving = False
            return
        # decay boost-speed from distance
        speed = cell.boostDistance / 9  # val: 87
        cell.boostDistance -= speed  # decays from speed
        cell.position.add(cell.boostDirection, speed)

        # update boundries
        cell.checkBorder(self.border)
        self.updateNodeQuad(cell)

    def autoSplit(self, cell, player):
        # get size limit based off of rec mode
        if player.rec:
            maxSize = 1e9  # increase limit for rec (1 bil)
        else:
            maxSize = self.config.playerMaxSize

        # check size limit
        if player.mergeOverride or cell.size < maxSize:
            return
        if len(player.cells) >= self.config.playerMaxCells or self.config.mobilePhysics:
            # cannot split => just limit
            cell.setSize(maxSize)
        else:
            # split in random direction
            angle = random.random() * 2 * math.pi
            self.splitPlayerCell(player, cell, angle, cell.mass * .5)

    def updateNodeQuad(self, node):
        # update quad tree
        item = node.quadItem.bound
        item.minx = node.position.x - node.size
        item.miny = node.position.y - node.size
        item.maxx = node.position.x + node.size
        item.maxy = node.position.y + node.size
        self.quadTree.remove(node.quadItem)
        self.quadTree.insert(node.quadItem)

    # Checks cells for collision
    @staticmethod
    def checkCellCollision(cell, check):
        p = check.position.clone().sub(cell.position)
        # create collision manifold
        return Collision(cell, check, p.sqDist(), p)

    # Checks if collision is rigid body collision
    def checkRigidCollision(self, m):
        if not m.cell.owner or not m.check.owner:
            return False

        if m.cell.owner != m.check.owner:
            # Minions don't collide with their team when the config value is 0
            if self.gameMode.haveTeams and m.check.owner.isMi or m.cell.owner.isMi and self.config.minionCollideTeam == 0:
                return False
            else:
                # Different owners => same team
                return self.gameMode.haveTeams and m.cell.owner.team == m.check.owner.team

        r = 1 if self.config.mobilePhysics else 13
        if m.cell.getAge() < r or m.check.getAge() < r:
            return False  # just splited => ignore

        return not m.cell.canRemerge or not m.check.canRemerge

    # Resolves rigid body collisions
    @staticmethod
    def resolveRigidCollision(m):
        push = (m.cell.size + m.check.size - m.d) / m.d
        if push <= 0 or m.d == 0:
            return  # do not extrude

        # body impulse
        rt = m.cell.radius + m.check.radius
        r1 = push * m.cell.radius / rt
        r2 = push * m.check.radius / rt

        # apply extrusion force
        m.cell.position.sub2(m.p, r2)
        m.check.position.add(m.p, r1)

    # Resolves non-rigid body collision
    def resolveCollision(self, m):
        cell = m.cell
        check = m.check
        if cell.size > check.size:
            cell = m.check
            check = m.cell

        # Do not resolve removed
        if cell.isRemoved or check.isRemoved:
            return

        # check eating distance
        check.div = 20 if self.config.mobilePhysics else 3
        if m.d >= check.size - cell.size / check.div:
            return  # too far => can't eat

        # collision owned => ignore, resolve, or remerge
        if cell.owner and cell.owner == check.owner:
            if cell.getAge() < 13 or check.getAge() < 13:
                return  # just splited => ignore
        elif check.size < cell.size * 1.15 or not check.canEat(cell):
            return  # Cannot eat or cell refuses to be eaten

        # Consume effect
        check.onEat(cell)
        cell.onEaten(check)
        cell.killedBy = check

        # Remove cell
        self.removeNode(cell)

    def splitPlayerCell(self, player, parent, angle, mass):
        size = math.sqrt(mass * 100)
        size1 = math.sqrt(parent.radius - size * size)

        # Too small to split
        if not size1 or size1 < self.config.playerMinSize:
            return

        # Remove size from parent cell
        parent.setSize(size1)

        # Create cell and add it to node list
        newCell = PlayerCell(self, player, parent.position, size)
        newCell.setBoost(self.config.splitVelocity * math.pow(size, 0.0122), angle)
        self.addNode(newCell)

    def randomPos(self):
        return Vec2(self.border.minx + self.border.width * random.random(),
                    self.border.miny + self.border.height * random.random())

    def spawnCells(self):
        # spawn food at random size
        maxCount = self.config.foodMinAmount - len(self.nodesFood)
        spawnCount = min(maxCount, self.config.foodSpawnAmount)
        for i in range(spawnCount):
            cell = Food(self, None, self.randomPos(), self.config.foodMinSize)
            if self.config.foodMassGrow:
                maxGrow = self.config.foodMaxSize - cell.size
                cell.setSize(cell.size + maxGrow * random.random())

            cell.color = self.getRandomColor()
            self.addNode(cell)

        # spawn viruses (safely)
        if len(self.nodesVirus) < self.config.virusMinAmount:
            virus = Virus(self, None, self.randomPos(), self.config.virusMinSize)
            if not self.willCollide(virus):
                self.addNode(virus)

    def spawnPlayer(self, player, pos):
        if self.disableSpawn:
            return

        # Check for special starting size
        size = self.config.playerStartSize
        if player.spawnmass:
            size = player.spawnmass

        # Check if can spawn from ejected mass
        if self.nodesEjected:
            eject = random.choice(self.nodesEjected) # Randomly selected
            if random.random() <= self.config.ejectSpawnPercent and eject and eject.boostDistance < 1:
                # Spawn from ejected mass
                pos = eject.position.clone()
                player.color = eject.color
                size = max(size, eject.size * 1.15)

        # Spawn player safely (do not check minions)
        cell = PlayerCell(self, player, pos, size)
        if self.willCollide(cell) and not player.isMi:
            pos = self.randomPos()  # Not safe => retry
        self.addNode(cell)

        # Set initial mouse coords
        player.mouse = pos

    def willCollide(self, cell):
        notSafe = False  # Safe by default
        sqSize = cell.radius
        pos = self.randomPos()
        d = cell.position.clone().sub(pos)
        if d.dist() + sqSize <= sqSize * 2:
            notSafe = True

        def callback_fun(n):
            nonlocal notSafe
            if n.cellType == 0:
                notSafe = True

        self.quadTree.find(Bound(cell.position.x - cell.size, cell.position.y - cell.size, cell.position.x + cell.size,
                                 cell.position.y + cell.size), callback_fun)

        return notSafe

    def splitCells(self, player):
        # Split cell order decided by cell age
        cellToSplit = [cell for cell in player.cells]

        for cell in cellToSplit:
            d = player.mouse.clone().sub(cell.position)
            if d.dist() < 1:
                d.x = 1
                d.y = 0

            if cell.size < self.config.playerMinSplitSize:
                return  # cannot split

            # Get maximum cells for rec mode
            if player.rec:
                max_cell_rec = 200  # rec limit
            else:
                max_cell_rec = self.config.playerMaxCells
            if len(player.cells) >= max_cell_rec:
                return

            # Now split player cells
            self.splitPlayerCell(player, cell, d.angle(), cell.mass * .5)

    def canEjectMass(self, player):
        if player.lastEject is None:
            # first eject
            player.lastEject = self.tickCounter
            return True

        dt = self.tickCounter - player.lastEject
        if dt < self.config.ejectCooldown:
            # reject (cooldown)
            return False

        player.lastEject = self.tickCounter
        return True

    def ejectMass(self, player):
        if not self.canEjectMass(player) or player.frozen:
            return
        for cell in player.cells:
            if cell.size < self.config.playerMinEjectSize:
                continue  # Too small to eject

            d = player.mouse.clone().sub(cell.position)
            sq = d.sqDist()
            d.x = d.x / sq if sq > 1 else 1
            d.y = d.y / sq if sq > 1 else 0

            # Remove mass from parent cell first
            loss = self.config.ejectSizeLoss
            loss = cell.radius - loss * loss
            cell.setSize(math.sqrt(loss))

            # Get starting position
            pos = Vec2(cell.position.x + d.x * cell.size, cell.position.y + d.y * cell.size)
            angle = d.angle() + (random.random() * .6) - .3

            # Create cell and add it to node list
            if not self.config.ejectVirus:
                ejected = EjectedMass(self, None, pos, self.config.ejectSize)
            else:
                ejected = Virus(self, None, pos, self.config.ejectSize)

            ejected.color = cell.color
            ejected.setBoost(self.config.ejectVelocity, angle)
            self.addNode(ejected)

    def shootVirus(self, parent, angle):
        # Create virus and add it to node list
        pos = parent.position.clone()
        newVirus = Virus(self, None, pos, self.config.virusMinSize)
        newVirus.setBoost(self.config.virusVelocity, angle)
        self.addNode(newVirus)