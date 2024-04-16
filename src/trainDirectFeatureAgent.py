from pyglet.window import key
import numpy as np
from Env import AgarEnv
from models.DirectFeaturePPOModel import DirectFeaturePPOModel
from models.ReplayBuffer import ReplayBuffer
import torch
import math 
from trainingConfig import trainingConfig
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import pydot

def checkIfAgentIsRemoved(obs: dict):
    playerObs = obs['player']
    for player in playerObs:
        if player[3]:
            return False
    return True

def getAgentActionVec(action: torch.Tensor, prevActionVec: np.ndarray, numDirections: int = 16):
    action = int(action.cpu().numpy())
    actionVec = prevActionVec

    if action == numDirections:
        # Split action
        # actionVec[2] = 1
        actionVec[2] = 0
    elif action == numDirections + 1:
        # Eject action
        # actionVec[2] = 2
        actionVec[2] = 0
    else:
        # Movement
        actionVec[2] = 0
        angle = 2 * np.pi / numDirections * action
        actionVec[0] = math.cos(angle)
        actionVec[1] = math.sin(angle)

    return actionVec

# Config environment
render = True
env = AgarEnv(
    num_agents = 1, 
    num_bots = trainingConfig.numBots, 
    gamemode = 0
)
env.configureRewardCoeffs(
    trainingConfig.massRewardCoeff,
    trainingConfig.killRewardCoeff,
    trainingConfig.consumptionRewardCoeff,
    trainingConfig.killedPenaltyCoeff,
    trainingConfig.deadPenalty,
    trainingConfig.passivePenalty
)

window = None
playerActionVec = np.zeros((1, 3))
# env.seed(0)

# USE GPU?
device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")

# Load model
model = DirectFeaturePPOModel(
    numDirections = trainingConfig.numDirections,
    usePrevFrame = trainingConfig.usePrevFrame,
    usePrevAction = trainingConfig.usePrevAction
)

if not trainingConfig.startFromScratch:
    model.loadFromFile(trainingConfig.modelLoadPath)
model = model.to(device)

# Prepare replay buffer
buffer = ReplayBuffer(
    model.featureDims, 
    model.numActions, 
    device, 
    bufferSize = trainingConfig.replayBufferSize, 
    gamma = trainingConfig.gamma
)

# Initialize optimizer
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = trainingConfig.learningRate
)

# Initialize training variables
agent = None
resetEnvironment = True
ejectCooldown = None
value = None
currStateEncodings = None
fullStateEncodings = None
action = None
probs = None
logProb = None 
entropy = None
rewards = None
wasTerminal = False
observations = None
gameNumber = 1

# Initialize tensorboard for tracking
writer = SummaryWriter()
totalReward = 0

stepNum = 0
totalSteps = 0
for iterNum in range(trainingConfig.numIters):
    with torch.no_grad():
        model.eval()
        while not buffer.isFilled():
            stepNum += 1
            totalSteps += 1
            if totalSteps % 100 == 0:
                print(f'Iter {iterNum + 1}, totalSteps {totalSteps + 1}')

            if (resetEnvironment):
                # Reset the environment
                print("Resetting environment...")
                observations = env.reset().obs[0]
                agent = env.players[0]
                ejectCooldown = 0
                env.render(0)
                
                currStateEncodings = model.getSingleFrameEncoding(observations, ejectCooldown, device)
                prevStateEncodings = currStateEncodings.clone() # Assume that the prev state is the same when starting out
                prevAction = torch.tensor([0]).to(device) # Assume that the prev action is 0
                fullStateEncodings = model.getFullStateEncoding(prevStateEncodings, currStateEncodings, prevAction)
                action, logProb, entropy = model.getAction(fullStateEncodings)
                value = model.getValue(fullStateEncodings)

                playerActionVec[0] = getAgentActionVec(action, playerActionVec[0])
                agarObs, rewards, done, info = env.step(playerActionVec)
                observations = agarObs.obs[0]
                resetEnvironment = checkIfAgentIsRemoved(observations)
                continue

            # Render a new window if need be
            env.render(0)
            if not window:
                window = env.viewer.window
            
            if checkIfAgentIsRemoved(observations) or stepNum == trainingConfig.maxSteps:
                print("Agent died or stepNum == maxSteps! Resetting environment")
                resetEnvironment = True
                window.close()
                window = None

                # Write to tensorboard
                writer.add_scalar("totalReward/gameNumber", totalReward, gameNumber)
                writer.flush()
                totalReward = 0
                gameNumber += 1

            # Abit confusing, but nextFullStateEncodings is the state encodings of this state. The reason why we have the
            # word next is cos we need to add the buffer entry of the PREVIOUS step, which relies on the value of this state
            ejectCooldown = env.server.getEjectCooldown(agent)
            if not resetEnvironment:
                nextStateEncodings = model.getSingleFrameEncoding(observations, ejectCooldown, device)
                nextFullStateEncodings = model.getFullStateEncoding(currStateEncodings, nextStateEncodings, action)
                nextAction, nextLogProb, nextEntropy = model.getAction(nextFullStateEncodings)
                nextValue = model.getValue(nextFullStateEncodings)

            # Add to replay buffer
            buffer.addEntry(
                fullStateEncodings, # Frm prev iter
                action, # Frm prev iter
                logProb, # Frm prev iter
                value, # Frm prev iter
                nextValue, # Frm curr iter, the value of curr state to calculate advantage
                torch.tensor([rewards[0]]).to(device), # Frm prev iter
                resetEnvironment
            )
            
            currStateEncodings = nextStateEncodings
            fullStateEncodings = nextFullStateEncodings
            action = nextAction
            logProbs = nextLogProb
            value = nextValue
            entropy = nextEntropy

            # Get the player action vec, and step the environment to get the rewards for this iteration
            playerActionVec[0] = getAgentActionVec(action, playerActionVec[0])
            agarObs, rewards, done, info = env.step(playerActionVec)
            observations = agarObs.obs[0]
            totalReward += rewards[0]


    # MODEL OPTIMIZATION
    print("OPTIMIZING")
    model.train()
    for epochNum in range(trainingConfig.numEpochs):
        indices = np.arange(trainingConfig.replayBufferSize)

        totalPolicyLoss = 0
        totalValueLoss = 0
        totalLoss = 0
        for startIndice in range(0, trainingConfig.replayBufferSize, trainingConfig.batchSize):
            optimizer.zero_grad()
            
            endIndice = startIndice + trainingConfig.batchSize
            batchIndices = indices[startIndice: endIndice]

            newLogProbs, newEntropy = model.getLogProbGivenAction(
                buffer.encodedObservations[batchIndices],
                buffer.actions[batchIndices]
            )
            logratio = newLogProbs - buffer.logProbs[batchIndices]
            ratio = torch.exp(logratio)
            advantages = buffer.advantages[batchIndices]
            normalizedAdvantages = torch.nn.functional.normalize(advantages, dim = 0, eps = 1e-8)

            # Calculate policy loss. It is negated as we want to maximize instead of minimize
            policyLoss1 = normalizedAdvantages * ratio
            policyLoss2 = normalizedAdvantages * torch.clamp(
                ratio,
                1 - trainingConfig.EPSClip,
                1 + trainingConfig.EPSClip
            )
            policyLoss = -torch.mean(torch.minimum(policyLoss1, policyLoss2))

            # Calculate value loss
            returns = advantages + buffer.values[batchIndices]
            newValues = model.getValue(buffer.encodedObservations[batchIndices])
            valueLossUnclipped = torch.square(newValues - returns)

            newValuesClipped = buffer.values[batchIndices] + torch.clamp(
                newValues - buffer.values[batchIndices],
                -trainingConfig.EPSClip,
                trainingConfig.EPSClip
            )
            valueLossClipped = torch.square(newValuesClipped - returns)
            valueLoss = torch.mean(torch.maximum(valueLossUnclipped, valueLossClipped))

            # Calculate entropy loss
            entropyLoss = torch.mean(newEntropy)

            # Combine them losses
            loss = policyLoss - trainingConfig.entropyCoeff * entropyLoss + valueLoss * trainingConfig.valueCoeff

            loss.backward()

            with torch.no_grad():
                totalPolicyLoss += policyLoss
                totalValueLoss += valueLoss
                totalLoss += loss

            torch.nn.utils.clip_grad_norm(model.actor.parameters(), trainingConfig.maxGradNorm)
            optimizer.step()

        with torch.no_grad():
            divFactor = trainingConfig.replayBufferSize / trainingConfig.batchSize

            writer.add_scalar(f"iter{iterNum}-meanPolicyLoss/epoch", totalPolicyLoss/divFactor, epochNum)
            writer.add_scalar(f"iter{iterNum}-meanValueLoss/epoch", totalValueLoss/divFactor, epochNum)
            writer.add_scalar(f"iter{iterNum}-meanloss/epoch", totalLoss/divFactor, epochNum)
            writer.flush()

    # Save latest model to disk
    fileName = "latest"
    filePath = trainingConfig.modelSaveDir + "/latest.pt" 
    torch.save(model.state_dict(), filePath)

    buffer.reset()
    
writer.close()
env.close()