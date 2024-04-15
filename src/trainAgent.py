from pyglet.window import key
import numpy as np
from Env import AgarEnv
from models.PPOModel import PPOModel
from models.ReplayBuffer import ReplayBuffer
import torch
import math 
from trainingConfig import trainingConfig

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
window = None
playerActionVec = np.zeros((1, 3))
# env.seed(0)

# USE GPU?
device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")

# Load model
model = None
if trainingConfig.startFromScratch:
    model = PPOModel()
else:
    model = PPOModel.fromFile(trainingConfig.modelLoadDir)
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
prevValue = None
stateEncodings = None
action = None
probs = None
logProb = None 
entropy = None
rewards = None
wasTerminal = False

stepNum = 0
for iterNum in range(trainingConfig.numIters):
    with torch.no_grad():
        model.eval()
        while not buffer.isFilled():
            stepNum += 1
            if stepNum % 32 == 0:
                print(f'Iter {iterNum + 1}, step {stepNum + 1}')

            if (resetEnvironment):
                # Reset the environment
                print("Resetting environment...")
                env.reset()
                agent = env.players[0]
                ejectCooldown = 0
                view = env.render(0, mode = "rgb_array")
                
                stateEncodings = model.getEncoding(view, ejectCooldown, device)
                action, logProb, entropy = model.getAction(stateEncodings)
                value = model.getValue(stateEncodings)

                playerActionVec[0] = getAgentActionVec(action, playerActionVec[0])
                observations, rewards, done, info = env.step(playerActionVec)
                resetEnvironment = agent.isRemoved
                continue

            # PPO AGENT CODE HERE
            view = env.render(0, mode = "rgb_array")
            if not window:
                window = env.viewer.window
            
            ejectCooldown = env.server.getEjectCooldown(agent)
            nextStateEncodings = model.getEncoding(view, ejectCooldown, device)
            nextAction, nextLogProb, nextEntropy = model.getAction(stateEncodings)
            nextValue = model.getValue(stateEncodings)

            if agent.isRemoved:
                print("Agent died! Resetting environment")
                resetEnvironment = True
                window.close()
            buffer.addEntry(
                stateEncodings, # Frm prev iter
                action, # Frm prev iter
                logProb, # Frm prev iter
                value, # Frm prev iter
                nextValue, # Frm curr iter, the value of curr state to calculate advantage
                torch.tensor([rewards[0]]).to(device), # Frm prev iter
                resetEnvironment
            )

            stateEncodings = nextStateEncodings
            action = nextAction
            logProbs = nextLogProb
            value = nextValue
            entropy = nextEntropy

            playerActionVec[0] = getAgentActionVec(action, playerActionVec[0])
            observations, rewards, done, info = env.step(playerActionVec)


    # MODEL OPTIMIZATION
    print("OPTIMIZING")
    model.train()
    for epochNum in range(trainingConfig.numEpochs):
        indices = np.arange(trainingConfig.replayBufferSize)

        for startIndice in range(0, trainingConfig.replayBufferSize, trainingConfig.batchSize):
            endIndice = startIndice + trainingConfig.batchSize
            batchIndices = indices[startIndice: endIndice]

            newLogProbs, newEntropy = model.getLogProbGivenAction(
                buffer.encodedObservations[batchIndices],
                buffer.actions[batchIndices]
            )
            logratio = newLogProbs - buffer.logProbs[batchIndices]
            ratio = logratio.exp()
            advantages = buffer.advantages[batchIndices]
            normalizedAdvantages = torch.nn.functional.normalize(advantages, dim = 0, eps = 1e-8)

            # Calculate policy loss. It is negative as we want to maximize instead of minimize
            policyLoss1 = -normalizedAdvantages * ratio
            policyLoss2 = -normalizedAdvantages * torch.clamp(
                ratio,
                1 - trainingConfig.EPSClip,
                1 + trainingConfig.EPSClip
            )
            policyLoss = torch.mean(torch.maximum(policyLoss1, policyLoss2))

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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.actor.parameters(), trainingConfig.maxGradNorm)
            optimizer.step()
    
    # Save latest model to disk
    fileName = "latest"
    filePath = trainingConfig.modelSaveDir + "/latest.pt" 
    torch.save(model.state_dict(), filePath)

    buffer.reset()
    

env.close()