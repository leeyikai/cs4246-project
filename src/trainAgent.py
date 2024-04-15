from pyglet.window import key
import numpy as np
from Env import AgarEnv
import time
from models.PPOModel import PPOModel
from models.ReplayBuffer import ReplayBuffer
import torch
import math 

def getAgentActionVec(action: torch.Tensor, prevActionVec: np.ndarray, numDirections: int = 16):
    action = int(action.cpu().numpy())
    actionVec = prevActionVec

    if action == numDirections:
        # Split action
        actionVec[2] = 1
    elif action == numDirections + 1:
        # Eject action
        actionVec[2] = 2
    else:
        # Movement
        actionVec[2] = 0
        angle = 2 * np.pi / numDirections * action
        actionVec[0] = math.cos(angle)
        actionVec[1] = math.sin(angle)

    return actionVec

render = True
num_agents = 1
num_bots = 20
gamemode = 0
env = AgarEnv(num_agents, num_bots, gamemode)
# env.seed(0)

# USE GPU?
device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")

# MODEL THINGS
bufferSize = 256
model = PPOModel()
buffer = ReplayBuffer(model.featureDims, model.numActions, device, bufferSize = bufferSize, gamma = 0.9)
model = model.to(device)
model.eval()
model.imageEncoder.eval()
model.actor.eval()
model.critic.eval()

step = 1
window = None
playerActionVec = np.zeros((num_agents, 3))

start = time.time()
numIters = 100
numEpochs = 5
batchSize = 16
EPSClip = 0.2
entropyCoeff = 0.01
valueCoeff = 0.08
learningRate = 1e-5
maxGradNorm = 10

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = learningRate
)

# INITIALIZE ENVIRONMENT
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
delta = 0.9
wasTerminal = False

# if not window:
#     window = env.viewer.window

for iterNum in range(numIters):
    stepNum = 0
    with torch.no_grad():
        while not buffer.isFilled():
            stepNum += 1
            if step % 32 == 0:
                print(f'Epoch {iterNum + 1}, step {stepNum + 1}')
                print(step / (time.time() - start))

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
            step+=1



    # MODEL OPTIMIZATION
    print("OPTIMIZING")
    for epochNum in range(numEpochs):
        indices = np.arange(bufferSize)

        for startIndice in range(0, bufferSize, batchSize):
            endIndice = startIndice + batchSize
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
                1 - EPSClip,
                1 + EPSClip
            )
            policyLoss = torch.mean(torch.maximum(policyLoss1, policyLoss2))

            # Calculate value loss
            returns = advantages + buffer.values[batchIndices]
            newValues = model.getValue(buffer.encodedObservations[batchIndices])
            valueLossUnclipped = torch.square(newValues - returns)

            newValuesClipped = buffer.values[batchIndices] + torch.clamp(
                newValues - buffer.values[batchIndices],
                -EPSClip,
                EPSClip
            )
            valueLossClipped = torch.square(newValuesClipped - returns)
            valueLoss = torch.mean(torch.maximum(valueLossUnclipped, valueLossClipped))

            # Calculate entropy loss
            entropyLoss = torch.mean(newEntropy)

            # Combine them losses
            loss = policyLoss - entropyCoeff * entropyLoss + valueLoss * valueCoeff

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.actor.parameters(), maxGradNorm)
            optimizer.step()

    buffer.reset()
    

env.close()

