from pyglet.window import key
import numpy as np
from Env import AgarEnv
from models.VisionPPOModel import VisionPPOModel
from models.ReplayBuffer import ReplayBuffer
import torch
import math 
from trainingConfig import trainingConfig
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import pydot

def cyclical_lr(step, base_lr=1e-3, max_lr=1e-2, step_size=trainingConfig.stepSize_lr, mode="triangular"):
    cycle = np.floor(1 + step / (2 * step_size))
    x = np.abs(step / step_size - 2 * cycle + 1)
    if mode == "triangular":
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return lr

def linear_decay_lr(current_step, initial_lr, min_lr, total_steps):
    if current_step >= total_steps:
        return min_lr
    return initial_lr - (initial_lr - min_lr) * (current_step / total_steps)

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
model = VisionPPOModel(
    numDirections = trainingConfig.numDirections,
    usePrevFrame = trainingConfig.usePrevFrame,
    usePrevAction = trainingConfig.usePrevAction,
    trainFeatureExtractor = trainingConfig.trainFeatureExtractor
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
    lr = trainingConfig.base_lr
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
gameNumber = 1

# Initialize tensorboard for tracking
writer = SummaryWriter()
totalReward = 0

batches_per_epoch = -(-trainingConfig.replayBufferSize // trainingConfig.batchSize)  # Using ceiling division
totalEndSteps = trainingConfig.numIters * trainingConfig.numEpochs * batches_per_epoch

stepNum = 0
totalSteps = 0

for iterNum in range(trainingConfig.numIters):
    with torch.no_grad():
        model.eval()
        while not buffer.isFilled():
            stepNum += 1
            totalSteps += 1

            if trainingConfig.cyclical_lr:
                current_lr = cyclical_lr(totalSteps, base_lr=trainingConfig.base_lr, max_lr=trainingConfig.max_lr, step_size=trainingConfig.stepSize_lr)
            
            if trainingConfig.decay_lr:
                current_lr = current_lr = linear_decay_lr(stepNum, trainingConfig.base_lr, trainingConfig.min_lr, totalEndSteps)

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            if totalSteps % 100 == 0:
                print(f'Iter {iterNum + 1}, totalSteps {totalSteps + 1}')

            if (resetEnvironment):
                # Reset the environment
                print("Resetting environment...")
                env.reset()
                agent = env.players[0]
                ejectCooldown = 0
                view = env.render(0, mode = "rgb_array")
                
                currStateEncodings = model.getSingleFrameEncoding(view, ejectCooldown, device)
                prevStateEncodings = currStateEncodings.clone() # Assume that the prev state is the same when starting out
                prevAction = torch.tensor([0]).to(device) # Assume that the prev action is 0
                fullStateEncodings = model.getFullStateEncoding(prevStateEncodings, currStateEncodings, prevAction, device)
                action, logProb, entropy = model.getAction(fullStateEncodings)
                value = model.getValue(fullStateEncodings)

                playerActionVec[0] = getAgentActionVec(action, playerActionVec[0])
                observations, rewards, done, info = env.step(playerActionVec)
                resetEnvironment = agent.isRemoved
                continue

            # Render a new window if need be
            view = env.render(0, mode = "rgb_array")
            if not window:
                window = env.viewer.window
            
            # Abit confusing, but nextFullStateEncodings is the state encodings of this state. The reason why we have the
            # word next is cos we need to add the buffer entry of the PREVIOUS step, which relies on the value of this state
            ejectCooldown = env.server.getEjectCooldown(agent)
            nextStateEncodings = model.getSingleFrameEncoding(view, ejectCooldown, device)
            nextFullStateEncodings = model.getFullStateEncoding(currStateEncodings, nextStateEncodings, action, device)
            nextAction, nextLogProb, nextEntropy = model.getAction(nextFullStateEncodings)
            nextValue = model.getValue(nextFullStateEncodings)

            # Add to replay buffer
            buffer.addEntry(
                encodedObservation = fullStateEncodings, # Frm prev iter
                action = action, # Frm prev iter
                logProb = logProb, # Frm prev iter
                value = value, # Frm prev iter
                nextValue = nextValue, # Frm curr iter, the value of curr state to calculate advantage
                reward = torch.tensor([rewards[0]]).to(device), # Frm prev iter
                isTerminal = resetEnvironment
            )

            if agent.isRemoved or stepNum == trainingConfig.maxSteps:
                print("Agent died or stepNum == maxSteps! Resetting environment")
                resetEnvironment = True
                window.close()
                window = None

                # Write to tensorboard
                writer.add_scalar("totalReward/gameNumber", totalReward, gameNumber)
                writer.flush()
                totalReward = 0
                gameNumber += 1
                stepNum = 0
            
            currStateEncodings = nextStateEncodings
            fullStateEncodings = nextFullStateEncodings
            action = nextAction
            logProbs = nextLogProb
            value = nextValue
            entropy = nextEntropy

            # Get the player action vec, and step the environment to get the rewards for this iteration
            playerActionVec[0] = getAgentActionVec(action, playerActionVec[0])
            observations, rewards, done, info = env.step(playerActionVec)
            totalReward += rewards[0]


    # MODEL OPTIMIZATION
    print("OPTIMIZING")
    model.train()
    for epochNum in range(trainingConfig.numEpochs):
        indices = np.arange(trainingConfig.replayBufferSize)
        epsilon = trainingConfig.EPSClip

        totalPolicyLoss = 0
        totalValueLoss = 0
        totalLoss = 0
        for startIndice in range(0, trainingConfig.replayBufferSize, trainingConfig.batchSize):

            if trainingConfig.cyclical_lr:
                current_lr = cyclical_lr(totalSteps, base_lr=trainingConfig.base_lr, max_lr=trainingConfig.max_lr, step_size=trainingConfig.stepSize_lr)
            
            if trainingConfig.decay_lr:
                current_lr = linear_decay_lr(stepNum, trainingConfig.base_lr, trainingConfig.min_lr, totalEndSteps)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            writer.add_scalar("LearningRate", current_lr, totalSteps)

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
                1 - epsilon,
                1 + epsilon
            )
            policyLoss = -torch.mean(torch.minimum(policyLoss1, policyLoss2))

            # Calculate value loss
            returns = advantages + buffer.values[batchIndices]
            newValues = model.getValue(buffer.encodedObservations[batchIndices])
            valueLossUnclipped = torch.square(newValues - returns)

            newValuesClipped = buffer.values[batchIndices] + torch.clamp(
                newValues - buffer.values[batchIndices],
                -epsilon,
                epsilon
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
        
        epsilon = epsilon * 0.99 # annealed clip range

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