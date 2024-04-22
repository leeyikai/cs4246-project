from pyglet.window import key
import numpy as np
from Env import AgarEnv
from models.VisionPPOModel import VisionPPOModel
from models.ReplayBuffer import ReplayBuffer
import torch
import tqdm
import math
import time
from trainingConfig import trainingConfig
from Config import SPEED_MULTIPLIER

from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import pydot

step_rate = 5 # 5 Hz, 5 steps per second
step_time = 1/(step_rate * SPEED_MULTIPLIER) * 1000000000 # in nanoseconds  
gameNumber = 1
resetEnvironment = True

# Initialize tensorboard for tracking
writer = SummaryWriter()

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

# Config environment
render = True
env = AgarEnv(
    num_agents = 1, 
    num_bots = trainingConfig.numBots, 
    gamemode = 0
)
window = None
playerActionVec = np.zeros((1, 3))
env.seed(0)

# # USE GPU?
device = torch.device("cpu")
# if torch.cuda.is_available():
#   device = torch.device("cuda:0")

# Load model
model = VisionPPOModel(
    numDirections = trainingConfig.numDirections,
    usePrevFrame = trainingConfig.usePrevFrame,
    usePrevAction = trainingConfig.usePrevAction,
    trainFeatureExtractor = trainingConfig.trainFeatureExtractor
)

if not trainingConfig.startFromScratch:
    model.load_state_dict(torch.load(trainingConfig.modelLoadPath, map_location=torch.device('cpu')))


model = model.to(device)
model.eval()

total_rewards = []
episode_lengths = []
total_masses = []
totalEps = 100
start = time.perf_counter_ns()
done = False

for episode in range(totalEps):
    state = env.reset()
    total_reward = 0
    total_mass = 0
    stepNum = 0

    while not done:
        
        if stepNum >= trainingConfig.maxSteps:
            break
        if (time.perf_counter_ns() - start < step_time):
            continue
        else:
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
                total_mass = agent.cells[-1].mass
                resetEnvironment = agent.isRemoved
                continue

            # Render a new window if need be
            view = env.render(0, mode = "rgb_array")
            if not window:
                window = env.viewer.window
                    
            
            ejectCooldown = env.server.getEjectCooldown(agent)
            nextStateEncodings = model.getSingleFrameEncoding(view, ejectCooldown, device)
            nextFullStateEncodings = model.getFullStateEncoding(currStateEncodings, nextStateEncodings, action, device)
            nextAction, nextLogProb, nextEntropy = model.getAction(nextFullStateEncodings)
            nextValue = model.getValue(nextFullStateEncodings)

            if agent.isRemoved or stepNum == trainingConfig.maxSteps:
                print("Agent died or stepNum == maxSteps! Resetting environment")
                resetEnvironment = True
                window.close()
                window = None

                # Write to tensorboard
                writer.add_scalar("totalReward/gameNumber", total_reward, gameNumber)
                writer.add_scalar("totalMass/gameNumber", total_reward, gameNumber)

                writer.flush()
                total_reward = 0
                gameNumber += 1
                stepNum = 0
                break
                    
            currStateEncodings = nextStateEncodings
            fullStateEncodings = nextFullStateEncodings
            action = nextAction
            logProbs = nextLogProb
            value = nextValue
            entropy = nextEntropy

            # Get the player action vec, and step the environment to get the rewards for this iteration
            playerActionVec[0] = getAgentActionVec(action, playerActionVec[0])
            observations, rewards, done, info = env.step(playerActionVec)
            total_reward += rewards[0]
            # total_mass = agent.cells[-1].mass
            # total_mass += sum([c.mass for c in agent.cells])
            # print(total_mass)

            stepNum += 1
            state = stepNum

        total_rewards.append(total_reward)
        episode_lengths.append(stepNum)
        total_masses.append(total_mass)

        # for t in total_masses:
        #     print(t)

avg_reward = sum(total_rewards) / len(total_rewards)
avg_mass = sum(total_masses) / len(total_masses)
avg_length = sum(episode_lengths) / len(episode_lengths)

print(f"Average Reward: {avg_reward}, Average Episode Length: {avg_length}")
print(f"Average Mass: {avg_mass}")
writer.close()
env.close()
    