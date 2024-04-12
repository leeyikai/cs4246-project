from pyglet.window import key
import numpy as np
from Env import AgarEnv
import time
from DeepQTrainer import Agent
from utils import plotLearning

num_agents = 1
num_bots = 19
gamemode = 0
env = AgarEnv(num_agents, num_bots, gamemode)
#env.seed(0)

agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=3, eps_end=0.001,
                  input_dims=[5], lr=0.005)
scores, eps_history = [], []
n_games = 500
    
step = 1
window = None
action = np.zeros((num_agents, 3))

# def on_mouse_motion(x, y, dx, dy):
#     action[0][0] = (x / 1920 - 0.5) * 2
#     action[0][1] = (y / 1080 - 0.5) * 2

# def on_key_press(k, modifiers):
#     if k == key.W:
#         action[0][2] = 2
#     elif k == key.SPACE:
#         action[0][2] = 1
#     else:
#         action[0][2] = 0

start = time.time()

for i in range (n_games):
    score = 0
    done = False
    observation = env.reset()
    print("Observation = " + str(observation))
    while not done:
        if step % 40 == 0:
            print('step', step)
            print(step / (time.time() - start))

        action = agent.choose_action(observation)
        print("Action = " + str(action))
        action1 = [0.5, 0.5, 0]
        observation_, reward, done, truncated, info = env.step(action1)
        score += reward
        agent.store_transition(observation, action, reward, 
                                    observation_, done)
        agent.learn()
        observation = observation_

        action[0][2] = 0
        step+=1
    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-100:])

    print('episode ', i, 'score %.2f' % score,
            'average score %.2f' % avg_score,
            'epsilon %.2f' % agent.epsilon)

x = [i+1 for i in range(n_games)]
filename = 'test_train.png'
plotLearning(x, scores, eps_history, filename)