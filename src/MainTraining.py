from pyglet.window import key
import numpy as np
from Env import AgarEnv
import time
from DeepQTrainer import Agent
from utils import plotLearning

num_agents = 1
render = True
num_bots = 200
gamemode = 0
env = AgarEnv(num_agents, num_bots, gamemode)
#env.seed(0)

agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=100, eps_end=0.001,
                  input_dims=[5], lr=0.005)
scores, eps_history = [], []
n_games = 500
    
step = 1
window = None
action = np.zeros((num_agents, 3))



start = time.time()

for i in range (n_games):
    score = 0
    done = False
    observation = env.reset()
    split_observation(observation)
    print("Observation = " + str(observation))
    while not done and step < 100:
        print("Step = " + str(step))
        if step % 40 == 0:
            print('step', step)
            print(step / (time.time() - start))
        if render:
            env.render(0)
            if not window:
                window = env.viewer.window
        action = agent.choose_action(observation)
        #get x and y coordinates for the number action
        degree = action * 3.6
        action1 = [np.cos(degree), np.sin(degree), 0, 0]
        observation_, reward, done, info = env.step(action1)
        score += reward[0]
        print("Reward = " + str(reward))
        agent.store_transition(observation, action, reward, 
                                    observation_, done)
        agent.learn()
        observation = observation_

        # action[0][2] = 0
        step+=1
    scores.append(score)
    eps_history.append(agent.epsilon)
    # print("Scores = " + str(scores))
    # for score in scores:
    #     if isinstance(score, (list, np.ndarray)):
    #         print("Found a sequence in scores:", score)
    # avg_score = np.mean(scores[-100:])

    print('episode ', i, 'score %.2f' % score,
            'average score %.2f' % 0,
            'epsilon %.2f' % agent.epsilon)

x = [i+1 for i in range(n_games)]
filename = 'test_train.png'
plotLearning(x, scores, eps_history, filename)

