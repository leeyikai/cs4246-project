from pyglet.window import key
import numpy as np
from Env import AgarEnv
import time
from DeepQTrainer import Agent
from utils import plotLearning
import torch as T
import tqdm

num_agents = 1
render = False
num_bots = 200
gamemode = 0
env = AgarEnv(num_agents, num_bots, gamemode)
#env.seed(0)

agent = Agent(gamma=0.99, epsilon=0.2, batch_size=64, n_actions=100, eps_end=0.001,
                  input_dims=[8], lr=0.001, load_model=False)
scores, eps_history = [], []
n_games = 500
total_score = 0
step = 1
window = None
action = np.zeros((num_agents, 3))



start = time.time()
#implement tqdm for progress bar

for i in tqdm.tqdm(range(n_games), "Loading"):
    score = 0
    done = False
    step = 0
    observation = env.reset()
    observation = env.split_observation(observation)
    # print("Reset Observation = " + str(observation))
    while not done and step < 2000:
        # print("Step = " + str(step))
        # if step % 40 == 0:
            # print('step', step)
            # print(step / (time.time() - start))
        if render:
            env.render(0)
            if not window:
                window = env.viewer.window

        action, choice = agent.choose_action(observation)
        # if choice == 1:
        #     print("Random Action" + str(action))
        # else:
        #     print("Action" + str(action))
        #get x and y coordinates for the number action
        degree = action * 3.6
        action1 = [np.cos(degree), np.sin(degree), 0, 0]
        observation_, reward, done, info = env.step(action1)
        # print("Observation_ = " + str(observation_))
        score += reward[0]
        # print("Reward = " + str(reward))
        agent.store_transition(observation, action, reward, 
                                    observation_, done)
        agent.learn()
        observation = observation_

        # action[0][2] = 0
        step+=1
    scores.append(score)
    total_score += score
    eps_history.append(agent.epsilon)
    avg_score =total_score/i if i > 0 else score
    # print("Scores = " + str(scores))
    # for score in scores:
    #     if isinstance(score, (list, np.ndarray)):
    #         print("Found a sequence in scores:", score)
    # avg_score = np.mean(scores[-100:])
    #write output to txt file
    with open("output.txt", "a") as f:
        f.write("episode " + str(i) + " score " + str(score) + " average score " + str(avg_score) + " epsilon " + str(agent.epsilon) + "\n")
    print("RESULT OF GAME: ", str(i))
    print('episode ', i, 'score %.2f' % score,
            'average score %.2f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
T.save(agent.Q_eval.state_dict(),"agar_model.pth")
x = [i+1 for i in range(n_games)]
filename = 'test_train.png'
plotLearning(x, scores, eps_history, filename)

