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
train = True
test = True
data_path = "agar_model_base.pth"
img_path = "agar_model_base.img"
output_path = "output_base.txt"
load_model = True
num_bots = 200
num_steps = 10000
gamemode = 0
env = AgarEnv(num_agents, num_bots, gamemode)
#env.seed(0)

agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=160, eps_end=0.001,
                  input_dims=[10], lr=0.001, load_model=load_model, model_path=data_path)

scores, eps_history = [], []
n_games = 500
n_test_games = 10
total_score = 0
step = 1
window = None
action = np.zeros((num_agents, 3))

start = time.time()
#implement tqdm for progress bar

if test:
    print("TESTING")
    for i in tqdm.tqdm(range(n_test_games), "Testing"):
        score = 0
        done = False
        step = 0
        reward = 0
        observation = env.reset()
        observation = env.split_observation(observation)
        
        while not done and step < num_steps:
            print("Steps: ", step)
            step += 1
            if observation[0] is not None:
                
                if render:
                    env.render(0)
                    if not window:
                        window = env.viewer.window
                if observation[0] is not None:
                    action, choice = agent.choose_action(observation)
                    print("action: " + str(action))
                    if action <= 40:
                        degree = action * 9
                        print("X = " + str(np.cos(degree)) + " Y = " + str(np.sin(degree)))
                        action1 = [np.cos(degree), np.sin(degree), 0, 0]
                    elif action <= 80:
                        degree = (action - 40) * 9
                        print("X = " + str(np.cos(degree)) + " Y = " + str(np.sin(degree)))
                        action1 = [np.cos(degree), np.sin(degree), 0, 1]
                    elif action <= 120:
                        degree = (action - 80) * 9
                        print("X = " + str(np.cos(degree)) + " Y = " + str(np.sin(degree)))
                        action1 = [np.cos(degree), np.sin(degree), 1, 0]
                    else:
                        degree = (action - 120) * 9
                        print("X = " + str(np.cos(degree)) + " Y = " + str(np.sin(degree)))
                        action1 = [np.cos(degree), np.sin(degree), 1, 1]
                    
                    observation_, reward, done, info = env.step(action1, reward)
                    # print("Reward = " + str(reward))
                    score += reward[0]
                    agent.store_transition(observation, action, reward, 
                                                observation_, done)
        scores.append(score)
        avg_score = np.mean(scores)
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    print("Average Score: ", np.mean(scores))  
else:
    for i in tqdm.tqdm(range(n_games), "Loading"):
        print("TRAINING")
        score = 0
        done = False
        step = 0
        reward = 0
        observation = env.reset()
        observation = env.split_observation(observation)
        # print("Reset Observation = " + str(observation))
        while not done and step < num_steps:
            #print("Steps: " + str(step))
            step+=1
            if observation[0] is not None:
                
                if step % 100 == 0:
                    print('step', step)
                    #print(time.time() - start)
                    start = time.time()
                if render:
                    env.render(0)
                    if not window:
                        window = env.viewer.window
                try:
                    action, choice = agent.choose_action(observation)
                    if action <= 40:
                        degree = action * 9
                        action1 = [np.cos(degree), np.sin(degree), 0, 0]
                    elif action <= 80:
                        degree = (action - 40) * 9
                        action1 = [np.cos(degree), np.sin(degree), 0, 1]
                    elif action <= 120:
                        degree = (action - 80) * 9
                        action1 = [np.cos(degree), np.sin(degree), 1, 0]
                    else:
                        degree = (action - 120) * 9
                        action1 = [np.cos(degree), np.sin(degree), 1, 1]
                    
                    observation_, reward, done, info = env.step(action1, reward)
                    # print("Reward = " + str(reward))
                    score += reward[0]
                    agent.store_transition(observation, action, reward, 
                                                observation_, done)
                    agent.learn()
                    observation = observation_

                    # action[0][2] = 0
                    
                except KeyboardInterrupt:
                    if train:
                        print("Saving Model?")
                        T.save(agent.Q_eval.state_dict(), data_path)
                        print("Model Saved")
                        x = [i+1 for i in range(n_games)]
                        filename = img_path
                        plotLearning(x, scores, eps_history, filename)
                        exit()
            #print("Rewards = " + str(reward))
        scores.append(score)
        total_score += score
        eps_history.append(agent.epsilon)
        if i > 0:
            avg_score = total_score / i
        else:
            avg_score = total_score
        with open(output_path, "a") as f:
            f.write("episode " + str(i) + " score " + str(score) + " average score " + str(avg_score) + " epsilon " + str(agent.epsilon) + "\n")
        print("RESULT OF GAME: ", str(i))
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    T.save(agent.Q_eval.state_dict(), data_path)
    x = [i+1 for i in range(n_games)]
    filename = img_path
    plotLearning(x, scores, eps_history, filename)

