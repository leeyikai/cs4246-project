from pyglet.window import key
import numpy as np
from Env import AgarEnv
import time
from DeepQTrainer import Agent
from utils import plotLearning
import torch as T
import tqdm
from Config import SPEED_MULTIPLIER

num_agents = 1
render = False
train = False
test = True
data_path1 = "agar_model_clocked_base5.pth"
img_path1 = "agar_model_clocked_base5.png"
output_path1 = "agar_model_clocked_base5.txt"

data_path2 = "agar_model_clocked_base_growth5.pth"
img_path2 = "agar_model_clocked_base_growth5.png"
output_path2 = "agar_model_clocked_base_growth5.txt"

data_path3 = "agar_model_clocked_base_growth_triple5.pth"
img_path3 = "agar_model_clocked_base_growth_triple5.png"
output_path3 = "agar_model_clocked_base_growth_triple5.txt"

data_path4 = "agar_model_clocked_base_growth_killed5.pth"
img_path4 = "agar_model_clocked_base_growth_killed5.png"
output_path4 = "agar_model_clocked_base_growth_killed5.txt"

data_path5 = "agar_model_clocked_base_growth_double5.pth"
img_path5 = "agar_model_clocked_base_growth_double5.png"
output_path5 = "agar_model_clocked_base_growth_double5.txt"

load_model = True
num_bots = 200
num_steps = 500
step_rate = 5 # 5 Hz, 5 steps per second
step_time = 1/(step_rate * SPEED_MULTIPLIER) * 1000000000 # in nanoseconds
gamemode = 0
env = AgarEnv(num_agents, num_bots, gamemode)
#env.seed(0)

agent = Agent(gamma=0.99, epsilon=0.001, batch_size=64, n_actions=160, eps_end=0.01,
                  input_dims=[13], lr=0.001, load_model=load_model, model_path=data_path5)

scores, eps_history, steps, mass  = [], [], [], []
curr_mass = 0
n_games = 100
n_test_games = 100
total_score = 0
step = 0
window = None
action = np.zeros((num_agents, 3))

start = time.perf_counter_ns()
#implement tqdm for progress bar

if test:
    print("TESTING")
    for i in tqdm.tqdm(range(n_test_games), "Testing"):
        print("TESTING")
        score = 0
        done = False
        step = 0
        reward = 0
        observation = env.reset()
        observation = env.split_observation(observation)
        start = time.perf_counter_ns()
        # print("Reset Observation = " + str(observation))
        while not done and step < num_steps:
            if (time.perf_counter_ns() - start < step_time):
                continue
            else:
                #print("Steps: " + str(step))
                start = time.perf_counter_ns()
                #print(start)
                step+=1
                if observation[0] is not None and observation[11] is not None:
                    
                    if step % 100 == 0:
                        print('step', step)
                    if render:
                        env.render(0)
                        if not window:
                            window = env.viewer.window
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
                    
                    observation_, reward, done, info, mass1 = env.step(action1, reward)
                    # print("Reward = " + str(reward))
                    score += reward[0]
                    agent.store_transition(observation, action, reward, 
                                                observation_, done)
                    if mass1 > 0.00:
                        curr_mass = mass1
                    observation = observation_
        print("DONE: " + str(done))
        scores.append(score)
        steps.append(step)
        mass.append(curr_mass)
        avg_score = np.mean(scores)
        avg_step = np.mean(steps)
        avg_mass = np.mean(mass)

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon,
                'average steps %.2f' % avg_step,
                'average mass %.2f' % avg_mass)
        with open(output_path5, "a") as f:
            f.write("episode " + str(i) + " score " + str(score) + " average score " + str(avg_score) + " average steps " + str(avg_step) + " epsilon " + str(agent.epsilon) + " average mass " + str(avg_mass) + "\n")

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
        start = time.perf_counter_ns()
        # print("Reset Observation = " + str(observation))
        while not done and step < num_steps:
            if (time.perf_counter_ns() - start < step_time):
                continue
            else:
                #print("Steps: " + str(step))
                start = time.perf_counter_ns()
                #print(start)
                step+=1
                if observation[0] is not None and observation[11] is not None:
                    
                    if step % 100 == 0:
                        print('step', step)
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
        print("DONE: " + str(done))
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
                'epsilon %.5f' % agent.epsilon)
    T.save(agent.Q_eval.state_dict(), data_path)
    x = [i+1 for i in range(n_games)]
    filename = img_path
    plotLearning(x, scores, eps_history, filename)

