#!/usr/bin/env python
# coding: utf-8

from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import pandas as pd
from unityagents import UnityEnvironment
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from td3_agent import AgentTD3


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_CollabAndCompete(env, brain_name, n_episodes, max_t, \
                           solved_score, consec_episodes, print_every, \
                           actor_path, critic_path, postfix):
    """
    Params
    ======
        n_episodes (int)      : maximum number of training episodes
        max_t (int)           : maximum number of timesteps per episode
        solved_score (float)  : min avg score over consecutive episodes
        consec_episodes (int) : number of consecutive episodes used to calculate score
        print_every (int)     : interval to display results
        actor_path (str)      : directory to store actor network weights
        critic_path (str)     : directory to store critic network weights

    """

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    brain = env.brains[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


    agents = []
    for agent_number in range(num_agents):
        new_agent = AgentTD3(state_size=state_size, action_size=action_size,
                             agents_size=1 , random_seed=1)

        agents.append(new_agent) 


    mean_scores = []                               # list of mean scores from each episode
    min_scores = []                                # list of lowest scores from each episode
    max_scores = []                                # list of highest scores from each episode
    best_score = -np.inf
    scores_window = deque(maxlen=consec_episodes)  # mean scores from most recent episodes
    moving_avgs = []                               # list of moving averages
    df = pd.DataFrame(columns=['episode', 'duration', 'min', 'max', 'std', 'mean'])


    # now execute up to maximum "maxEpisodes" episodes
    for i_episode in range(1, n_episodes+1):
        # 1.Step: reset the environment - set the train_mode to True !!
        env_info = env.reset(train_mode=True)[brain_name]  

        # 2. Step: get the current state for each agent
        #states = env_info.vector_observations     
        #print("States = " , states)

        # 2a. Step Create one big state vector (size = num_agents * state_size) 
        #          from array of observations (=states) from each agents
        states = np.reshape(env_info.vector_observations, (1,num_agents*state_size))              
        #print("States1 = " , states1)

        # 3.Step: set the score of the current episode to 0 for each agent
        scores = np.zeros(num_agents)                          

        # 3a. Step: Reset all agents
        for agent in agents:
            agent.reset()
        
        start_time = time.time()

        # 4.Step: while episode has not ended (done = True or t < max_t) repeat
        for t in range(max_t):
            # 5.Step: Calculate the next action from agent with noise
            #actions = agent.act(states, add_noise=True)
            actions = [agent.act(states, add_noise=True) for agent in agents]        
            actions = np.reshape(actions, (1, num_agents*action_size))

            # 6.Step: Tell the environment about this action and get result
            env_info = env.step(actions)[brain_name]            

            # 7.Step: now let's get the state observation from observation            
            #next_states = env_info.vector_observations          
            next_states = np.reshape(env_info.vector_observations, (1, num_agents*state_size))

            # 8.Step: now let's get the reward observation from observation            
            rewards = env_info.rewards    
            #print("Rewards = " , rewards)                     

            # 9.Step: now let's get the done observation from observation
            dones = env_info.local_done                         

            # 10.Step: Add the reward of the last action-state result  
            #scores += rewards
            scores += np.max(rewards)
 
            # 11.Step: Execute a training step of the agent
            # save experience to replay buffer, perform learning step at defined interval
#            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
#                agent.step(state, action, reward, next_state, done, t)
            #for i in enumerate(agents):
            #    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            #        agent.step(state, action, reward, next_state, done, t, i)
            for i, agent in enumerate(agents):
                agent.step(states, actions, rewards[i], next_states, dones, t,  i)

            # 12.Step: Continue while-loop with next_state as current state            
            states = next_states

            # 13.Step: in case of end of episode print the result and break loop 
            if np.any(dones):                                   # exit loop when episode ends
                break

        duration = time.time() - start_time
        min_scores.append(np.min(scores))             # save lowest score for a single agent
        max_scores.append(np.max(scores))             # save highest score for a single agent
        mean_scores.append(np.mean(scores))           # save mean score for the episode
        scores_window.append(mean_scores[-1])         # save mean score to window
        moving_avgs.append(np.mean(scores_window))    # save moving average

        df.loc[i_episode-1] = [i_episode] + list([duration, np.min(scores),
                                                  np.max(scores),
                                                  np.std(scores),
                                                  np.mean(scores)])

        # 16.Step: Print results every print_every episodes 
        if i_episode % print_every == 0:
            print('\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}\tMov. Avg: {:.1f}'.format(
                  i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]))


        # 17.Step: save actor and critic in case a new best score has been encountered 
        if mean_scores[-1] > best_score:
            torch.save(agent.actor_local.state_dict(), actor_path)
            torch.save(agent.critic_local.state_dict(), critic_path)

        # 18.Step: In case the performance "threshold" is exceeded --> stop and save the current agents neural network
        if moving_avgs[-1] >= solved_score and i_episode >= consec_episodes:
            print('\nEnvironment SOLVED in {} episodes!\tMoving Average ={:.1f} over last {} episodes'.format(
                i_episode-consec_episodes, moving_avgs[-1], consec_episodes))
            torch.save(agent.actor_local.state_dict(), actor_path)
            torch.save(agent.critic_local.state_dict(), critic_path)
    return df


def plot_minmax(df):
    """Print min max plot of DQN Agent analytics

    Params
    ======
        df :    Dataframe with scores
    """
    ax = df.plot(x='episode', y='mean')
    plt.fill_between(x='episode', y1='min', y2='max', color='lightgrey', data=df)
    x_coordinates = [0, 150]
    y_coordinates = [30, 30]
    plt.plot(x_coordinates, y_coordinates, color='red')
    plt.show()


def load_reacher(env, brain_name, actor_path, critic_path):

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    brain = env.brains[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = AgentTD3(state_size=state_size, action_size=action_size,
                     random_seed=1)

    if torch.cuda.is_available():
        agent.actor_local.load_state_dict(torch.load(actor_path))
        agent.critic_local.load_state_dict(torch.load(critic_path))
    else:
        agent.actor_local.load_state_dict(torch.load(actor_path,
                                                     map_location=lambda storage, loc: storage))
        agent.critic_local.load_state_dict(torch.load(critic_path,
                                                      map_location=lambda storage, loc: storage))
    return agent


def test_reacher(env, brain_name, agent,  runs=100):
    # set overall sum of scores to 0
    mean_scores = []

    # now execute up to maximum runs episodes
    for i_episode in range(runs):

        # 1.Step: reset the environment - set the train_mode to False !!
        env_info = env.reset(train_mode=True)[brain_name]     

        # 2. Step: get the current state 
        states = env_info.vector_observations                  
        
        # 3.Step: set the score of the current episode to 0
        num_agents = len(env_info.agents)
        scores = np.zeros(num_agents)                          
        
        start_time = time.time()
        
        # 4.Step: while episode has not ended (done = True) repeat
        while True:
            # 5.Step: Calculate the next action from agent with epsilon 0 
            #         add_noise = False because we are not in training mode !
            actions = agent.act(states, add_noise=False)         

            # 6.Step: Tell the environment about this action and get result
            env_info = env.step(actions)[brain_name]           

            # 7.Step: now let's get the state observation from observation
            next_states = env_info.vector_observations         

            # 8.Step: now let's get the reward observation from observation
            rewards = env_info.rewards                        

            # 9.Step: now let's get the done observation from observation
            dones = env_info.local_done                         
            
            # 10.Step: Add the reward of the last action-state result  
            scores += env_info.rewards                         
            
            # 11.Step: Continue while-loop with next_state as current state
            states = next_states                                           

            # 12.Step: in case of end of episode print the result and break loop 
            if np.any(dones):                                 
                break

        # 13.Step: Finally append the score of last epsisode to the overall scores
        mean_scores.append(np.mean(scores))

        duration = time.time() - start_time
        print('\rEpisode {} ({} sec)\tMean: {:.1f}'.format(
            i_episode, round(duration), mean_scores[-1]))


    return mean_scores





env = UnityEnvironment(file_name='../UnityTennis/Tennis.exe')

# Environments contain **_brains_** which are responsible for deciding 
# the actions of their associated agents. Here we check for the first brain 
# available, and set it as the default brain we will be controlling from Python.

# get the default brain
brain_name = env.brain_names[0]

# number of agents
env_info = env.reset(train_mode=True)[brain_name]


# Set the minimum score the agent has to reach in order to solve this task
threshold = 38.0

# Set the maximum number of episodes which the agent 
max_episodes = 1000
max_t = 10000

print_every = 1
consec_episodes = 2

# Set this variable to "True" in case you want to retrain your agent
train = True

# Set the filename for storage of the trained model
actor_filename = "actor_ckpt"
critic_filename = "critic_ckpt"
postfix = ".pth"

if train:
    df = train_CollabAndCompete(env, brain_name, max_episodes, max_t, threshold, consec_episodes, print_every,  \
                       actor_filename, critic_filename, postfix)

    plot_minmax(df)

if train == False:
    # load a previously stored solution
    agent = load_reacher(env, brain_name, actor_filename , critic_filename)

    # execute 100 consecutive tests and calculate the average score
    runs = 100
    scores = test_reacher(env, brain_name, agent,  runs)
    
    print('Total score (averaged over agents) for 100 episodes: {}'.format(np.mean(scores)))



# When finished, you can close the environment.
env.close()