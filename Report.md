[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

## Instructions

I'm using Windows10/64 without any GPU, only CPU available on my laptop. 
Furthermore I was using Visual Studio Code and Anaconda. The development of the solution took place under Visual Studio Code and the Anaconda Shell. 

Additionally I used the Reacher version 2, the 20 Agents variants: Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

### Install Environment
1. Anaconda

Download and install: https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe

2. Environment
Python 3.6 Environment and not newer is necessary because of Unity 0.4 dependencies like Tensorflow
```
    conda create -n unity_mlagent python=3.6
    conda activate unity_mlagent
```
3. Tensorflow 

    Download of wheel because pip install was not working directly
https://files.pythonhosted.org/packages/fd/70/1a74e80292e1189274586ac1d20445a55cb32f39f2ab8f8d3799310fcae3/tensorflow-1.7.1-cp36-cp36m-win_amd64.whl
```
    pip install tensorflow-1.7.1-cp36-cp36m-win_amd64.whl
```
4. Unity ML Agents

    Download Unity ml agents https://codeload.github.com/Unity-Technologies/ml-agents/zip/0.4.0b and unzip
```
    cd ml-agents-0.4.0b\python
    pip install .
```

5. Pytorch

    Pytorch will be used for the DQN Agent
```
    conda install -c pytorch pytorch
```

6. Additional packages
```
    pip install pandas git
```

7. Tennis Environment

    Download (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip  
    Unzip in project folder so that ```Tennis.exe```can be found

8. Install version control system Git

Download and install:
https://github.com/git-for-windows/git/releases/download/v2.26.2.windows.1/Git-2.26.2-64-bit.exe


### Perform Train and testing of DQN Agent via jupyter Notebook
1. Start python environment
    
    Call Anaconda prompt via Windows Start menu

![](static/anaconda.jpg) 

```
    cd <project folder>
    conda activate drlnd
``` 

2. Clone project
```
    git clone https://github.com/AntoniaSophia/DRLND_CollaborateAndCompete.git
    cd DRLND_CollaborateAndCompete
```
3. Start of jupyter environment
```
    jupyter notebook
```

4. Open Solution.ipynb and follow the steps

## Description of my solution

Most obviously the project 2 (Continuous Control of the robot arm) the TD3 is really fast and stable - I guess much faster and more stable compared to all other known algorithms (even the DDPG). But for this task the TD3 was really really hard. To be honest without some changes I have seen in different solutions, I would not have chosen the changes in the TD3 neural network.

I was not able to train an TD3 agent from scratch and came up with the following idea:
- train a good DDPG agent (taken from a different source)
- use this to use a running DDPG agent and train the TD3 agent 

Experience is that in most cases the training process doesn't succeed in a well-trained model. This also holds for the DDPG agent. I'm not able to quantify what exactly means "most cases" - this would require much more comparison attempts which I could not execute do to a lack in time.

In this chapter I describe my own solution to this project. My solutions consists of the following files:
   - `Solution.ipynb` - this is the Jupyter notebook containing my solution
   - `td3_agent.py` - this is a vanilla DQN agent and contains the following parameters
     - `env` - the environment where the Reacher is contained in
     - `brain_name` - the brain_name of the Reacher
     - `max_episodes` - maximum limit of episodes to be executed
     - `threshold` - the threshold of average reward in order to successfully finish training
     - `seed` - seed of the randomizer (keep constant in case you want to get reproducable results!)
   - `td3_py.py` - the neural network of the TD3 neural network
   - `maac.py` - The wrapper class containing the Multi-Agent Player_1 / Player_2 agent structure with Actor-Critic architecture
   - `FifoMemory.py`  - an alternative implementation of the replay memory based on a simple first-in-first-out (FIFO) eviction scheme
   - `Solution.py` - this basically has the same content as the Jupyter notebook - it was my starting point for development of the solution ("offline")


I am using the TD3 algorithm (= Twin Delayed Deep Deterministic Policy Gradient), which is the successor to the Deep Deterministic Policy Gradient (DDPG)(Lillicrap et al, 2016). 
The TD3 algorithm is addressing function approximation error in Actor-Critic methods caused by the algorithm continuously over estimating the Q values of the critic (value) network. These estimation errors build up over time and can lead to the agent falling into a local optima or experience catastrophic forgetting. TD3 addresses this issue by focusing on reducing the overestimation bias seen in previous algorithms. 

This is done with the addition of 3 key features: 
- Using a pair of critic networks (The twin part of the title)
- Delayed updates of the actor (The delayed part)
- Action noise regularisation 

The original paper for the TD3 algorithm can be found at: https://arxiv.org/abs/1802.09477

A pretty nice TD3 description can be found at (and some parts of this description) is taken from: https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93

The base for implementation is taken from: https://github.com/prasoonkottarathil/Twin-Delayed-DDPG-TD3-/blob/master/TD3.ipynb

As additional changes I've implemented the following additions:
- using FIFO memory instead of a the standard replay memory (the same FIFO memory which I already used in the first project BananaCollector)
- using an additional "success memory" which stores much fewer samples (only the last 10 samples before a positive reward has been received)
   
In the training process:
- the regular replay memory is used once for training (LEARN_NUM_MEMORY)
- the success replay memory is used twice for training (LEARN_NUM_MEMORY_SUCCESS)

The main idea behind this success memory is that only randomly the agents is hit by the ball and recognizes that hitting a ball could be a good idea. The success memory stores exactly those events (which are pretty rare at the beginning of a training process) and gives a kind of "kick-start" during the starting phase of the training - at least this was my idea behind....

        # Success memory contains only the last 10 samples which led to a positive reward
        self.memory_success = FifoMemory(int(BUFFER_SIZE), int(BATCH_SIZE))

        # Rolling sample memory of last 10 samples
        self.memory_short = FifoMemory(10, 10)

        
## The MAAC agent
   
The class `maac.py` contains a wrapper class `MAAC` for a two-player DRL training for Actor-Critic approaches.
Basically the MAAC agent stand for a multi-agent which contains exactly two agents (opponents) and distributes the step/act/reset/load/save methods of two independent agents.

The constructor class for `MAAC` contains the following six parameters:
- `state_size` - the state size of the environment
- `action_size` - the action size of the environment
- `agent_1` - player 1
- `agent_2` - player 2
- `trainable (bool` - whether player 1 shall be a trainable agent 
- `trainable (bool` - whether player 2 shall be a trainable agent 

Using this class it is possible to run e.g. two different agents, one TD3 (as player 1) and one DDPG (as player 2) where only player 2 (agent_2) is trainable:

    agent_1 = TD3Agent(state_size, action_size)
    agent_2 = DDPGAgent(state_size, action_size)

    agent_1_path = 'results/td3_opponent/00_best_td3_model.checkpoint'
    agent_2_path = 'results/ddgp_solo/01_best_model.checkpoint'

    agent = MAAC(state_size, action_size, agent_1, agent_2, False, True)

    agent.load(agent_1_path,0)
    agent.load(agent_2_path,1) 


In order to run the same agent for both opponents, just put the same agent in the constructor of the MAAC, e.g.

    agent_1 = DDPGAgent(state_size, action_size)
    agent_1_path = 'results/ddgp_solo/01_best_model.checkpoint'
    agent = MAAC(state_size, action_size, agent_1, agent_1, True, True)
    agent.load(agent_1_path,0)


## TD3 Agent

In `td3_agent.py` I've followed the implementation from the original TD3 paper

As additional features I've implemented the following additions:
- using FIFO memory instead of a the standard replay memory (the same FIFO memory which I already used in the first project BananaCollector)
- using an additional "success memory" which stores much fewer samples (only the last 10 samples before a positive reward has been received)
   
In the training process:
- the regular replay memory is used once for training (LEARN_NUM_MEMORY)
- the success replay memory is used twice for training (LEARN_NUM_MEMORY_SUCCESS)

The main idea behind this success memory is that only randomly the agents is hit by the ball and recognizes that hitting a ball could be a good idea. The success memory stores exactly those events (which are pretty rare at the beginning of a training process) and gives a kind of "kick-start" during the starting phase of the training - at least this was my idea behind....

Here is the implementation of the learning step function (lines 104-140):

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        reached = True
        if len(self.memory_success) < BATCH_SIZE:
            reached = False

        self.memory.add(state, action, reward, next_state, done)
        self.memory_short.add(state, action, reward, next_state, done)

        # Fill the success memory in case this agents receives positive reward
        if reward > 0.0:
            for i in range(len(self.memory_short)):
                self.memory_success.add(
                    self.memory_short.samples[i].state, \
                    self.memory_short.samples[i].action, \
                    self.memory_short.samples[i].reward, \
                    self.memory_short.samples[i].next_state, \
                    self.memory_short.samples[i].done)
                    
            self.memory_short.clear()

        if reached == False and len(self.memory_success) > BATCH_SIZE:
            print("Success memory ready for use!")

        # Train with the complete replay memory
        if len(self.memory) > BATCH_SIZE:
            for i in range(LEARN_NUM_MEMORY):
                experiences = self.memory.sample() 
                # delay update of the policy and only update every 2nd training
                self.learn(experiences, 0 , GAMMA)

        # Train with the success replay memory
        if (len(self.memory_success) > self.memory_success.batch_size):
            for i in range(LEARN_NUM_MEMORY_SUCCESS):
                experiences_success = self.memory_success.sample() 
                self.learn(experiences_success, 0 ,GAMMA)


The core of the learning step is using the minimum of both Critic values for the Q-target (td3_agent.py lines 184-196) and the sum of both critic loss for backward step

        Q_targets_next1, Q_targets_next2 = self.critic_target(next_states, actions_next)

        # TD3 --> Take the minimum of both critic in order to avoid overestimation
        Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected1, Q_expected2 = self.critic_local(states, actions)

        # compute critic loss [HOW MUCH OFF?] as sum of both loss from target
        critic_loss = F.mse_loss(Q_expected1, Q_targets)+F.mse_loss(Q_expected2, Q_targets)


## The TD3 neural network

The neural network can be found at the file `td3_model.py` and is based on the implementation from the orginal TD3 paper. The Actor is still only one network, whereas the Critic is consisting of two networks. I tried to use the identical agent as for the 2nd project (Continuous Control with robot arm), but I miserably failed. 
The two changes compared to the original TD3 paper I had to make are:
- choose  `fc1_units = 64` and  `fc2_units = 128`
- introduce an additional layer in the Critics

The input layer of the Actor receives 24 states and the output layer has size 2 with following dimensions (nodes) of the hidden layers:  `fc1_units = 64` and  `fc2_units = 128`:

       `self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)`

As mentioned the Critic contains two networks which conquer the potential over estimation. Both networks have identical structure, the input layer of the Critic also receives 24 states and the output layer has size 1 (Critic Q-target values) with following dimensions (nodes) of the hidden layers:  `fc1_units = 64` and  `fc2_units = 128`:

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fcs1_units)
        self.fc4 = nn.Linear(fcs1_units, 1)

        self.fcs4 = nn.Linear(state_size, fcs1_units)
        self.fc5 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc6 = nn.Linear(fc2_units, fcs1_units)        
        self.fc7 = nn.Linear(fcs1_units, 1)

Compared to the original TD§ paper the following layers are additional `self.fc3 = nn.Linear(fc2_units, fcs1_units)` and `self.fc6 = nn.Linear(fc2_units, fcs1_units)`


## DDPG Agent

The DDPG agent I've taken from https://github.com/SusannaMaria/DRLND_P3_CollabCompet in order to a starting point for my TD3 training.


### Used parameters

In the next section I'm going to explain all the parameters I have chosen. 
The reason to chose the parameters exactly like this is basically a mix of different sources:
- https://arxiv.org/pdf/1802.09477.pdf
- https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93
- https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/random_process.py
- Try & Error 
- Searching in the Internet for people trying out different parameters for this Tennis problem 

  - `EPSILON = 5.0` - starting with maximum exploration 
  - `EPSILON_DECAY = 0.995` - 
  - `seed = 1` - the seed for the randomizer (I kept it constant in order to be able to compare results)
  - `GAMMA = 0.95` - the discount factor
  - `TAU = 1e-3` - # for soft update of target parameters
  - `BUFFER_SIZE = int(1e6)` - having 100000 samples to store 
  - `BATCH_SIZE = 64` - The batch size for every replay memory learning after each step
  - `WEIGHT_DECAY = 0` # L2 weight decay
  - `TAU = 1e-3` # soft updating of target params
  - `LR_ACTOR = 1e-5` # learning rate of the gradient descent of the neural network's actor
  - `LR_CRITIC = 1e-4` # learning rate of the gradient descent of the neural network's critic
  - `LEARN_NUM_MEMORY = 1` # number of learning passes for the regular memory
  - `LEARN_NUM_MEMORY_SUCCESS = 3` # number of learning passes for the success replay memory 
  - `OU_SIGMA = 0.20` # Ornstein-Uhlenbeck noise parameter
  - `OU_THETA = 0.15` # Ornstein-Uhlenbeck noise parameter


### Other parameters

  - `max_episodes` - maximum limit of episodes to be executed
  - `max_t` - the maximum number of steps within an episode before terminating the episode
  - `threshold` - the threshold of average reward in order to successfully finish training
  - `conseq_episodes` - 
  - `actor_filename` - the filename to store the trained agent's actor network  
  - `critic_filename` - the filename to store the trained agent's critic network  


## Discussion

Most obviously the project 2 (Continuous Control of the robot arm) the TD3 is really fast and stable - I guess much faster and more stable compared to all other known algorithms (even the DDPG). But for this task the TD3 was really really hard. To be honest without some changes I have seen in different solutions, I would not have chosen the changes in the TD3 neural network.

I was not able to train an TD3 agent from scratch and came up with the following idea:
- train a good DDPG agent (taken from a different source)
- use this to use a running DDPG agent and train the TD3 agent 

Experience is that in most cases the training process doesn't succeed in a well-trained model. This also holds for the DDPG agent. I'm not able to quantify what exactly means "most cases" - this would require much more comparison attempts which I could not execute do to a lack in time.

My best result so far are:
- DDPG: https://github.com/AntoniaSophia/DRLND_CollaborateAndCompete/blob/4aadecfe68e402349cebb7d71226a17069193093/results/ddgp_solo/01_best_model.checkpoint
- TD3: https://github.com/AntoniaSophia/DRLND_CollaborateAndCompete/blob/4aadecfe68e402349cebb7d71226a17069193093/results/td3_solo/00_best_td3_model.checkpoint

If I run those two models - which were trained in completely different training sessions

See the score table over episodes:

![](docs/TD3_Result.png) 


    agent_1 = TD3Agent(state_size, action_size)
    agent_2 = DDPGAgent(state_size, action_size)

    agent_1_path = 'results/td3_opponent/00_best_td3_model.checkpoint'
    agent_2_path = 'results/ddgp_solo/01_best_model.checkpoint'

    agent = MAAC(state_size, action_size, agent_1, agent_2, False, False)

    agent.load(agent_1_path,0)
    agent.load(agent_2_path,1) 

    scores = test_CollabAndCompete(env, brain_name, agent,  runs=100)
    print('Total score (averaged over agents) for 100 episodes: {}'.format(np.mean(scores)))

This creates a total score (averaged over agents) for 100 episodes of: 2.553350038053468  

I'm not sure which effect my additions (using FIFO memory and an additional success memory) have on the overall performance of my TD3 implementation. For a detailed statement I would have to run several rounds of different feature sets. But as I only have an older CPU the training process took almost 12 hours for 2500 episodes. 

Basically the data structure of the regular replay memory and FIFO is almost similar, just the order of the elements is different. As already indicated in the first project "BananaCollector" my result could indicate that using a FIFO is more efficient, but in fact just running a few tests in one project is definitely not enough to conclude on that. I admit, I was just playing around with some ideas I found in the Internet....

As I recognized the calculations in Jupyter notebook is different compared to execution on command line. Executed on command line the task was even solved in 18 episodes - even much faster! I've observed the same behavior in the first project BananaCollector, but just thought it was randomly because the initial weights might have been different. Let's observe this further in future - but in case someone knows the reason of this behavior I would be grateful!


Further potential improvements:
- Tuning the batch size and size of the short term memory
- Train against a set of different opponents at the same time (let's say 10 opponents and every action is calculated by a different random opponent) to learn all strategies at once
- Targeted tests in order to find out whether the idea of the "success memory" is really performing better or worked in my special case (or even just on my laptop)
- Further tuning of the hyperparameters, e.g. by evolution algorithms


## Final remark - famous last works... ;-)
I felt this project was really really tough one.... Not because it was difficult to implement the basic structure for multi-agent training (this weas the easy part), but because of the following facts:
- the training process took a relatively long time (more than 12 hours on my CPU for a really good agent)
- the results are neither reproducible
- agents collapse pretty often
- the success of a training effect is highly depending on random "events" - as such when a trained agent is executing a good action which results in a positive reward ("it couldn't avoid being hit by the ball and thus learned that it might be a good idea to hit the ball itself")

I guess this also was the goal of this project: to show that deep reinforcement learning can be very very powerful - "but still doesn' work" (see https://www.alexirpan.com/2018/02/14/rl-hard.html), as cited "Currently, deep RL isn’t stable at all, and it’s just hugely annoying for research."

But despite some frustrating during the project: all in all, again it was a lot of fun, I really learned a lot of new things and enjoyed!

Thank you very much Udacity for this excellent course!!

