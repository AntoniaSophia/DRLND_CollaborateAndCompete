from unityagents import UnityEnvironment
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm

# from td3_agent import MADDPG
from ddpg_agent import DDPGAgent
from maddpg import MADDPG
from td3_agent import AgentTD3

unity_environment_path = "../../UnityTennis/Tennis.exe"

def draw(scores, path="fig.png", title="Performance", xlabel="Episode #", ylabel="Score"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path)


def train(args):
    # prepare environment
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(
        states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    agent_1 = DDPGAgent(state_size, action_size)
    #agent_2 = DDPGAgent(state_size, action_size)
    agent_2 = AgentTD3(state_size, action_size)

    agent_1_path = '../results/ddgp_solo/00_best_model.checkpoint'
    #agent_2_path = '../results/temp/new_ddpg_model_2.checkpoint'
    agent_2_path = '../results/temp/td3_model.checkpoint'

    agent = MADDPG(state_size, action_size, agent_1, agent_2, False, True)
    agent.load(agent_1_path,0)
    #agent.load(agent_2_path,1)  

    total_rewards = []
    avg_scores = []
    max_avg_score = -1
    max_score = -1
    threshold_init = 20
    noise_t = args.epsilon
    noise_decay = args.epsilon_decay
    latest_avg_score = -1
    # for early-stopping training if consistently worsen for # episodes
    worsen_tolerance = threshold_init
    for i_episode in range(1, 1+args.num_episodes):

        env_inst = env.reset(train_mode=True)[
            brain_name]    # reset the environment
        states = env_inst.vector_observations                # get the current state
        # initialize score array
        scores = np.zeros(num_agents)
        dones = [False]*num_agents
        while not np.any(dones):
            # select an action
            actions = agent.act(states, noise_t)
            # send the action to the environment
            env_inst = env.step(actions)[brain_name]
            next_states = env_inst.vector_observations       # get the next state
            rewards = env_inst.rewards                       # get the reward
            dones = env_inst.local_done                      # see if episode has finished
            agent.update(states, actions, rewards, next_states, dones, i_episode)

            noise_t *= noise_decay
            scores += rewards                                # update scores
            states = next_states

        episode_score = np.max(scores)
        total_rewards.append(episode_score)
        print("\rEpisodic {} Score: {:.4f}\t Avg Score: {:.4f}".format(
            i_episode, episode_score, latest_avg_score), end=' ')

        if max_score <= episode_score:
            max_score = episode_score
            # save best model so far
            agent.save(agent_1_path , agent_2_path)

        # record avg score for the latest 100 steps
        if len(total_rewards) >= args.test_n_run:
            latest_avg_score = sum(
                total_rewards[(len(total_rewards)-args.test_n_run):]) / args.test_n_run
            avg_scores.append(latest_avg_score)

            if max_avg_score <= latest_avg_score:           # record better results
                worsen_tolerance = threshold_init           # re-count tolerance
                max_avg_score = latest_avg_score
            else:
                if max_avg_score > 2.0:
                    worsen_tolerance -= 1                   # count worsening counts
                    print("Loaded from last best model.")
                    # continue from last best-model
                    agent.reload(agent_1_path,0)
                    agent.reload(agent_2_path,1)
                if worsen_tolerance <= 0:                   # earliy stop training
                    print("Early Stop Training.")
                    break
    return total_rewards


def test(args):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # dim of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # dim of the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent_1 = DDPGAgent(state_size, action_size)
    #agent_2 = DDPGAgent(state_size, action_size)
    agent_2 = AgentTD3(state_size, action_size)

    agent_1_path = '../results/ddgp_solo/01_best_model.checkpoint'
    #agent_2_path = '../results/temp/new_ddpg_model.checkpoint'
    agent_2_path = '../results/td3_opponent/00_best_td3_model.checkpoint'

    agent = MADDPG(state_size, action_size, agent_2, agent_1, False, False)
    agent.load(agent_2_path,0)
    agent.load(agent_1_path,1)   
    

    test_scores = []
    for i_episode in tqdm(range(1, 1+args.test_n_run)):
        # initialize the scores
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[
            brain_name]   # reset the environment
        states = env_info.vector_observations               # get the current states
        dones = [False]*num_agents
        while not np.any(dones):
            actions = agent.act(states)                     # select actions
            # send the actions to the environment
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations      # get the next states
            rewards = env_info.rewards                      # get the rewards
            dones = env_info.local_done                     # see if episode has finished
            scores += rewards                               # update the scores
            # roll over the states to next time step
            states = next_states

        test_scores.append(np.max(scores))

    avg_score = sum(test_scores)/len(test_scores)
    print("Test Score: {}".format(avg_score))

    return avg_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', default=int(10000), type=int)
    parser.add_argument('--model_path', default='../results/ddgp_solo/00_best_model.checkpoint')
    parser.add_argument('--test_n_run', default=int(100), type=int)
    parser.add_argument('--epsilon', default=float(1.0), type=float)
    parser.add_argument('--epsilon_decay', default=float(.995), type=float)
    parser.add_argument('--main_n_loop', default=int(10), type=int)
   
    args = parser.parse_args()

    env = UnityEnvironment(file_name=unity_environment_path)
    score = test(args)
    exit()

    project = {}
    project["args"] = args
    project["scores"] = []
    project["rewards"] = []
    mp = args.model_path
    for i in range(args.main_n_loop):
        args.model_path = "{:02d}_".format(i)+mp
        reward = train(args)
        project["rewards"].append(reward)
        score = test(args)
        project["scores"].append(score)
        print(score)
    
    f = open("project.json","w")
    f.write( str(project) )
    f.close()

    env.close()


    print(np.min(project["scores"]))
    print(np.max(project["scores"]))
    print(np.mean(project["scores"]))
