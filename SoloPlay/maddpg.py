import torch
import numpy as np

class MADDPG:
    """
    The Multi-Agent consisting of two Actor-Critic based agents
    """
    def __init__(self, state_size, action_size, agent_1, agent_2, \
                train_agent_1=True, train_agent_2=True):

        super(MADDPG, self).__init__()

        self.train_agent_1 = train_agent_1
        self.train_agent_2 = train_agent_2

        #agent = DDPGAgent(state_size, action_size)
        self.adversarial_agents = [agent_1, agent_2]     # the agent self-plays with itself
        
    def get_actors(self):
        """
        get actors of all the agents in the MADDPG object
        """
        actors = [ddpg_agent.actor_local for ddpg_agent in self.adversarial_agents]
        return actors

    def get_target_actors(self):
        """
        get target_actors of all the agents in the MADDPG object
        """
        target_actors = [ddpg_agent.actor_target for ddpg_agent in self.adversarial_agents]
        return target_actors

    def act(self, states_all_agents, noise_t=0.0):
        """
        get actions from all agents in the MADDPG object
        """
        actions = []

        agent = self.adversarial_agents[0]
        if self.train_agent_1 == True:
            actions.append(agent.act(states_all_agents[0], noise_t))
        else:     
            actions.append(agent.act(states_all_agents[0], 0.0))
        
        agent = self.adversarial_agents[1]
        if self.train_agent_2 == True:
            actions.append(agent.act(states_all_agents[1], noise_t))
        else:
            actions.append(agent.act(states_all_agents[1], 0.0))

        return np.stack(actions, axis=0)

    def update(self, *experiences):
        """
        update the critics and actors of all the agents
        """
        states, actions, rewards, next_states, dones, i_episode = experiences
        for agent_idx, agent in enumerate(self.adversarial_agents):
            state = states[agent_idx,:]
            action = actions[agent_idx,:]
            reward = rewards[agent_idx]
            next_state = next_states[agent_idx,:]
            done = dones[agent_idx]

            if agent_idx == 0 and self.train_agent_1 == True:
                agent.update_model(state, action, reward, next_state, done)
            
            if agent_idx == 1 and self.train_agent_2 == True:
                agent.update_model(state, action, reward, next_state, done)
            
    def save(self, path_agent_1 , path_agent_2):
        """
        Save the model
        """
        agent = self.adversarial_agents[0]
        if self.train_agent_1 == True:
            torch.save((agent.actor_local.state_dict(), \
                agent.critic_local.state_dict()), path_agent_1)
        
        agent = self.adversarial_agents[1]
        if self.train_agent_2 == True:
            torch.save((agent.actor_local.state_dict(), \
                agent.critic_local.state_dict()), path_agent_2)


    def load(self, path, agent_number):
        """
        Load model 
        """
        agent = self.adversarial_agents[agent_number]

        actor_state_dict, critic_state_dict = torch.load(path)
        agent = self.adversarial_agents[agent_number]
        agent.actor_local.load_state_dict(actor_state_dict)
        agent.actor_target.load_state_dict(actor_state_dict)
        agent.critic_local.load_state_dict(critic_state_dict)
        agent.critic_target.load_state_dict(critic_state_dict)
        agent.lr_actor *= agent.lr_decay
        agent.lr_critic *= agent.lr_decay
        for group in agent.actor_optimizer.param_groups:
            group['lr'] = agent.lr_actor
        for group in agent.critic_optimizer.param_groups:
            group['lr'] = agent.lr_critic
    
        self.adversarial_agents[agent_number] = agent
        
        print("Loaded model for agent " , agent_number , "  from path " , path)

    def reload(self, path, agent_number):
        """
        Load model 
        """
        agent = self.adversarial_agents[agent_number]
        if (agent_number == 0 and self.train_agent_1 == True) or \
           (agent_number == 1 and self.train_agent_2 == True):

            actor_state_dict, critic_state_dict = torch.load(path)
            agent = self.adversarial_agents[agent_number]
            agent.actor_local.load_state_dict(actor_state_dict)
            agent.actor_target.load_state_dict(actor_state_dict)
            agent.critic_local.load_state_dict(critic_state_dict)
            agent.critic_target.load_state_dict(critic_state_dict)
            agent.lr_actor *= agent.lr_decay
            agent.lr_critic *= agent.lr_decay
            for group in agent.actor_optimizer.param_groups:
                group['lr'] = agent.lr_actor
            for group in agent.critic_optimizer.param_groups:
                group['lr'] = agent.lr_critic
        
            self.adversarial_agents[agent_number] = agent
            
            print("Reloaded model for agent " , agent_number , "  from path " , path)