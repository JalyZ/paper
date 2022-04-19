import random, pickle, os.path, math, glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


from IPython.display import clear_output
from tensorboardX import SummaryWriter

M = np.zeros([20,20],dtype=int)
# M[4,5] = 0
M[14,10] = 1
# M[7,15] = 0
N=1
action_number = 4
action_space = N*action_number
state_channel = N*2
class PathEnv1:
    def __init__(self, render : bool = False):
        self._render = render
        # 定义动作空间
        self.action = np.zeros([N,action_number],dtype=int)
        # 定义状态空间
        self.state = np.zeros([N,2],dtype=int)
        # 定义回报
        self.r = 0
        # 计数器
        self.step_num = 0
    
    def __apply_action(self, action):
        transMatrix = np.array([[0,1],[0,-1],[-1,0],[1,0]])
        self.state = self.state + np.dot(action,transMatrix)
        for s in self.state:
            if min(s[0],s[1])<0 or max(s[0],s[1])>19:
                s[0] = np.clip(s[0],0,19)
                s[1] = np.clip(s[1],0,19)
                self.r += -100
    
    def reset(self):
        self.state = np.array([[0,0]]) #np.array([[0,0],[19,0],[0,19]])
        self.r = 0
        self.step_num = 0
        return self.state

    def reward(self):
        r = 0
        for s in self.state:
            r += M[s[0],s[1]]# - np.linalg.norm(np.array([14,10])-s)
        self.r += r

    def judge(self):
        if self.step_num < 2000:
            if abs(np.array([14,10])-self.state[0]).sum() == 0:
                print("state: ",self.state)
                return True
            return False
        return True
    
    def step(self, action):
        self.r = 0
        self.__apply_action(action)
        self.step_num += 1
        self.reward()
        if self.judge():
            done = True
        else:
            done = False
        info = {}
        return self.state, self.r, done, info
    
    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        pass

class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=5):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_channels, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(100, 100)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        y = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        z = F.relu(self.fc2(y))
        actions_value = self.out(z)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value                                                    # 返回动作值

class Memory_Buffer(object):
    def __init__(self, memory_size=1000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size: # buffer not full
            self.buffer.append(data)
        else: # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def size(self):
        return len(self.buffer)

class DDQNAgent:
    def __init__(self, in_channels = 1, action_space = [], USE_CUDA = False, memory_size = 10000, epsilon  = 1, lr = 1e-4):
        self.epsilon = epsilon
        self.action_space = action_space
        self.memory_buffer = Memory_Buffer(memory_size)
        self.DQN = DQN(in_channels = in_channels, num_actions = action_space)
        self.DQN_target = DQN(in_channels = in_channels, num_actions = action_space)
        self.DQN_target.load_state_dict(self.DQN.state_dict())


        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.DQN = self.DQN.cuda()
            self.DQN_target = self.DQN_target.cuda()
        #self.optimizer = optim.RMSprop(self.DQN.parameters(),lr=lr, eps=0.001, alpha=0.95)
        self.optimizer = optim.Adam(self.DQN.parameters(),lr=lr, eps=0.001, betas=(0.9,0.99))

    def observe(self, lazyframe):
        state =  torch.from_numpy(lazyframe).float()
        if self.USE_CUDA:
            state = state.cuda()
        return state

    def value(self, state):
        q_values = self.DQN(state)
        return q_values

    def act(self, state, epsilon = None):
        if epsilon is None: epsilon = self.epsilon
        action=np.zeros([N,action_number],int)
        q_values = self.value(state).cpu().detach().numpy()
        if random.random()<epsilon:
            for i in range(N):
                action[i,int(np.random.randint(0, action_number))] = 1
        else:
            actions_value = torch.from_numpy(q_values.reshape(N,action_number))
            for i in range(N):
                a = torch.unsqueeze(actions_value[i], 0)
                temp = torch.max(a,1)[1].data.numpy()
                action[i,int(temp)] = 1
            #aciton = q_values.argmax(1)[0]
        return action

    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        """ Compute td loss using torch operations only. Use the formula above. """
        actions = torch.tensor(actions).long()    # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype =torch.float)  # shape: [batch_size]
        is_done = torch.tensor(is_done, dtype = torch.uint8)  # shape: [batch_size]

        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        # get q-values for all actions in current states
        predicted_qvalues = self.DQN(states.reshape(-1,state_channel))

        # select q-values for chosen actions
        x = actions.reshape(-1,action_space)#torch.tensor(range(1,16))
        #x.requires_grad = True
        predicted_qvalues_for_actions = predicted_qvalues[range(states.shape[0]),:]*x # ?
        # predicted_qvalues_for_actions.requires_grad = True
        predicted_qvalues_for_actions = torch.sum(predicted_qvalues_for_actions, dim=1)
        
        # compute q-values for all actions in next states
        ## Where DDQN is different from DQN
        predicted_next_qvalues_current = self.DQN(next_states)
        predicted_next_qvalues_target = self.DQN_target(next_states)
        # compute V*(next_states) using predicted next q-values
        next_state_values =  predicted_next_qvalues_target.gather(1, torch.max(predicted_next_qvalues_current, 1)[1].unsqueeze(1)).squeeze(1)

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = rewards + gamma *next_state_values
        
        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)
        l = torch.nn.MSELoss(reduction='sum')
        loss = l(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())
        return loss

    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done= data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def learn_from_experience(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)
            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            #for param in self.DQN.parameters():
                #param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return(td_loss.item())
        else:
            return(0)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_training(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-100:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss, average on 100 stpes')
    plt.plot(losses,linewidth=0.2)
    plt.show()

# if __name__ == '__main__':

# Training DQN in PongNoFrameskip-v4
env = PathEnv1(render=True)


gamma = 0.99
epsilon_max = 0.5
epsilon_min = 0.01
eps_decay = 30000
frames = 50000
USE_CUDA = True
learning_rate = 1e-3
max_buff = 10000
update_tar_interval = 1000
batch_size = 32
print_interval = 1000
log_interval = 1000
learning_start = 10000
initState=np.array([10,10])

agent = DDQNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate)

frame = env.reset()

episode_reward = 0
all_rewards = []
losses = []
episode_num = 0
is_win = False
# tensorboard
summary_writer = SummaryWriter(log_dir = "DDQN", comment= "good_makeatari")

# e-greedy decay
epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
            -1. * frame_idx / eps_decay)
# plt.plot([epsilon_by_frame(i) for i in range(10000)])
i = 0
loss = 0
action_buffer = np.zeros(action_number)
while loss>100 or episode_num<300:#for i in range(frames):
    i+=1
    epsilon = epsilon_by_frame(i)
    state = agent.observe(frame)
    action = agent.act(state.reshape(1,-1), epsilon)
    # while(np.linalg.norm(np.dot((action[0]+action_buffer),np.array([[0,1],[0,-1],[-1,0],[1,0]])))<1):
    #     #print(action,action_buffer)
    #     action = agent.act(state.reshape(1,-1), 1)
    # action_buffer=action[0]
    next_frame, reward, done, _ = env.step(action)

    episode_reward += reward
    agent.memory_buffer.push(frame.reshape(-1,state_channel), action, reward, next_frame.reshape(-1,state_channel), done)
    frame = next_frame

    loss = 0
    if agent.memory_buffer.size() >= learning_start:
        loss = agent.learn_from_experience(batch_size)
        losses.append(loss)


    # if i % print_interval == 0:
    #     print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (i, episode_reward, loss, epsilon, episode_num))


    if i % update_tar_interval == 0:
        agent.DQN_target.load_state_dict(agent.DQN.state_dict())

    if done:

        frame = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1
        print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (i, np.mean(all_rewards[-1:]), loss, epsilon, episode_num))
        #avg_reward = float(np.mean(all_rewards[-100:]))

summary_writer.close()
# 保存网络参数
#torch.save(agent.DQN.state_dict(), "trained model/DDQN_dict.pth.tar")
plot_training(i, all_rewards, losses)


done = False
frame = env.reset()
while not done:
    epsilon = 0
    state = agent.observe(frame)
    action = agent.act(state.reshape(1,-1), epsilon)
    next_frame, reward, done, _ = env.step(action)
    frame = next_frame
    episode_reward += reward
    print(action,frame,reward)