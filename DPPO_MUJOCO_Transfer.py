import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from collections import deque
from utils import *
import time
import pickle
import os
seedset = 600
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
# parser.add_argument('--mode', default='test', type=str)
# parser.add_argument('--env', type=str, default="Hopper-v1")
# parser.add_argument('--env', type=str, default="Swimmer-v1")
# parser.add_argument('--env', type=str, default="InvertedPendulum-v1")
parser.add_argument('--env', type=str, default="BipedalWalker-v2")
# parser.add_argument('--render', default=False)
parser.add_argument('--render', default=True)
parser.add_argument('--test_iteration', default=200, type=int)
parser.add_argument('--max_episode', default=2000, type=int)  # num of games
parser.add_argument('--getting_data', default=True, type=bool)
parser.add_argument('--load', default=False, type=bool)
# parser.add_argument('--load', default=True, type=bool)
parser.add_argument('--threshold', default=2.50, type=float)
parser.add_argument('--mixed_version', default=False, type=bool)
# parser.add_argument('--mixed_version', default=True, type=bool)
args = parser.parse_args()
# from ppo import train_model
if args.mode is "train":
    EP_MAX = args.max_episode
else:
    EP_MAX = args.test_iteration


def load_dt(game):
    if game == "Hopper-v1":
        f2 = open("decision_model/" + 'dt_Hopper-v115.txt', 'rb')
    elif game == "Swimmer-v1":
        f2 = open("decision_model/" + 'dt_Swimmer-v111.txt', 'rb')
    elif game == "InvertedPendulum-v1":
        f2 = open("decision_model/" + 'dt_InvertedPendulum-v17.txt', 'rb')
    elif game == "Walker2d-v1":
        f2 = open("decision_model/" + 'dt_Walker2d-v117.txt', 'rb')
    elif game == "BipedalWalker-v2":
        f2 = open("decision_model/" + 'dt_BipedalWalker-v215.txt', 'rb')
    return f2


f2 = load_dt(args.env)
s2 = f2.read()
rdt = pickle.loads(s2)
prob = rdt.tree_.impurity


class HyperParams:
    gamma = 0.99
    lamda = 0.98
    hidden = 64
    critic_lr = 0.0003
    actor_lr = 0.0003
    batch_size = 64
    l2_rate = 0.001
    clip_param = 0.2


hp = HyperParams()


class PPO:

    def get_gae(self, rewards, masks, values):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)

        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + \
                hp.gamma * running_returns * masks[t]
            running_tderror = rewards[t] + hp.gamma * \
                previous_value * masks[t] - values.data[t]
            running_advants = running_tderror + hp.gamma * hp.lamda * \
                running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants

        advants = (advants - advants.mean()) / advants.std()
        return returns, advants

    def surrogate_loss(
            self,
            actor,
            advants,
            states,
            old_policy,
            actions,
            index):
        mu, std, logstd = actor(torch.Tensor(states))
        new_policy = log_density(actions, mu, std, logstd)
        old_policy = old_policy[index]

        ratio = torch.exp(new_policy - old_policy)
        surrogate = ratio * advants
        return surrogate, ratio

    def train_model(self, actor, critic, memory, actor_optim, critic_optim):
        memory = np.array(memory)
        states = np.vstack(memory[:, 0])
        actions = list(memory[:, 1])
        rewards = list(memory[:, 2])
        masks = list(memory[:, 3])
        values = critic(torch.Tensor(states))

        # ----------------------------
        # step 1: get returns and GAEs and log probability of old policy
        returns, advants = self.get_gae(rewards, masks, values)
        mu, std, logstd = actor(torch.Tensor(states))
        old_policy = log_density(torch.Tensor(actions), mu, std, logstd)

        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)

        # ----------------------------
        # step 2: get value loss and actor loss and update actor & critic
        for epoch in range(10):
            np.random.shuffle(arr)

            for i in range(n // hp.batch_size):
                batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                inputs = torch.Tensor(states)[batch_index]
                returns_samples = returns.unsqueeze(1)[batch_index]
                advants_samples = advants.unsqueeze(1)[batch_index]
                actions_samples = torch.Tensor(actions)[batch_index]

                loss, ratio = self.surrogate_loss(
                    actor, advants_samples, inputs, old_policy.detach(), actions_samples, batch_index)

                values = critic(inputs)
                critic_loss = criterion(values, returns_samples)
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                clipped_ratio = torch.clamp(ratio,
                                            1.0 - hp.clip_param,
                                            1.0 + hp.clip_param)
                clipped_loss = clipped_ratio * advants_samples
                actor_loss = -torch.min(loss, clipped_loss).mean()

                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, num_outputs)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


def train():
    # you can choose other environments.
    # possible environments: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2,
    # HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2, Walker2d-v2
    env = gym.make(args.env)
    env.seed(seedset)
    torch.manual_seed(500)
    total_with_PPO = 0
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)
    if args.load:
        save_name = Save_name(args.env)
        actor.load_state_dict(torch.load('PPO/DT/' + args.env + "actor.pth"))
        critic.load_state_dict(torch.load('PPO/DT/' + args.env + "critic.pth"))
        actor = actor.eval()
    ppo = PPO()
    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                              weight_decay=hp.l2_rate)

    # running_state = ZFilter((num_inputs,), clip=5)
    episodes = 0

    for iter in range(15000):
        actor.eval(), critic.eval()
        memory = deque()
        steps = 0
        with_PPO_step = 0
        scores = []
        while with_PPO_step < 256:
            episodes += 1
            state = env.reset()
            with_PPO = 0
            # state = running_state(state)
            score = 0
            for single_step in range(10000):
                if args.render:
                    env.render()
                steps += 1
                s_nexis = state[np.newaxis, :]
                node_indicator = (rdt.decision_path(s_nexis))
                node = rdt.apply(s_nexis)
                mse = prob[node]
                # print(node)
                if mse < args.threshold:
                    action = rdt.predict(s_nexis)
                    if args.env == "BipedalWalker-v2":
                        action = np.squeeze(action)
                    next_state, reward, done, _ = env.step(action)
                else:
                    with_PPO_step += 1
                    mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                    action = get_action(mu, std)[0]
                    next_state, reward, done, _ = env.step(action)
                    with_PPO += 1
                    total_with_PPO += 1
                    if done:
                        mask = 0
                    else:
                        mask = 1
                    memory.append([state, action, reward, mask])
                score += reward
                state = next_state
                if done:
                    PPO_interrupt.append(with_PPO)
                    print('{} episode score is {:.2f}'.format(episodes, score))
                    GLOBAL_RUNNING_STEP.append(single_step)
                    GLOBAL_RUNNING_R.append(score)
                    print_time_information(
                        start, GLOBAL_EP=episodes, MAX_GLOBAL_EP=EP_MAX)
                    print(with_PPO)
                    break
            scores.append(score)
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        actor.train(), critic.train()
        ppo.train_model(actor, critic, memory, actor_optim, critic_optim)
        nowtime = time.time()
        if episodes > EP_MAX:
            # torch.save(actor, 'PPO/DT/' + str(episodes) + args.env + 'actor.pkl')
            break
    torch.save(actor.state_dict(), 'PPO/DT/' + args.env + 'actor.pth')
    torch.save(critic.state_dict(), 'PPO/DT/' + args.env + 'critic.pth')
    print('model has saved')


def Save_name(game):
    if game == "Hopper-v1":
        name = 'Hopper-v1actor.pth'
    elif game == "Swimmer-v1":
        name = 'Swimmer-v1actor.pth'
    elif game == "InvertedPendulum-v1":
        name = 'InvertedPendulum-v1actor.pth'
    elif game == "Walker2d-v1":
        name = 'Walker2d-v1actor.pth'
    return name


def test():
    # you can choose other environments.
    # possible environments: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2,
    # HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2, Walker2d-v2
    env = gym.make(args.env)
    env.seed(500)
    torch.manual_seed(500)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    print('state size:', num_inputs)
    print('action size:', num_actions)
    # actor = torch.load('net/2001Swimmer-v1actor.pkl')
    if args.mixed_version is True:
        actor = Actor(num_inputs, num_actions)  # 先初始化一个模型
        # actor = torch.load('PPO/10005Walker2d-v1actor.pkl')
        # save_name = Save_name(args.env)
        actor.load_state_dict(torch.load('PPO/DT/' + args.env + 'actor.pth'))
        actor = actor.eval()
        print(actor)
    # running_state = ZFilter((num_inputs,), clip=5)
    episodes = 0
    for iter in range(200):
        steps = 0
        scores = []
        while steps < 2048:
            episodes += 1
            state = env.reset()
            with_PPO = 0
            score = 0
            for single_step in range(10000):
                steps += 1
                s_nexis = state[np.newaxis, :]
                if args.mixed_version is True:
                    node_indicator = (rdt.decision_path(s_nexis))
                    node = rdt.apply(s_nexis)
                    mse = prob[node]
                    # print(node)
                    if mse < args.threshold:
                        action = rdt.predict(s_nexis)
                        if args.env == "BipedalWalker-v2":
                            action = np.squeeze(action)
                    else:
                        mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                        action = get_action(mu, std)[0]
                        with_PPO += 1
                else:
                    action = rdt.predict(s_nexis)
                    if args.env == "BipedalWalker-v2":
                        action = np.squeeze(action)
                # env.render()
                next_state, reward, done, _ = env.step(action)
                if args.render:
                    env.render()
                score += reward
                state = next_state
                if done:
                    # print(score)
                    print('{} episode score is {:.2f}'.format(episodes, score))
                    GLOBAL_RUNNING_STEP.append(single_step)
                    GLOBAL_RUNNING_R.append(score)
                    print_time_information(
                        start, GLOBAL_EP=episodes, MAX_GLOBAL_EP=EP_MAX)
                    print(with_PPO)
                    PPO_interrupt.append(with_PPO)
                    break
            scores.append(score)
        score_avg = np.mean(scores)
        if episodes > EP_MAX:
            break
        # print('{} episode score is {:.2f}'.format(episodes, score_avg))


if __name__ == '__main__':
    start = time.time()
    GLOBAL_RUNNING_R = []
    GLOBAL_RUNNING_STEP = []
    PPO_interrupt = []
    # GLOBAL_EP = 0
    if args.mode is "train":
        train()
    else:
        test()
    plt_reward_step(
        GLOBAL_RUNNING_R=GLOBAL_RUNNING_R,
        GLOBAL_RUNNING_STEP=GLOBAL_RUNNING_STEP,
        title=args.env)
    sum_PPO = np.sum(PPO_interrupt)
    print('sum_PPO', sum_PPO)
    sum_step = np.sum(GLOBAL_RUNNING_STEP)
    print('avg_interrupt', sum_PPO * 100 / sum_step)
    total_time = time.time() - start
    print('sum_step', sum_step)
    print('avg_time', total_time * 100 / sum_step)
