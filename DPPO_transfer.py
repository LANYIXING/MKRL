"""
Author： Lanyixing
"""
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import threading
import queue
import os
import time
import plot_record as plot_record  # 画图与记录
import time_record as time_record
import pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
f2 = open("decision_model/" + 'dt_Pendulum-v014.txt', 'rb')
s2 = f2.read()
rdt = pickle.loads(s2)
prob = rdt.tree_.impurity
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', type=str)
# parser.add_argument('--mode', default='train', type=str)
parser.add_argument("--env_name", default="Pendulum-v2")
# parser.add_argument("--env_name", default="Hopper-v1")
parser.add_argument('--test_iteration', default=500, type=int)
parser.add_argument('--max_episode', default=1000, type=int)  # num of games
parser.add_argument('--max_step', default=200, type=int)  # num of games
parser.add_argument('--n_workers', default=1, type=int)
parser.add_argument('--A_LR', default=1e-4, type=float)
parser.add_argument('--C_LR', default=2e-4, type=float)
parser.add_argument('--batch_size', default=64, type=int)  # mini batch size
parser.add_argument('--update_step', default=10, type=int)
parser.add_argument('--epsilon', default=0.2, type=float)
parser.add_argument('--gamma', default=0.9, type=int)  # discounted factor
parser.add_argument('--threshold', default=0.08, type=float)  # discounted factor
# parser.add_argument('--mixed_version', default=True, type=bool)
parser.add_argument('--mixed_version', default=True, type=bool)
# parser.add_argument('--render', default=False, type=bool) # show UI or not
# parser.add_argument('--exploration_noise', default=0.1, type=float)


args = parser.parse_args()

EP_MAX = args.max_episode
EP_LEN = args.max_step
N_WORKER = args.n_workers  # parallel workers
GAMMA = args.gamma  # reward discount factor
A_LR = args.A_LR  # learning rate for actor
C_LR = args.C_LR  # learning rate for critic
MIN_BATCH_SIZE = args.batch_size  # minimum batch size for updating PPO
UPDATE_STEP = args.update_step  # loop update operation n-steps
EPSILON = args.epsilon  # for clipping surrogate objective
GAME = args.env_name
S_DIM, A_DIM = 3, 1  # state and action dimension
if args.mode is "train":
    EP_MAX = args.max_episode
else:
    EP_MAX = args.test_iteration
    N_WORKER = 1
start = time.time()


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        # operation of choosing action
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        self.update_oldpi_op = [
            oldp.assign(p) for p, oldp in zip(
                pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv  # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        if args.mode is "train":
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
        else:
            self.restore()

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.sess.run(self.update_oldpi_op)  # copy pi to old pi
                # collect data from all workers
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:,
                                                S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(
                    self.advantage, {
                        self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op,
                               {self.tfs: s,
                                self.tfa: a,
                                self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(
                    UPDATE_STEP)]
                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(
                self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM,
                                     tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(
                l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def restore(self):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, path + "/transfer_net.ckpt")

    def save_net(self):
        save_path = self.saver.save(self.sess, path + "/transfer_net.ckpt")
        print("model has saved in", save_path)


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, prob
        if args.mode == 'train':
            total_with_PPO = 0
            while not COORD.should_stop():
                s = self.env.reset()
                ep_r = 0
                with_PPO = 0
                buffer_s, buffer_a, buffer_r = [], [], []
                for t in range(EP_LEN):
                    if not ROLLING_EVENT.is_set():  # while global PPO is updating
                        ROLLING_EVENT.wait()  # wait until PPO is updated
                        # clear history buffer, use new policy to collect data
                        buffer_s, buffer_a, buffer_r = [], [], []
                    s_nexis = s[np.newaxis, :]
                    node_indicator = (rdt.decision_path(s_nexis))
                    node = rdt.apply(s_nexis)
                    mse = prob[node]
                    # print(node)
                    if mse < args.threshold:
                        a = rdt.predict(s_nexis)
                        s_, r, done, info = self.env.step(a)
                        ep_r += r
                    else:
                        a = self.ppo.choose_action(s)
                        with_PPO += 1
                        total_with_PPO += 1
                        s_, r, done, info = self.env.step(a)
                        ep_r += r
                        buffer_s.append(s)
                        buffer_a.append(a)
                        buffer_r.append(r)
                    s = s_
                    GLOBAL_UPDATE_COUNTER += 1
                    # if t == EP_LEN - 1 or with_PPO >= MIN_BATCH_SIZE:
                    if total_with_PPO >= MIN_BATCH_SIZE and buffer_s:
                        v_s_ = self.ppo.get_v(s_)
                        discounted_r = []  # compute discounted reward
                        for r in buffer_r[::-1]:
                            v_s_ = r + GAMMA * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()
                        # print('6')
                        # print(buffer_s)
                        bs = np.vstack(buffer_s)
                        ba = np.vstack(buffer_a)
                        br = np.array(discounted_r)[:, np.newaxis]
                        buffer_s, buffer_a, buffer_r = [], [], []
                        QUEUE.put(np.hstack((bs, ba, br)))
                        if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                            ROLLING_EVENT.clear()
                            UPDATE_EVENT.set()

                        if GLOBAL_EP >= EP_MAX or (np.array(GLOBAL_RUNNING_R[-20:]) > -300).all():  # stop training
                            plot_record.plt_reward_step(
                                GLOBAL_RUNNING_R=GLOBAL_RUNNING_R,
                                GLOBAL_RUNNING_STEP=GLOBAL_RUNNING_STEP,
                                title=GAME)
                            COORD.request_stop()
                            break

                # record reward changes, plot later
                # if len(GLOBAL_RUNNING_R) == 0:
                #     GLOBAL_RUNNING_R.append(ep_r)
                # else:
                #     GLOBAL_RUNNING_R.append(
                #         GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
                GLOBAL_RUNNING_R.append(ep_r)
                GLOBAL_EP += 1
                print(with_PPO)
                time_record.print_time_information(
                    start, GLOBAL_EP=GLOBAL_EP, MAX_GLOBAL_EP=EP_MAX)

                print(
                    '{0:.1f}%'.format(
                        GLOBAL_EP /
                        EP_MAX *
                        100),
                    '|W%i' %
                    self.wid,
                    '|Ep_r: %.2f' %
                    ep_r,
                )

        else:
            while not COORD.should_stop():
                s = self.env.reset()
                ep_r = 0
                with_PPO = 0
                buffer_s, buffer_a, buffer_r = [], [], []
                for t in range(EP_LEN):
                    s_nexis = s[np.newaxis, :]
                    if args.mixed_version is True:
                        node_indicator = (rdt.decision_path(s_nexis))
                        node = rdt.apply(s_nexis)
                        mse = prob[node]
                        # print(node)
                        if mse < args.threshold:
                            a = rdt.predict(s_nexis)
                        else:
                            a = self.ppo.choose_action(s)
                            with_PPO += 1
                    # # simple version
                    else:
                        a = rdt.predict(s_nexis)

                    s_, r, done, _ = self.env.step(a)
                    # buffer_s.append(s)
                    # buffer_a.append(a)
                    # buffer_r.append(r)
                    s = s_
                    ep_r += r
                    # count to minimum batch size, no need to wait other
                    # workers
                    if t == EP_LEN - 1 or done:
                        GLOBAL_RUNNING_STEP.append(t)
                        GLOBAL_RUNNING_R.append(ep_r)
                        time_record.print_time_information(
                            start, GLOBAL_EP=GLOBAL_EP, MAX_GLOBAL_EP=EP_MAX)
                        print(with_PPO)
                        # data = np.array(buffer_s)
                        # label = np.array(buffer_a)
                GLOBAL_EP += 1
                if GLOBAL_EP >= EP_MAX:  # stop training
                    plot_record.plt_reward_step(
                        GLOBAL_RUNNING_R=GLOBAL_RUNNING_R,
                        GLOBAL_RUNNING_STEP=GLOBAL_RUNNING_STEP,
                        title=GAME)
                    COORD.request_stop()
                    break
                # record reward changes, plot later
                if len(GLOBAL_RUNNING_R) == 0:
                    GLOBAL_RUNNING_R.append(ep_r)
                else:
                    GLOBAL_RUNNING_R.append(ep_r)
                GLOBAL_EP += 1
                print(
                    '{0:.1f}%'.format(
                        GLOBAL_EP /
                        EP_MAX *
                        100),
                    '|W%i' %
                    self.wid,
                    '|Ep_r: %.2f' %
                    ep_r,
                )
        GLOBAL_EP += 1


if __name__ == '__main__':
    path = "PPO/" + GAME
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # not update now
    ROLLING_EVENT.set()  # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    GLOBAL_RUNNING_STEP = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()  # workers putting data in this queue
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()  # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
    threads[-1].start()
    COORD.join(threads, ignore_live_threads=True)
    if args.mode is "train":
        GLOBAL_PPO.save_net()
    # plot reward change and test

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.ion()
    plt.show()
