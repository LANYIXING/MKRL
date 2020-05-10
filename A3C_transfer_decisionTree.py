"""
by lan
"""
import threading  # 多线程
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import time
import plot_record as plot_record  # 画图与记录
import time_record as time_record
import pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
f2 = open("decision_model/"+'dt_Acrobot-v110.txt', 'rb')
# f2 = open("decision_model/"+'dt_cartpole5.txt', 'rb')
# f2 = open("decision_model/"+'dt_CartPole-v614.txt', 'rb')
# f2 = open("decision_model/"+'dt_MountainCar5.txt', 'rb')
# GAME = 'MountainCar-v0'
GAME = 'Acrobot-v1'
# GAME = 'CartPole-v7'
s2 = f2.read()
clf2 = pickle.loads(s2)
TRAIN = False
# TRAIN = True
# TEST_RENDER = True
TEST_RENDER = False  # display
mixed_version = False
# mixed_version = True
OUTPUT_GRAPH = False  # tensorboards
LOG_DIR = './log'
N_WORKERS = 16
if TRAIN:
    MAX_GLOBAL_EP = 100000  # total episodes
else:
    MAX_GLOBAL_EP = 500  # total episodes
MAX_ROUND_EP = 10000  # max steps each episode
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99
ENTROPY_BETA = 0.001
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_RUNNING_STEP = []
GLOBAL_EP = 0
start = time.time()
env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class ACNet(object):
    def __init__(self, scope, globalAC = None):

        if scope == GLOBAL_NET_SCOPE:  # GLOBAL net
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                # return actor and critic's params under the scope of GLOBAL
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                # s, a_his, v_target are designed to
                # collect interactive data
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                # the output of Actor net
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(
                    tf.float32, [None, 1], 'Vtarget')
                # return params under the scope of xx找
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(
                    scope)
                self.td = tf.subtract(self.v_target, self.v, name='TD_error')
                # calculate c_loss for minimizing td_error
                with tf.name_scope('c_loss'):
                    # c_loss = reduce_mean(td^2), promise to be 0
                    self.c_loss = tf.reduce_mean(tf.square(self.td))
                # calculate a_loss for maximizing expectation
                with tf.name_scope('a_loss'):
                    # discrete version
                    self.log_prob = tf.reduce_sum(
                        tf.log(
                            self.a_prob +
                            1e-5) *
                        tf.one_hot(
                            self.a_his,
                            N_A,
                            dtype=tf.float32),
                        axis=1,
                        keepdims=True)
                    # 原本为TensorFlow计算图中的一个op（节点）转为一个常量td，
                    # 这时候对于loss的求导反传就不会传到td去了.
                    exp_v = self.log_prob * tf.stop_gradient(self.td)
                    # 我们为了使得输出的分布更加均衡，所以要最大化这个entropy，那么就是
                    # minimize这个负的entropy。
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keepdims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                # calculate local_grad
                with tf.name_scope('local_grad'):
                    # tf.gradients(ys, xs, grad_ys=None, name='gradients',stop_gradients=None,)
                    # tf.gradients = tf.compute_gradient
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    # l_p: local parameters, g_p: global parameters
                    # 把全局参数直接给assign 给局部参数
                    # 使用zip函数实际为optimizer.compute_gradient和apply_gradient 需要
                    self.pull_a_params_op = [
                        l_p.assign(g_p) for l_p, g_p in zip(
                            self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [
                        l_p.assign(g_p) for l_p, g_p in zip(
                            self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    # 梯度实际上已经被计算,因此只要定义优化算法和使用apply_gradient即可
                    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
                    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
                    self.update_a_op = OPT_A.apply_gradients(
                        zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(
                        zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        # actor network, 2 layers
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(
                self.s,
                200,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='la')
            a_prob = tf.layers.dense(
                l_a,
                N_A,
                tf.nn.softmax,
                kernel_initializer=w_init,
                name='ap')
        # critic network, 2 layers
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(
                self.s,
                100,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='lc')
            v = tf.layers.dense(
                l_c,
                1,
                kernel_initializer=w_init,
                name='v')  # state value
        # tf.get_collection(key, scope=None)用来获取名称域中所有放入‘key’的变量的列表
        a_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope + '/actor')
        c_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def choose_action(self, s):  # run by a local

        prob_weights = SESS.run(self.a_prob, feed_dict={
                                self.s: s[np.newaxis, :]})
        prob_weights /= prob_weights.sum()  # normalize
        if TRAIN is False:
            action = np.argmax(prob_weights)
        else:
            action = np.random.choice(
                range(
                    prob_weights.shape[1]),
                p=prob_weights.ravel())

        return action

    def update_global(self, feed_dict):  # local grads applies to global net
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            j = 0
            with_A3C = 0
            while TRAIN:
                if self.name == 'W_0':
                    if TEST_RENDER:
                        self.env.render()
                # mixed version
                s_nexis = s[np.newaxis, :]
                prob = clf2.predict_prob(s_nexis)
                if np.max(prob) > 0.9:
                    a = np.squeeze(clf2.predict(s_nexis))
                    s_, r, done, info = self.env.step(a)
                    # buffer_s.append(s_)
                    # buffer_a.append(a)
                    # buffer_r.append(r)
                    ep_r += r
                else:
                    a = self.AC.choose_action(s)
                    with_A3C += 1
                    s_, r, done, info = self.env.step(a)
                    ep_r += r
                    buffer_s.append(s_)
                    buffer_a.append(a)
                    buffer_r.append(r)

                # update global and assign to local net
                if (with_A3C % UPDATE_GLOBAL_ITER == 0 or done) and (len(buffer_a) > 0):
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        # [0,0] is for getting a number instead of a matrix
                        v_s_ = SESS.run(
                            self.AC.v, {
                                self.AC.s: s_[
                                    np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    buffer_s, buffer_a, buffer_v_target = np.vstack(
                        buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    # print(SESS.run(self.AC.v_target, {self.AC.v_target: buffer_v_target}))
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                j += 1
                if done or j > MAX_ROUND_EP:
                    GLOBAL_RUNNING_STEP.append(j)
                    GLOBAL_RUNNING_R.append(ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    time_record.print_time_information(
                        start, GLOBAL_EP=GLOBAL_EP, MAX_GLOBAL_EP=MAX_GLOBAL_EP)
                    GLOBAL_EP += 1
                    break
            if TRAIN:
                # 控制训练时倒数20次都达到最大值附近即停止训练
                flag = GLOBAL_RUNNING_R[-20:]
                if (np.array(GLOBAL_RUNNING_R[-60:]) > 8000).all():
                    break

            while not TRAIN:
                if TEST_RENDER:
                    self.env.render()
                s_nexis = s[np.newaxis, :]
                if mixed_version is True:
                # # mixed version
                    prob = clf2.predict_prob(s_nexis)
                    if np.max(prob) > 0.95:
                        a = np.squeeze(clf2.predict(s_nexis))
                    else:
                        a = self.AC.choose_action(s)
                        with_A3C += 1
                # # simple version
                else:
                    a = np.squeeze(clf2.predict(s_nexis))
                s_, r, done, info = self.env.step(a)
                ep_r += r
                s = s_
                total_step += 1
                j += 1
                if done or j > MAX_ROUND_EP:
                    GLOBAL_RUNNING_STEP.append(j)
                    GLOBAL_RUNNING_R.append(ep_r)

                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    time_record.print_time_information(
                        start, GLOBAL_EP=GLOBAL_EP, MAX_GLOBAL_EP=MAX_GLOBAL_EP)
                    print (with_A3C)
                    GLOBAL_EP += 1
                    break





def main():
    if TRAIN:
        with tf.device("/cpu:0"):

            GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
            workers = []
            # Create worker
            for i in range(N_WORKERS):
                i_name = 'W_%i' % i  # worker name
                workers.append(Worker(i_name, GLOBAL_AC))

        SESS.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if OUTPUT_GRAPH:
            if os.path.exists(LOG_DIR):
                shutil.rmtree(LOG_DIR)
            tf.summary.FileWriter(LOG_DIR, SESS.graph)

        worker_threads = []
        # loop main
        for worker in workers:
            def job(): return worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)
        save_path = saver.save(SESS, path + "/save_net.ckpt")
        print("model has saved in", save_path)
        plot_record.plt_reward_step(
            GLOBAL_RUNNING_R=GLOBAL_RUNNING_R,
            GLOBAL_RUNNING_STEP=GLOBAL_RUNNING_STEP,
            title=GAME)

    else:
        with tf.device("/cpu:0"):
            GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
            workers = []
            # Create worker
            for i in range(1):
                i_name = 'W_%i' % i  # worker name
                workers.append(Worker(i_name, GLOBAL_AC))
        saver = tf.train.Saver()
        saver.restore(SESS, path + "/save_net.ckpt")
        # SESS.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            if os.path.exists(LOG_DIR):
                shutil.rmtree(LOG_DIR)
            tf.summary.FileWriter(LOG_DIR, SESS.graph)

        worker_threads = []
        for worker in workers:
            def job(): return worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)
        plot_record.plt_reward_step(
            GLOBAL_RUNNING_R=GLOBAL_RUNNING_R,
            GLOBAL_RUNNING_STEP=GLOBAL_RUNNING_STEP,
            title=GAME)


if __name__ == "__main__":
    SESS = tf.Session()
    COORD = tf.train.Coordinator()  # 设为全局变量
    path = "A3C/" + GAME
    main()
