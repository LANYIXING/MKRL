"""
记录时间与剩余时间
"""
import time


def time_change(time_init):
    """
    定义将秒转换为时分秒格式的函数
    """
    time_list = []
    if time_init / 3600 > 1:
        time_h = int(time_init / 3600)
        time_m = int((time_init - time_h * 3600) / 60)
        time_s = int(time_init - time_h * 3600 - time_m * 60)
        time_list.append(str(time_h))
        time_list.append('h ')
        time_list.append(str(time_m))
        time_list.append('m ')

    elif time_init / 60 > 1:
        time_m = int(time_init / 60)
        time_s = int(time_init - time_m * 60)
        time_list.append(str(time_m))
        time_list.append('m ')
    else:
        time_s = int(time_init)

    time_list.append(str(time_s))
    time_list.append('s')
    time_str = ''.join(time_list)
    return time_str


def print_time_information(start, GLOBAL_EP, MAX_GLOBAL_EP, train_round=0,):
    """
    记录时间，剩余时间
    :param start:
    :param GLOBAL_EP:
    :param MAX_GLOBAL_EP:
    :return:
    """
    process = GLOBAL_EP * 1.00 / MAX_GLOBAL_EP
    if process > 1:
        process = 1
    end = time.time()
    use_time = end - start
    all_time = use_time / (process + 1e-5)
    res_time = all_time - use_time
    if res_time < 1:
        res_time = 1
    str_ues_time = time_change(use_time)
    str_res_time = time_change(res_time)

    print("Round:%s Percentage of progress:%.2f%%  Used time:%s  Rest time:%s "
          % (train_round + 1, process * 100, str_ues_time, str_res_time))


if __name__ == '__main__':
    import tensorflow as tf
    import mujoco_py
    import os
    import gym
    env = gym.make('Ant-v3')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(10000):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
