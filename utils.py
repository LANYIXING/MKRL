"""
some function are refer in https://github.com/dnddnjs/mujoco-pg
"""
import time
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime


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


def plt_reward_step(
        GLOBAL_RUNNING_R=[],
        GLOBAL_RUNNING_STEP=[],
        title="mountaincar"):
    plt.subplot(2, 1, 1)
    plt.title(title)  # 表头
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)),
             GLOBAL_RUNNING_R)  # 结束后绘制reward图像
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(GLOBAL_RUNNING_STEP)),
             GLOBAL_RUNNING_STEP)  # 结束后绘制reward图像
    plt.xlabel('episode')
    plt.ylabel('step')
    plt.show()
    time = datetime.datetime.now()
    statistical_data = rank_and_average(GLOBAL_RUNNING_R)
    with open('TEST_RECORD.csv', 'a', newline='') as myFile:
        Writer = csv.writer(myFile, dialect='excel')
        Writer.writerow([title] + [str(time)])
        Writer.writerow(GLOBAL_RUNNING_STEP)
        Writer.writerow(GLOBAL_RUNNING_R)
        Writer.writerow(statistical_data)

    print("write over")


def plot_error_bar(data):
    num_of_methods = data.shape[0]
    average = data[:, 2]
    average = list(map(float, average))
    std_error = data[:, 3]
    std_error = list(map(float, std_error))
    fig, ax = plt.subplots()

    # 画误差线，x轴一共7项，y轴显示平均值，y轴误差为标准差
    ax.errorbar(np.arange(num_of_methods), average,
                yerr=std_error,
                fmt="o", color="blue", ecolor='grey', elinewidth=2, capsize=4)
    ax.set_xticks(np.arange(num_of_methods))
    ax.set_xticklabels(['method1', 'method2'])  # 设置x轴刻度标签，并使其倾斜45度，不至于重叠
    plt.title("Comparison")
    plt.ylabel("Average Reward")
    plt.show()


def rank_and_average(GLOBAL_RUNNING_R=[]):
    minInRecord = min(GLOBAL_RUNNING_R)
    maxInRecord = max(GLOBAL_RUNNING_R)
    average = np.mean(GLOBAL_RUNNING_R)
    std_error = np.sqrt(np.var(GLOBAL_RUNNING_R)) / \
        np.sqrt(np.size(GLOBAL_RUNNING_R))
    print("Max = %.2f  Min = %.2f  Average = %.2f  std_error = %.2f"
          % (maxInRecord, minInRecord, average, std_error))
    statistical_data = [maxInRecord, minInRecord, average, std_error]
    return statistical_data


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)
