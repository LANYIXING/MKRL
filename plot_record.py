"""
画图&记录
调用之前定义两个列表 GLOBAL_RUNNING_R  GLOBAL_RUNNING_STEP
"""
# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime
import pandas as pd


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
    with open('A3C_TEST_RECORD.csv', 'a', newline='') as myFile:
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


if __name__ == '__main__':
    a = [1]
    print(len(a) == 0)
    a = np.vstack(a)