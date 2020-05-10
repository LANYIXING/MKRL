import gym
import time
# GAME = 'HalfCheetah-v1'
# GAME = 'Hopper-v1'
GAME = 'Pendulum-v0'
GAME = 'CartPole-v0'
GAME = 'MountainCar-v0'
GAME = 'Acrobot-v1'

env = gym.make(GAME)
env.reset()
env.render()
time.sleep(100)
env.close()