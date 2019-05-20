import gym
from gym.utils.play import play

#from gym.utils.play import *
'''
env = gym.make("CartPole-v0")
if hasattr(env, 'get_keys_to_action'):
    keys_to_action = env.get_keys_to_action()
elif hasattr(env.unwrapped, 'get_keys_to_action'):
    keys_to_action = env.unwrapped.get_keys_to_action()
print(keys_to_action)
'''

a = {(ord("a"),):0,(ord("d"),):1}
play(gym.make("CartPole-v0"),keys_to_action = a)
