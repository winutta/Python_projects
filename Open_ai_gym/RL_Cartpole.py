import gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline


from keras.models import model_from_json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from collections import deque
import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D
from keras.optimizers import Adam
import random
from keras.models import load_model
from keras import backend as K

LEARNING_RATE = 0.045
env = gym.make("CartPole-v0")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
# This method seems to overtrain so that only one action (1) is predicted
model1 = Sequential()

model1.add(Dense(12, input_shape = (1,observation_space,), activation="relu"))
model1.add(BatchNormalization())
#model.add(Dropout(rate = 0.2))
model1.add(Dense(32, activation = 'relu'))
model1.add(BatchNormalization())
#model.add(Dropout(rate = 0.2))
model1.add(Dense(12, activation = 'relu'))
model1.add(Dense(action_space,activation = 'linear'))
model1.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
model3 = model1

game_count1 = 0
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]
#print(observation_space)

state = env.reset()
memory1 = deque(maxlen = 1000000)
#obs = env.step(env.action_space.sample())
s = 0
save = 0
for _ in range(100000):
    state = np.reshape(state, [1,1,observation_space])
    action = np.argmax(model1.predict(state)[0][0])
    #r = (100)/(game_count+1)
    r = -(1/(6000**2))*(game_count1**2)+1
    #r = (1-((game_count1+1)/(game_count1+100)))
    #r = 0.99**(game_count1)
    r = np.clip(r,0.02,1)
    # r could also be (game_count+1)/(game_count+100)

    if np.random.uniform(0,1) < r :
        action = env.action_space.sample()
    #env.render()
    new_state, reward, done, info = env.step(action)
    
    new_state = np.reshape(new_state, [1,1,observation_space])
    memory1.append((new_state,state,reward,action))
    state = new_state
   
    memory_len = len(memory1)
    if ((memory_len >= 20)):
        
        rank = np.arange(memory_len,0,-1)
        #print(rank)
        rank = 1/rank
        rank = rank**0.6
        #print(rank)
        total = np.sum(rank)
        numerator = rank
        dist = numerator/total
        
        #print(memory_len)
        cum_dist = np.cumsum(dist)
        '''
        cum_dist = [0]*memory_len
        #print(dist)
        #print(cum_dist)
        cum_dist[0] = dist[0]
        for i in range(1,len(cum_dist)):
            #print(cum_dist[i-1])
            #print(dist[i])
            cum_dist[i] = cum_dist[i-1] + dist[i]
        #print(cum_dist)
        '''
        p = int(np.ceil(memory_len**(2/np.pi)))
        p = np.clip(p,1,20)
        p = int(p)
        
        unif = np.random.uniform(0,1,size = p)
        choices = []
        for u in range(p):
            choi = 0 
            for j in range(p):
                if unif[u] > cum_dist[j]:
                    choi += 1
            choices.append(choi)
        #print(choices)    
        temp_mem = np.array(memory1)[choices]
        #for q in choices:
        #    temp_mem.append(memory1[q])
        #plt.bar(x = range(0,len(dist)), height = dist)
        #plt.show()
        #temp_mem = random.sample(memory1,20)
        #q_val_list = deque(maxlen = int(p))
        #states = deque(maxlen = int(p))
        q_val_list = []
        states = []
        
        
        
        
        for new_state, state, reward, action in temp_mem:
            
            if done == True:
                update = reward
            else:
                update = reward + 0.95*(np.amax(model3.predict(new_state)[0][0]))
            q_vals = model3.predict(state)[0][0]
            q_vals[action] = update
            q_vals = np.reshape(q_vals,(1,1,2))
            q_val_list.append(q_vals)
            states.append(state)
            #model1.fit(state,q_vals,verbose = 0)
            #print(q_vals)
        #print(np.array(states))
        #states = np.reshape(np.array(states),[20,1,4])
        #states = np.array([item for sublist in states for item in sublist])
        states = np.reshape(states, (p,1,4))
        q_val_list = np.reshape(q_val_list, (p,1,2))
        #print(states)
        #print(q_val_list)
        #print('_____________')
        #states = temp_mem[]
        model1.fit(states,q_val_list,epochs = len(temp_mem),verbose = 0)
        
    if done == True:
        s += 1
        game_count1 += 1
        #print(_-save)
        model1.save('my_model3.h5')
        K.clear_session()
        model1 = load_model('my_model3.h5')
        model3 = model1

        print('Game Count: ' + str(game_count1) + ', length of game: ' + str(_-save))
        save = _
        state = env.reset()
        continue