#agent source code goes here...
import numpy as np
import matplotlib.pyplot as plt
import random

#And then for the agent
from tensorflow.keras import Model, layers
import tensorflow as tf
import math


# create agent class
class Agent():
    def __init__(self, N, gamma, epsilon, lr,
                 update_target, batchSize, nDefects,
                 nActions=5, maxMem=45, epsEnd=0.01,
                 epsDec=0.005, ddqn = False, batch_norm = False):
        self.N = N
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.update_target = update_target
        self.batchSize = batchSize
        self.epsMin = epsEnd
        self.epsDec = epsDec
        self.memsize = maxMem
        self.nActions = nActions
        self.nDefects = nDefects
        self.qvals_list = []
        self.episode_history = []
        self.episode_number = 0
        self.memcntr = 0
        self.ddqn = ddqn
        self.batch_norm = batch_norm

        # call and evaluate using DQN class (and make target)
        self.q_eval = DQN_Keras(input_image_dim=(self.N, self.N, 2),
                                input_numeric_dim=(self.nDefects * 5),
                                input_nDefects=self.nDefects,
                                dim_actions=self.nActions, actor_lr=self.lr)
        self.q_eval.compile()
        if ddqn:
            self.q_eval_target = DQN_Keras(input_image_dim=(self.N, self.N, 2),
                                           input_numeric_dim=(self.nDefects * 5), input_nDefects=self.nDefects,
                                           dim_actions=self.nActions, actor_lr=self.lr, batch_norm = self.batch_norm)
            self.q_eval_target.compile()

    def choose_action(self, obsv):
        is_valid = False
        if np.random.random() > self.epsilon:
            qvals = self.q_eval(obsv)
            action = np.argmax(qvals)
            action_possible = np.argsort(qvals) #sorted in order
            action_possible = action_possible[0][::-1]
            i=0
            while not is_valid:
                test_action = action_possible[i]
                is_valid = self.check_action_is_valid(obsv, test_action)
                i+=1
        else:
            action = np.random.choice(np.arange(self.nActions), size=1)
            while not is_valid:
                test_action = np.random.choice(np.arange(self.nActions), size=1)
                is_valid = self.check_action_is_valid(obsv, test_action)
        action = test_action        
        self.action = action
        return action

    def check_action_is_valid(self, obsv, action):
        #This function returns whether an action is valid based on the observat
        
        #recall tha the observation contians the positions of the defects.
        #print('action is {}'.format(action))
        idx = math.floor((action-1)/(4)) 
        mov_dir = (action-1)%4
        N = obsv[0].shape[1]
        if action==0: 
            is_valid = True
            return is_valid
        else:
            #print('action is {}, idx is {} and defect chosen idx is {}'.format(action, idx, defect_chosen_idx))
            defect_chosen_idx = list(obsv[1][:,-1]).index(idx)
            row_d, col_d = obsv[1][defect_chosen_idx][:2]
        
            if mov_dir == 0:
                new_col_d = (col_d - 1) % N  # Left
                new_row_d = row_d
            elif mov_dir == 1:
                new_row_d = (row_d - 1) % N  # Up
                new_col_d = col_d
            elif mov_dir == 2:
                new_col_d = (col_d + 1) % N # Right
                new_row_d = row_d
            elif mov_dir == 3:
                new_row_d = (row_d + 1) % N # Down
                new_col_d = col_d

            new_pixind_loc = self.xy_to_pixind(new_row_d, new_col_d,N)
            defect_locations = [self.xy_to_pixind(pos[0], pos[1],N) \
                                    for pos in obsv[1]]

            #print('Proposed location is {} and existing defect locations are {}'.format(new_pixind_loc, 
            #defect_locations))
            if new_pixind_loc not in defect_locations:
                    # Action is now valid
                is_valid=True
                
            else:
                #else it's not valid...
                is_valid=False
        #print('is_valid {}'.format(is_valid))
        return is_valid

    def xy_to_pixind(self, row, col, N):
        return row * N + col

    def store_transitions(self, transition):
        self.episode_history.append(transition)
        self.episode_history = self.episode_history[-self.memsize:]

    def learn(self):
    
        size_to_select = min(len(self.episode_history), self.batchSize)

        minibatch = random.sample(self.episode_history, size_to_select)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.q_eval.predict(next_state)[0]))
            
            target_f = self.q_eval(state)[0]
            target_f = target_f.numpy()
            #print('action is {}'.format(action))
            target_f[action] = target 
            # Filtering out states and targets for training
            states.append(state)
            targets_f.append(target_f)
        
        with tf.GradientTape() as tape:
            targets = tf.squeeze(tf.stack([self.q_eval(state) for state in states]))
            targets_true = tf.stack(targets_f)
            loss = tf.reduce_mean(targets_true - targets)**2
            Qgradients = tape.gradient(loss, self.q_eval.trainable_variables)
            self.q_eval.optimizer.apply_gradients(zip(Qgradients, self.q_eval.trainable_variables))
       
        self.epsilon = self.epsilon - self.epsDec if self.epsilon > self.epsMin else self.epsMin

        return loss
    
    
# create DQN class
class DQN_Keras(Model):
    def __init__(self, input_image_dim, input_numeric_dim,
                 input_nDefects=5, input_step_dim=1,
                 dim_actions=5, num_conv_filters_1=32,
                 num_conv_filters_2=32, num_hidden_nodes_1=256,
                 num_hidden_nodes_2=256,
                 actor_lr=0.001, batch_norm = False):
        
        # input_image_dim = dimension of image input
        # input_numeric_dim = dimension of numerical input
        # input_ndefects: number of defects. 
        # We will pick a defect and move it, so we need this for calculating argmax (Q)
        
        self.nDefects = input_nDefects
        self.num_conv_filter_1 = num_conv_filters_1
        self.num_conv_filter_2 = num_conv_filters_2
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.num_hidden_nodes_1 = num_hidden_nodes_1
        self.num_hidden_nodes_2 = num_hidden_nodes_2
        self.dim_actions = dim_actions
        self.actor_lr = actor_lr
        self.input_image_dim = input_image_dim
        self.input_numeric_dim = input_numeric_dim
        self.input_step_dim = input_step_dim
        self.input_nDefects = input_nDefects
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.batch_norm = batch_norm
        super(DQN_Keras, self).__init__()
        self.actor = self.build_actor()
        self.actor.compile(loss=tf.keras.losses.mean_squared_error, optimizer=self.optimizer,
                           metrics=['accuracy'])
        

    def build_actor(self):
        InputImage = layers.Input(shape=self.input_image_dim)
        InputNumeric = layers.Input(shape=self.input_numeric_dim)
        InputStep = layers.Input(shape=self.input_step_dim)

        cnet = layers.Conv2D(filters=self.num_conv_filter_1, kernel_size=(2, 2), strides=(1, 1),
                             activation=tf.nn.relu,
                             kernel_initializer=self.initializer, 
                             name='conv1')(InputImage)
        if self.batch_norm: cnet = layers.BatchNormalization()(cnet)
        cnet = layers.Conv2D(self.num_conv_filter_2, kernel_size=(4, 4), strides=(1, 1), activation=tf.nn.relu,
                             kernel_initializer=self.initializer, name='conv2')(cnet)
        if self.batch_norm: cnet = layers.BatchNormalization()(cnet)
        cnet = layers.Flatten()(cnet)

        cnet = Model(inputs=InputImage, outputs=cnet)

        numeric = layers.Dense(self.num_hidden_nodes_1, activation=tf.nn.relu,
                               kernel_initializer=self.initializer)(InputNumeric)
        if self.batch_norm:  numeric = layers.BatchNormalization()(numeric)
        numeric = layers.Dense(self.num_hidden_nodes_2, activation=tf.nn.relu,
                               kernel_initializer=self.initializer)(numeric)
        if self.batch_norm: numeric = layers.BatchNormalization()(numeric)
        numeric = Model(inputs=InputNumeric, outputs=numeric)

        step = layers.Dense(self.num_hidden_nodes_1, activation = tf.nn.relu,
                            kernel_initializer=self.initializer)(InputStep)

        step = Model(inputs=InputStep, outputs = step)

        combined = layers.concatenate([cnet.output, numeric.output, step.output])

        combined_network = layers.Dense(self.num_hidden_nodes_1, activation=tf.nn.relu,
                                        kernel_initializer=self.initializer)(combined)

        if self.batch_norm: combined_network = layers.BatchNormalization()(combined_network)

        combined_network = layers.Dense(self.num_hidden_nodes_2, activation=tf.nn.relu,
                                        kernel_initializer=self.initializer)(combined_network)

        if self.batch_norm: combined_network = layers.BatchNormalization()(combined_network)

        combined_network = layers.Dense(self.dim_actions, activation='linear',
                                        kernel_initializer=self.initializer,
                                        name='output_qvalues')(combined_network)

        actor = Model(inputs=[cnet.input, numeric.input, step.input], outputs=combined_network)

        return actor

    # define forward pass
    def call(self, states):

        state0 = states[0]
        state1 = states[1]
        state2 = np.array([states[2]])

        if state0.shape[0] != 1 :
            state0 = state0[None,:,:,:]
        if state1.shape[0] != 1:
            state1 = state1[None, :, :]
        if len(state2.shape)==1:
            state2 = state2[None,:]

        state0 = tf.reshape(state0, (state0.shape[0], state0.shape[2], state0.shape[3], state0.shape[1]))
        state1 = tf.reshape(state1, (state1.shape[0], -1))

        states = [state0, state1, state2]
        actions_output = self.actor(states)

        # this returns the qvalues
        return actions_output