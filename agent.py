#agent source code goes here...
import numpy as np
import matplotlib.pyplot as plt

#And then for the agent
from tensorflow.keras import Model, layers
import tensorflow as tf


# create agent class
class Agent():
    def __init__(self, N, gamma, epsilon, lr,
                 update_target, batchSize, nDefects,
                 nActions=5, maxMem=45, epsEnd=0.01,
                 epsDec=0.005, ddqn = False):
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

        # call and evaluate using DQN class (and make target)
        self.q_eval = DQN_Keras(input_image_dim=(self.N, self.N, 2),
                                input_numeric_dim=(self.nDefects * 5),
                                input_nDefects=self.nDefects,
                                dim_actions=self.nActions, actor_lr=self.lr)
        self.q_eval.compile()
        if ddqn:
            self.q_eval_target = DQN_Keras(input_image_dim=(self.N, self.N, 2),
                                           input_numeric_dim=(self.nDefects * 5), input_nDefects=self.nDefects,
                                           dim_actions=self.nActions, actor_lr=self.lr)
            self.q_eval_target.compile()

    def choose_action(self, obsv):
        if np.random.random() > self.epsilon:
            qvals = self.q_eval(obsv)
            action = np.argmax(qvals)
        else:
            action = np.random.choice(np.arange(self.nActions), size=1)
        self.action = action
        return action

    def store_transitions(self, transition):
        self.episode_history.append(transition)
        self.episode_history = self.episode_history[-self.memsize:]

    def learn(self):
    
        size_to_select = min(len(self.episode_history), self.batchSize)

        # experience replay
        batch = np.random.choice(np.arange(len(self.episode_history)),
                          size_to_select, replace=False)

        q_values_list, state_buffer, action_hist = [], [],[]
        batch_history = np.array(self.episode_history, dtype = 'object')

        for state, action, reward, done, next_state in batch_history[batch]:
            
            #in DDQN:
            #Get the Q values given by the target network, but teh actions given by present network
            #The difference between teh Q values predicted are then the error
            #We do this for the chosen actions only

            state_buffer.append(state)
            if self.ddqn:
                #get actions by Q-network
                qnet_qvals = self.q_eval(next_state)
                qnet_action = np.argmax(qnet_qvals)

                # calculate estimated Q-values with qnet_actions by using Target-network
                tnet_q_value = self.q_eval_target(next_state)[0][qnet_action]
                #print('tnet_q_value is {} and qnet_action is {}'.format(tnet_q_value, qnet_action))

                q_update = reward + self.gamma * tnet_q_value * done
                
            else:
                # qval (s,a) = reward + gamma*max(a')qval(s',a')
                qvals_initial = self.q_eval(state)[0]
                qvals_initial = qvals_initial.numpy()
                q_update = reward + self.gamma * np.max(self.q_eval(next_state)[0]) * done
                qvals_initial[action] = q_update
                q_update = qvals_initial
            
            q_values_list.append(q_update)
            
            try:
                action_hist.append(action[0])
            except:
                action_hist.append(action)
                
        q_values_list = tf.stack(q_values_list)
        
        if self.ddqn:
        
            selected_idx = np.vstack([np.arange(len(action_hist)), action_hist])

            with tf.GradientTape() as tape:

                q_output = tf.squeeze(tf.stack([self.q_eval(state)for state in state_buffer]))
                #print('q output shape is {} and action history is {}'.format(q_output.shape, action_hist))
                q_output = tf.gather_nd(q_output, selected_idx.T)
                loss = tf.square(q_output - q_values_list)
                valueGradient = tape.gradient(loss, self.q_eval.trainable_variables)
        else:
            with tf.GradientTape() as tape:
                q_output = tf.squeeze(tf.stack([self.q_eval(state)for state in state_buffer]))
                loss = tf.square(q_output - q_values_list)
                valueGradient = tape.gradient(loss, self.q_eval.trainable_variables)
            
        self.q_eval.optimizer.apply_gradients(zip(valueGradient, self.q_eval.trainable_variables))
        self.epsilon = self.epsilon - self.epsDec if self.epsilon > self.epsMin else self.epsMin
        
        if self.episode_number % self.update_target and self.ddqn:
            self.q_eval_target.set_weights(self.q_eval.get_weights())

        return loss
    
    
# create DQN class
class DQN_Keras(Model):
    def __init__(self, input_image_dim, input_numeric_dim,
                 input_nDefects=5, input_step_dim=1,
                 dim_actions=5, num_conv_filters_1=32,
                 num_conv_filters_2=32, num_hidden_nodes_1=256,
                 num_hidden_nodes_2=256,
                 actor_lr=0.001):
        
        # input_image_dim = dimension of image input
        # input_numeric_dim = dimension of numerical input
        # input_ndefects: number of defects. 
        # We will pick a defect and move it, so we need this for calculating argmax (Q)
        
        self.nDefects = input_nDefects
        self.num_conv_filter_1 = num_conv_filters_1
        self.num_conv_filter_2 = num_conv_filters_2
        self.initializer = tf.keras.initializers.he_uniform()
        self.num_hidden_nodes_1 = num_hidden_nodes_1
        self.num_hidden_nodes_2 = num_hidden_nodes_2
        self.dim_actions = dim_actions
        self.actor_lr = actor_lr
        self.input_image_dim = input_image_dim
        self.input_numeric_dim = input_numeric_dim
        self.input_step_dim = input_step_dim
        self.input_nDefects = input_nDefects
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
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
        cnet = layers.BatchNormalization()(cnet)
        cnet = layers.Conv2D(self.num_conv_filter_2, kernel_size=(4, 4), strides=(1, 1), activation=tf.nn.relu,
                             kernel_initializer=self.initializer, name='conv2')(cnet)
        cnet = layers.BatchNormalization()(cnet)
        cnet = layers.Flatten()(cnet)

        cnet = Model(inputs=InputImage, outputs=cnet)

        numeric = layers.Dense(self.num_hidden_nodes_1, activation=tf.nn.relu,
                               kernel_initializer=self.initializer)(InputNumeric)
        numeric = layers.BatchNormalization()(numeric)
        numeric = layers.Dense(self.num_hidden_nodes_2, activation=tf.nn.relu,
                               kernel_initializer=self.initializer)(numeric)
        numeric = layers.BatchNormalization()(numeric)
        numeric = Model(inputs=InputNumeric, outputs=numeric)

        step = layers.Dense(self.num_hidden_nodes_1, activation = tf.nn.relu,
                            kernel_initializer=self.initializer)(InputStep)

        step = Model(inputs=InputStep, outputs = step)

        combined = layers.concatenate([cnet.output, numeric.output, step.output])

        combined_network = layers.Dense(self.num_hidden_nodes_1, activation=tf.nn.relu,
                                        kernel_initializer=self.initializer)(combined)

        combined_network = layers.BatchNormalization()(combined_network)

        combined_network = layers.Dense(self.num_hidden_nodes_2, activation=tf.nn.relu,
                                        kernel_initializer=self.initializer)(combined_network)

        combined_network = layers.BatchNormalization()(combined_network)

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