import numpy as np
import tensorflow as tf
import tensorflow.keras as keras # type: ignore
from tensorflow.keras import layers # type: ignore
from collections import deque
import random

class DDPGAgent:
    def __init__(self, state_dim, action_dim, min_steps_to_learn=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.memory = deque(maxlen=2000)
        self.batch_size = 64
        self.min_steps_to_learn = min_steps_to_learn  # Minimum number of steps to start learning

        
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        
        self.update_target(self.target_actor.variables, self.actor.variables, tau=1.0)
        self.update_target(self.target_critic.variables, self.critic.variables, tau=1.0)
    
    # def build_actor(self):
    #     initializer = tf.keras.initializers.HeNormal()  # Use He initialization
    #     inputs = layers.Input(shape=(self.state_dim,))
    #     out = layers.Dense(256, activation='relu', kernel_initializer=initializer)(inputs)
    #     out = layers.Dense(256, activation='relu', kernel_initializer=initializer)(out)
    #     outputs = layers.Dense(self.action_dim, activation='sigmoid', kernel_initializer=initializer)(out)
    #     outputs = layers.Lambda(lambda x: x * 10)(outputs)  # Scale actions to range [-10, 10]
    #     model = keras.Model(inputs, outputs)
    #     return model

    def build_actor(self):
        initializer = tf.keras.initializers.HeNormal()  # Use He initialization
        inputs = layers.Input(shape=(self.state_dim,))
        out = layers.Dense(256, activation='relu', kernel_initializer=initializer)(inputs)
        out = layers.Dense(256, activation='relu', kernel_initializer=initializer)(out)
        outputs = layers.Dense(self.action_dim, activation='sigmoid', kernel_initializer=initializer)(out)
        outputs = layers.Lambda(lambda x: x * 0.2)(outputs)  # Scale actions to range [-10, 10]
        model = keras.Model(inputs, outputs)
        return model
    
    # def build_actor(self):
    #     initializer = tf.keras.initializers.HeNormal()  # Use He initialization
    #     inputs = layers.Input(shape=(self.state_dim,))
    #     out = layers.Dense(256, activation='relu', kernel_initializer=initializer)(inputs)
    #     out = layers.Dense(256, activation='relu', kernel_initializer=initializer)(out)
        
    #     # Two separate outputs with different scaling
    #     output1 = layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)(out)  # Output in [0, 1]
    #     output2 = layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)(out)  # Output in [0, 1]
        
    #     # Scale output2 to be in [1, 2]
    #     output2 = layers.Lambda(lambda x: x + 1)(output2)
        
    #     # Concatenate both outputs
    #     outputs = layers.Concatenate()([output1, output2])
        
    #     model = keras.Model(inputs, outputs)
    #     return model

    def build_critic(self):
        initializer = tf.keras.initializers.HeNormal()  # Use He initialization
        state_inputs = layers.Input(shape=(self.state_dim,))
        action_inputs = layers.Input(shape=(self.action_dim,))
        concat = layers.Concatenate()([state_inputs, action_inputs])
        out = layers.Dense(256, activation='relu', kernel_initializer=initializer)(concat)
        out = layers.Dense(256, activation='relu', kernel_initializer=initializer)(out)
        outputs = layers.Dense(1, kernel_initializer=initializer)(out)
        model = keras.Model([state_inputs, action_inputs], outputs)
        return model
    
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    # def get_action(self, state, current_step):
    #     if current_step < self.min_steps_to_learn:
    #         return np.zeros(self.action_dim)  # Return zeros if not enough steps have been taken
    #     state = np.reshape(state, (1, self.state_dim))
    #     action = self.actor(state).numpy()[0]
    #     noise = np.random.uniform(0, 2, self.action_dim)  # Add random noise from 0 to 0.1
    #     action = action + noise
    #     return np.clip(action, 0, 10)  # Clip actions to range [0, 100]
    #     # return action  # No need to clip, as action space is [0, inf]
    
    def get_action(self, state, current_step):
        if current_step < self.min_steps_to_learn:
            # return np.array([1.0, 1.0])  # Return initial values within the desired ranges
            return np.zeros(self.action_dim)  # Return zeros if not enough steps have been taken
        state = np.reshape(state, (1, self.state_dim))
        action = self.actor(state).numpy()[0]
        noise = np.random.uniform(0, 0.01, self.action_dim)  # Add random noise from 0 to 0.1
        action = action + noise
        return np.clip(action, 0, 0.2)  # Clip actions to range [0, 100]
        # return action  # No need to clip, as action space is [0, inf]
    
    # def get_action(self, state, current_step):
    #     if current_step < self.min_steps_to_learn:
    #         return np.array([1.0, 1.0])  # Return initial values within the desired ranges
    #     state = np.reshape(state, (1, self.state_dim))
    #     action = self.actor(state).numpy()[0]
    #     noise = np.random.uniform(0, 0.1, self.action_dim)  # Add random noise to both outputs
    #     action[0] -= noise[0]
    #     action[1] += noise[1]
        
    #     # Ensure that the first value stays within [0, 1] and the second within [1, 2]
    #     action[0] = np.clip(action[0], 0, 1)
    #     action[1] = np.clip(action[1], 1, 2)
        
    #     return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        next_states = np.vstack(next_states)
        dones = np.vstack(dones)
        
        target_actions = self.target_actor(next_states)
        target_q = self.target_critic([next_states, target_actions])
        target_q = rewards + (1 - dones) * self.gamma * target_q
        
        with tf.GradientTape() as tape:
            q = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(target_q - q))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            critic_value = self.critic([states, actions])
            actor_loss = -tf.reduce_mean(critic_value)
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        self.update_target(self.target_actor.variables, self.actor.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic.variables, self.tau)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0  # Epsilon for epsilon-greedy strategy
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.memory = deque(maxlen=2000)
        self.batch_size = 64

        self.q_network = self.build_q_network()
        self.target_q_network = self.build_q_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.q_network.compile(optimizer=self.optimizer, loss='mse')

        self.update_target_network()

    def build_q_network(self):
        initializer = tf.keras.initializers.HeNormal()
        inputs = layers.Input(shape=(self.state_dim,))
        out = layers.Dense(256, activation='relu', kernel_initializer=initializer)(inputs)
        out = layers.Dense(256, activation='relu', kernel_initializer=initializer)(out)
        outputs = layers.Dense(self.action_dim, kernel_initializer=initializer)(out)  # Outputs Q-values for each action
        model = keras.Model(inputs, outputs)
        return model

    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)  # Choose a random action
        state = np.reshape(state, [1, -1])
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])  # Choose the action with the highest Q-value

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        q_values = self.q_network.predict(states, verbose=0)
        target_q_values = self.target_q_network.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_q_values[i])

        self.q_network.fit(states, q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay