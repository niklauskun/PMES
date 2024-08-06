import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
import cvxpy as cp

# Custom Environment for Energy Storage Arbitrage
class EnergyStorageEnv(gym.Env):
    def __init__(self, prices, price_taker_bids):
        self.prices = prices
        self.price_taker_bids = price_taker_bids
        self.max_storage = 10.0  # Max storage capacity
        self.efficiency = 0.9  # Round-trip efficiency
        self.time_step = 0
        self.storage = 5.0  # Initial state of charge (SoC)
        self.action_space = gym.spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)  # Bidding prices for charge and discharge
        self.observation_space = gym.spaces.Box(low=0, high=np.max(prices), shape=(3,), dtype=np.float32)  # Include price-taker bid

    def reset(self):
        self.time_step = 0
        self.storage = 5.0
        return np.array([self.storage, self.prices[self.time_step], self.price_taker_bids[self.time_step]])

    def step(self, action):
        charge_bid, discharge_bid = action
        price = self.prices[self.time_step]

        # Market clearing optimization
        charge_power = cp.Variable()
        discharge_power = cp.Variable()

        constraints = [
            charge_power >= 0,
            discharge_power >= 0,
            charge_power * self.efficiency <= self.max_storage - self.storage,
            discharge_power / self.efficiency <= self.storage
        ]

        objective = cp.Minimize(charge_bid * charge_power - discharge_bid * discharge_power)
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Get the power values after market clearing
        charge_power = charge_power.value
        discharge_power = discharge_power.value

        # Calculate new state of charge (SoC)
        self.storage = min(self.max_storage, max(0, self.storage + charge_power * self.efficiency - discharge_power / self.efficiency))

        # Calculate reward based on the current price and power
        reward = charge_power * price - discharge_power * price

        self.time_step += 1
        done = self.time_step >= len(self.prices) - 1
        next_state = np.array([self.storage, self.prices[self.time_step], self.price_taker_bids[self.time_step]])

        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

# Actor-Critic Networks
def create_actor():
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(2, activation='linear')  # Bidding prices for charge and discharge
    ])
    return model

def create_critic():
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(1)
    ])
    return model

# Training the Actor-Critic Model
def train(env, actor, critic, price_taker_bids, episodes=1000, gamma=0.99, actor_lr=0.001, critic_lr=0.002, k=0.5):
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    mse_loss = tf.keras.losses.MeanSquaredError()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(len(env.prices) - 1):
            state = tf.convert_to_tensor([state], dtype=tf.float32)

            # Get policy action (a_A) from the actor network
            policy_action = actor(state).numpy()

            # Generate exploration action (a_E)
            exploration_action = np.random.uniform(low=-1, high=1, size=policy_action.shape)

            # Supervised action (a_S)
            supervised_action = price_taker_bids[step]

            # Combine actions
            action = (1 - k) * (policy_action + exploration_action) + k * supervised_action

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
            reward = tf.convert_to_tensor(reward, dtype=tf.float32)

            value = critic(state)
            next_value = critic(next_state)

            target = reward + gamma * next_value * (1 - int(done))
            td_error = target - value

            # Critic update
            with tf.GradientTape() as tape:
                critic_loss = td_error ** 2
            critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

            # Actor update with supervised action
            with tf.GradientTape() as actor_tape:
                actor_loss = mse_loss(actor(state), supervised_action)
            actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

            state = next_state.numpy()[0]
            if done:
                break

        print(f'Episode {episode + 1}: Total Reward: {total_reward}')

# Simulated price data and price-taker bids
prices = np.sin(np.linspace(0, 20, 100)) * 10 + 50
price_taker_bids = np.random.uniform(0, 100, size=(100, 2))  # Example price-taker bids for charge and discharge

env = EnergyStorageEnv(prices, price_taker_bids)
actor = create_actor()
critic = create_critic()
train(env, actor, critic, price_taker_bids)
