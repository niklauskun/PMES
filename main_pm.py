import pandas as pd
import numpy as np
import tensorflow as tf
from src.environment import EnergyStorageEnv
from src.agent import DDPGAgent, DQNAgent
from src.utils import create_initial_state

# Set the algorithm you want to use: 'ddpg' or 'dqn'
# algorithm = 'ddpg'
algorithm = 'dqn'  

# Simulation Parameters
num_days = 365
num_steps = 288
num_intervals = 12

# Load the data
net_load_data = pd.read_csv('data/CAISO_Load_2022.csv', usecols=['net_load']).values.flatten()
prices_file = 'data/initial_state/clearing_prices.csv'
load_file = 'data/initial_state/CAISO_Load_Ini.csv'
operations_file = 'data/initial_state/storage_operations.csv'
initial_state_data = create_initial_state(prices_file, load_file, operations_file)

# Generator data 
thermal_gen_data = pd.read_csv('data/thermal_gen_offer.csv')
P_g = thermal_gen_data['capacity_MW'].values
C_g = thermal_gen_data['energy_price'].values
num_gen = len(P_g)

# Storage parameters
num_segments = 1
P = 10000  # Total storage capacity in MW
E = 40000  # Total storage capacity in MWh
eta = 0.9  # Storage one-way efficiency
C_s = 20.0  # Storage marginal discharge cost
model = tf.keras.models.load_model('models/model0.h5')

# Define the minimum steps to start learning
min_steps_to_learn = 2016
env = EnergyStorageEnv(algorithm='dqn', num_segments=num_segments, initial_state=initial_state_data, net_load_data=net_load_data, P_g=P_g, C_g=C_g, P=P, E=E, eta=eta, C_s=C_s, num_intervals=num_intervals, num_gen=num_gen, model=model)
state_dim = env.observation_space.shape[0]

# Choose the agent based on the algorithm parameter
if algorithm == 'ddpg':
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, action_dim, min_steps_to_learn=min_steps_to_learn)
elif algorithm == 'dqn':
    action_dim = env.action_space.n  # For DQN, the action space should be discrete
    agent = DQNAgent(state_dim, action_dim)

episodes = 52
steps_per_episode = 288 * 7

# Initialize list to store step data
step_data = []

for episode in range(episodes):
    if episode == 0:
        state = env.reset(initial_state=initial_state_data)
    else:
        state = env.reset(initial_state=state)
    
    episode_reward = 0
    episode_revenue = 0
    
    for step in range(steps_per_episode):
        current_step = step + episode * steps_per_episode

        # Select action based on the algorithm
        if algorithm == 'ddpg':
            action = agent.get_action(state, current_step)
        elif algorithm == 'dqn':
            action = agent.act(state)

        next_state, reward, revenue, done, discharge, charge, real_time_price, unadjusted_charge_bid, unadjusted_discharge_bid, adjusted_charge_bid, adjusted_discharge_bid = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        episode_reward += reward
        episode_revenue += revenue
        
        # Record step data
        step_data.append([
            episode, step, discharge, charge, real_time_price,
            unadjusted_charge_bid, unadjusted_discharge_bid,
            adjusted_charge_bid, adjusted_discharge_bid,
            action, reward, revenue
        ])
        
        # Print the action taken at each step
        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}, Revenue: {episode_revenue}")
        
        if done:
            break
    
    # Update the target network at the end of each episode
    if algorithm == 'dqn':
        agent.update_target_network()
    
    print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}, Revenue: {episode_revenue}")

    if env.current_step >= len(net_load_data):
        break

# Save step data to a CSV file
columns = [
    'Episode', 'Step', 'Discharge', 'Charge', 'Real_Time_Price',
    'Unadjusted_Charge_Bid', 'Unadjusted_Discharge_Bid',
    'Adjusted_Charge_Bid', 'Adjusted_Discharge_Bid',
    'Action_Charge_Adjustment', 'Reward', 'Revenue'
]
step_df = pd.DataFrame(step_data, columns=columns)
step_df.to_csv('results/step_data.csv', index=False)

print("Simulation complete. Step data saved to 'step_data.csv'.")