import pandas as pd
import numpy as np
import time
import tensorflow as tf
from src.daily_solving import solve_daily_optimization
from src.period_solving import solve_period_optimization

# Load the data
load_data = pd.read_csv('data/CAISO_Load_2022.csv', usecols=['net_load']).values.flatten()
thermal_gen_data = pd.read_csv('data/thermal_gen_offer.csv', usecols=['capacity_MW', 'energy_price'])
model = tf.keras.models.load_model('models/model0')


# Parameters
num_days = 365
num_steps = 288
num_intervals = 12
P = 1  # Total storage capacity in MW
E = 4  # Total storage capacity in MWh
eta = 0.9  # Storage one-way efficiency
C_s = 20.0  # Storage marginal discharge cost

# Generator data 
P_g = thermal_gen_data['capacity_MW'].values # Thermal generator capacities (MW)
C_g = thermal_gen_data['energy_price'].values # Thermal generator offer ($/MW)

# Storage segment data
num_seg = 1

# Reshape load data for num_days of num_steps
load = load_data[0:num_days*num_steps].reshape((num_days, num_steps)) # Net load profile

# Number of generators
num_gen = len(P_g)

# Solve for each time step
all_daily_costs = []
all_gen_output = []
all_storage_ops = []
all_clearing_prices = []
all_soc_bids = []
storage_profit_cum = 0

total_start_time = time.time()  # Start time for the entire solving loop

for day in range(num_days):
    # solve daily for first day (288 periods) to generate initial price signals
    L = load[day, :]
    if day == 0:
        day = 0
        E0 = E / 2
        daily_cost, daily_gen, daily_charge, daily_discharge, daily_soc, dual_prices, run_time = solve_daily_optimization(L, P_g, C_g, P, E, eta, C_s, E0, num_gen, num_steps, num_intervals)
        print(f"Day {day + 1}: Cost = {daily_cost}, Run Time = {run_time} seconds")
    
        # Collect results for each day
        all_daily_costs.append({
            'day': day + 1,
            'cost': daily_cost,
            'run_time': run_time
        })
    
        all_gen_output.append(pd.DataFrame(daily_gen, columns=[f'gen_{i+1}' for i in range(num_gen)]))
        all_storage_ops.append(pd.DataFrame({
            'charge': daily_charge,
            'discharge': daily_discharge,
            'soc': daily_soc
        }))
        all_clearing_prices.append(pd.DataFrame({'clearing_price': dual_prices}))
    else:
        for ts in range(num_steps):
            if (day == 1) & (ts == 0):
                E0 = [0.2 * E, 0.2 * E, 0.1 * E, 0.0, 0.0]
                predictors = np.array(dual_prices).reshape(1, 288)
            else:
                E0 = list(step_soc)
                predictors = np.hstack((predictors[:, 1:], np.array([[dual_price]])))
            L_ts = L[ts]
            v = model.predict(predictors, verbose=0).T
            soc_bids = np.mean(v.reshape(num_seg, int(v.shape[0]/num_seg)), axis=1, keepdims=True).flatten()
            step_cost, step_gen, step_charge, step_discharge, step_soc, dual_price, storage_profit, run_time = solve_period_optimization(L_ts, P_g, C_g, P, E, eta, C_s, E0, num_intervals, num_gen, num_seg, soc_bids)
            storage_profit_cum += storage_profit/E*4
            print(f"Day {day + 1}: Step = {ts} Storage Profit = {storage_profit_cum}")

            all_gen_output.append(pd.DataFrame(step_gen.reshape(1,num_gen), columns=[f'gen_{i+1}' for i in range(num_gen)]))
            all_storage_ops.append(pd.DataFrame({
                'charge': [sum(step_charge)],
                'discharge': [sum(step_discharge)],
                'soc': [sum(step_soc)]
            }))
            all_clearing_prices.append(pd.DataFrame({'clearing_price': [dual_price]}))
            all_soc_bids.append(soc_bids)



total_end_time = time.time()  # End time for the entire solving loop
total_run_time = total_end_time - total_start_time

print(f"Total Run Time for all days: {total_run_time} seconds")

# Generator outputs
all_gen_output_df = pd.concat(all_gen_output)
all_gen_output_df.to_csv('results/ed/generator_outputs.csv')

# Storage operations
all_storage_ops_df = pd.concat(all_storage_ops)
all_storage_ops_df.to_csv('results/ed/storage_operations.csv')

# Clearing prices
all_clearing_prices_df = pd.concat(all_clearing_prices)
all_clearing_prices_df.to_csv('results/ed/clearing_prices.csv')

soc_bids_df = pd.DataFrame(all_soc_bids, columns=[f'soc_bid_{i+1}' for i in range(num_seg)])
soc_bids_df.to_csv('results/ed/soc_bids.csv')
