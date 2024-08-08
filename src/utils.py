import pandas as pd
import numpy as np

def create_initial_state(prices_file, load_file, operations_file):
    prices_df = pd.read_csv(prices_file)
    load_df = pd.read_csv(load_file)
    operations_df = pd.read_csv(operations_file)
    
    last_288_prices = prices_df['clearing_price'].values[-288:]
    last_288_net_loads = load_df['net_load'].values[-288:]
    last_288_dispatches = operations_df['discharge'].values[-288:] - operations_df['charge'].values[-288:]
    last_soc = operations_df['soc'].values[-1]
    
    initial_state = np.zeros(865)
    initial_state[:288] = last_288_dispatches
    initial_state[288] = last_soc
    initial_state[289:577] = last_288_prices
    initial_state[577:865] = last_288_net_loads
    
    return initial_state