import numpy as np
import gym
from gym import spaces
import cvxpy as cp
import time
import tensorflow as tf

class EnergyStorageEnv(gym.Env):
    def __init__(self, num_segments=1, initial_state=None, net_load_data=None, P_g=None, C_g=None, P=1, E=4, eta=0.9, C_s=20.0, num_intervals=12, num_gen=1, model=None):
        super(EnergyStorageEnv, self).__init__()
        
        self.num_segments = num_segments
        self.action_space = spaces.Box(low=0, high=20.0, shape=(self.num_segments,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(865,), dtype=np.float32)
        
        self.state = None
        self.initial_state = initial_state
        self.net_load_data = net_load_data
        self.current_step = 0

        # Generator data
        self.P_g = P_g
        self.C_g = C_g
        self.num_gen = num_gen
        
        # Storage parameters
        self.P = P
        self.E = E
        self.eta = eta
        self.C_s = C_s
        self.num_intervals = num_intervals

        # Load the prediction model for unadjusted bids
        self.model = model
    
    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = np.array(initial_state)
        else:
            self.state = np.array(self.initial_state) if self.initial_state is not None else np.zeros((865,))        
        return self.state
    
    def step(self, action):
        unadjusted_charge_bid, unadjusted_discharge_bid = self.get_unadjusted_bids()

        adjusted_charge_bid = unadjusted_charge_bid - action * self.eta
        adjusted_discharge_bid = unadjusted_discharge_bid + action / self.eta
        
        # adjustment_charge = action[:self.num_segments]
        # adjustment_discharge = action[self.num_segments:]
        # adjusted_charge_bid = unadjusted_charge_bid - adjustment_charge
        # adjusted_discharge_bid = unadjusted_discharge_bid + adjustment_discharge
        
        real_time_price, discharge, charge, soc = self.market_clearing_model(
                    self.net_load_data[self.current_step], 
                    self.P_g, self.C_g, self.P, self.E, self.eta, self.C_s, 
                    self.state[288], self.num_intervals, self.num_gen, self.num_segments,
                    adjusted_discharge_bid, adjusted_charge_bid
                )        
        
        reward = real_time_price * (discharge - charge)
        
        self.state[:288] = np.append(self.state[1:288], discharge - charge)
        self.state[288] = soc
        self.state[289:577] = np.append(self.state[290:577], real_time_price)
        self.state[577:865] = np.append(self.state[578:865], self.net_load_data[self.current_step])
        
        self.current_step += 1
        
        done = self.current_step >= len(self.net_load_data)
        return self.state, reward, done, discharge, charge, real_time_price, unadjusted_charge_bid, unadjusted_discharge_bid, adjusted_charge_bid, adjusted_discharge_bid
    
    def market_clearing_model(self, L, P_g, C_g, P, E, eta, C_s, E0, num_intervals, num_gen, num_seg, discharge_bids, charge_bids):
        p = cp.Variable(num_gen)
        c = cp.Variable(num_seg)
        d = cp.Variable(num_seg)
        e = cp.Variable(num_seg)
        
        constraints = []
        for g in range(num_gen):
            constraints += [p[g] >= 0, p[g] <= P_g[g]]
        
        for s in range(num_seg):
            constraints += [c[s] >= 0, d[s] >= 0]
        
        constraints += [cp.sum(c) <= P]
        constraints += [cp.sum(d) <= P]
        
        for s in range(num_seg):
            constraints += [e[s] == E0 + eta * c[s] - d[s] / eta]
            constraints += [e[s] <= E / num_seg]
            constraints += [e[s] >= 0]
        
        balance_constraint = cp.sum(p) + cp.sum(d) == L + cp.sum(c)
        constraints += [balance_constraint]
        
        cost = (cp.sum(cp.multiply(p, C_g.flatten())) + 
                cp.sum(cp.multiply(d, discharge_bids)) -
                cp.sum(cp.multiply(c, charge_bids))) / num_intervals
        objective = cp.Minimize(cost)
        
        problem = cp.Problem(objective, constraints)
        start_time = time.time()
        try:
            problem.solve(solver=cp.GUROBI, verbose=False, reoptimize=True)
            dual_price = balance_constraint.dual_value * -num_intervals
        except cp.SolverError:
            print("Gurobi not available. Falling back to a different solver.")
            problem.solve(solver=cp.ECOS, verbose=False)
            dual_price = balance_constraint.dual_value * -num_intervals
        end_time = time.time()
        
        total_discharge = sum(d.value)
        total_charge = sum(c.value)
        storage_profit = dual_price * (total_discharge - total_charge) - C_s * total_discharge
        
        return dual_price, total_discharge, total_charge, e.value[-1]
    
    def get_unadjusted_bids(self):
        predictors = self.state[289:577].reshape(1, -1)  # Last 288 real-time prices
        v = self.model.predict(predictors, verbose=0).T
        soc_bids = np.mean(v.reshape(self.num_segments, int(v.shape[0] / self.num_segments)), axis=1, keepdims=True).flatten()
        discharge_bids = soc_bids / self.eta + self.C_s
        charge_bids = soc_bids * self.eta
        return charge_bids, discharge_bids


