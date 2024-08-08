import cvxpy as cp
import time

# Function to solve the optimization problem for a single period
def solve_period_optimization(L, P_g, C_g, P, E, eta, C_s, E0, num_intervals, num_gen, num_seg, discharge_bids, charge_bids):
    # Variables
    p = cp.Variable(num_gen)  # Power generation by each generator
    c = cp.Variable(num_seg)             # Power charge by storage
    d = cp.Variable(num_seg)             # Power discharge by storage
    e = cp.Variable(num_seg)             # State of Charge (SoC) of storage

    # Constraints
    constraints = []

    # Generator capacity constraints
    for g in range(num_gen):
        constraints += [p[g] >= 0,
                        p[g] <= P_g[g]]

    # Storage constraints
    for s in range(num_seg):
        constraints += [c[s] >= 0,
                        d[s] >= 0]
    
    constraints += [cp.sum(c) <= P]
    constraints += [cp.sum(d) <= P]

    # SoC constraints
    for s in range(num_seg):
        constraints += [e[s] == E0[s] + eta * c[s] - d[s] / eta]
        constraints += [e[s] <= E / num_seg]
        constraints += [e[s] >= 0]

        
    # Balance constraints and capture dual variables
    balance_constraint = cp.sum(p) + cp.sum(d) == L + cp.sum(c)
    constraints += [balance_constraint]
    

    # Objective: Minimize system cost
    cost = (cp.sum(cp.multiply(p, C_g.flatten())) + 
            cp.sum(cp.multiply(d, discharge_bids)) -
            cp.sum(cp.multiply(c, charge_bids))) / num_intervals
    objective = cp.Minimize(cost)

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem using Gurobi if available, otherwise use a different solver
    start_time = time.time()
    try:
        problem.solve(solver=cp.GUROBI, verbose=False, reoptimize=True)
        dual_price = balance_constraint.dual_value * -num_intervals  # Extract dual values
    except cp.SolverError:
        print("Gurobi not available. Falling back to a different solver.")
        problem.solve(solver=cp.ECOS, verbose=False)  # You can use ECOS, SCS, or another solve
        dual_price = balance_constraint.dual_value  * -num_intervals  # Extract dual values
    end_time = time.time()

    # Calculate storage profit
    total_discharge = sum(d.value)
    total_charge = sum(c.value)
    storage_profit = dual_price * (total_discharge - total_charge) - C_s * total_discharge

    return problem.value, p.value, c.value, d.value, e.value, dual_price, storage_profit, end_time - start_time
