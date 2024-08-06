import cvxpy as cp
import time

# Function to solve the optimization problem for a single day
def solve_daily_optimization(L, P_g, C_g, P, E, eta, C_s, E0, num_gen, num_steps, num_intervals):
    # Variables
    p = cp.Variable((num_steps, num_gen))  # Power generation by each generator
    c = cp.Variable(num_steps)             # Power charge by storage
    d = cp.Variable(num_steps)             # Power discharge by storage
    e = cp.Variable(num_steps)             # State of Charge (SoC) of storage

    # Constraints
    constraints = []

    # Generator capacity constraints
    for g in range(num_gen):
        constraints += [p[:, g] >= 0,
                        p[:, g] <= P_g[g]]

    # Storage constraints
    constraints += [c >= 0,
                    c <= P,
                    d >= 0,
                    d <= P]

    # SoC constraints
    constraints += [e[0] == E0 + eta * c[0] - d[0] / eta]
    for t in range(1, num_steps):
        constraints += [e[t] == e[t-1] + eta * c[t] - d[t] / eta]
        
    constraints += [e >= 0, e <= E]

    # End of day SoC constraint
    constraints += [e[-1] == E / 2]  # End SoC is 50%

    # Balance constraints and capture dual variables
    balance_constraints = []
    for t in range(num_steps):
        balance_constraint = cp.sum(p[t, :]) + d[t] == L[t] + c[t]
        constraints += [balance_constraint]
        balance_constraints.append(balance_constraint)

    # Objective: Minimize system cost
    cost = (cp.sum(cp.multiply(p, C_g.reshape(1, -1))) + cp.sum(C_s * d))/ num_intervals
    objective = cp.Minimize(cost)

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem using Gurobi if available, otherwise use a different solver
    start_time = time.time()
    try:
        problem.solve(solver=cp.GUROBI, verbose=False, reoptimize=True)
        dual_prices = [balance_constraint.dual_value * -num_intervals for balance_constraint in balance_constraints]  # Extract dual values
    except cp.SolverError:
        print("Gurobi not available. Falling back to a different solver.")
        problem.solve(solver=cp.ECOS, verbose=False)  # You can use ECOS, SCS, or another solve
        dual_prices = [balance_constraint.dual_value  * -num_intervals for balance_constraint in balance_constraints]  # Extract dual values
    end_time = time.time()

    return problem.value, p.value, c.value, d.value, e.value, dual_prices, end_time - start_time