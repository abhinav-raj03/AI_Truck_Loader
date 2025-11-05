from ortools.sat.python import cp_model
from .config import ALPHA_VOL, BETA_WT

def select_subset(items, truck):
    model = cp_model.CpModel()
    n = len(items)
    x = [model.NewBoolVar(f"x{i}") for i in range(n)]
    weights = [int(round(it.weight)) for it in items]
    vols = [int(round(it.vol*1000)) for it in items]  # m3 -> liters
    model.Add(sum(weights[i]*x[i] for i in range(n)) <= int(truck.payload_kg))
    model.Add(sum(vols[i]*x[i] for i in range(n)) <= int(truck.L*truck.W*truck.H*1000))
    model.Maximize(sum(ALPHA_VOL*vols[i]*x[i] + BETA_WT*weights[i]*x[i] for i in range(n)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1.8
    solver.parameters.num_search_workers = 4
    solver.parameters.cp_model_presolve = True
    solver.parameters.linearization_level = 0
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    res = solver.Solve(model)
    chosen = []
    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n):
            if solver.Value(x[i])==1: chosen.append(items[i])
    else:
        chosen = items[:]
    return chosen
