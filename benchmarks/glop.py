"""
Glop Class for solving lp problem. Used for generating random initial states for training and testing when initial constraints exist in the benchmark
Abdelrahman Hekal
"""
from ortools.linear_solver import pywraplp

import random
import numpy as np
from scipy.stats import truncnorm


class ObjFn:
    UNIFORM, WEIGHTED, TRUNC_NORMAL_DIST = range(3)

    def __init__(self, variables_indices, var_range, modes_indices=None, dist_type=UNIFORM, beginning_of_mode=True):
        self.variables_indices = variables_indices
        self.range = var_range
        self.modes_indices = modes_indices
        self.dist_type = dist_type
        self.beginning_of_mode = beginning_of_mode


class Glop:

    def __init__(self, names, constrs, dist_type=ObjFn.UNIFORM):
        self.solver = pywraplp.Solver.CreateSolver('GLOP')
        self.names = names
        self.constrs = constrs
        self.add_vars()
        self.add_constr()
        self.status = None
        self.obj_fn = ""
        self.dist_type = dist_type

    def add_vars(self):
        for name in self.names:
            globals()[name] = self.solver.NumVar(-self.solver.infinity(), self.solver.infinity(), name)

    def add_constr(self):
        for constr in self.constrs:
            self.solver.Add(eval(constr))

    def solve(self):
        self.status = self.solver.Solve()

    def minimize(self):
        if self.obj_fn:
            self.solver.Minimize(eval(self.obj_fn))
            self.status = self.solver.Solve()
        else:
            self.solve()
            print("no obj function specified")

    def maximize(self):
        if self.obj_fn:
            self.solver.Maximize(eval(self.obj_fn))
            self.status = self.solver.Solve()
        else:
            self.solve()
            print("no obj function specified")

    def add_tend_value_obj_fn(self, name, var_range, seed):
        optim_var_1 = "t1_" + str(name)
        optim_var_2 = "t2_" + str(name)
        globals()[optim_var_1] = self.solver.NumVar(0, self.solver.infinity(), optim_var_1)
        globals()[optim_var_2] = self.solver.NumVar(0, self.solver.infinity(), optim_var_2)
        tend_val = self.get_tend_val(var_range, seed)  # value to tend variable to
        if var_range[1] - var_range[0] != 0:  # if valid range is exact value, don't add it to objective function
            self.solver.Add(
                eval(
                    f"{optim_var_1}-{optim_var_2} == ({name} -{tend_val})/({var_range[1]}-{var_range[0]})"))
        if self.obj_fn:
            self.obj_fn += "+"
        self.obj_fn += optim_var_1 + "+" + optim_var_2

    def get_tend_val(self, var_range, seed, mean=0, sd=1):
        random.seed(seed)
        np.random.seed(seed)
        if self.dist_type == ObjFn.TRUNC_NORMAL_DIST:
            tend_val = round(get_truncated_normal(mean=mean, sd=sd, low=var_range[0], upp=var_range[1]).rvs(1)[0], 3)
        else:  # uniform distribution as default
            # should we  round the data?
            # tend_val = round(random.uniform(var_range[0], var_range[1]),5)
            tend_val = random.uniform(var_range[0], var_range[1])
        return tend_val

    def print_sol_time(self):
        if self.status == pywraplp.Solver.OPTIMAL:
            print('Problem solved in %f milliseconds' % self.solver.wall_time())
        else:
            print('no solution was found')

    def get_all_sols(self):
        sol_dict = {}
        for name in self.names:
            val = globals()[name].solution_value()
            sol_dict[name] = val

        return sol_dict
