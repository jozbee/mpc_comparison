import numpy as np
from casadi import SX, vertcat
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

n = 32
acados_solver_type = 'SQP_RTI'
x_min = 1.2
x_mid = 1.5
x_max = 1.8
v_max = 0.1
a_max = 0.2
dt = 1. / 250.


def gen_leg_model():
    model = AcadosModel()

    x = SX.sym('x')
    v = SX.sym('v')
    model.x = vertcat(x, v)

    x_dot = SX.sym('x_dot')
    v_dot = SX.sym('v_dot')
    model.xdot = vertcat(x_dot, v_dot)

    a = Sx.sym('a')
    model.u = a

    f_expl = vertcat(v, a)

    model.f_impl_expr = model.xdot - f_expl
    model.f_expl_expr = f_expl

    return model

def gen_leg_ocp(model, acc_weight=1e3):
    ocp = AcadosOcp()
    ocp.model = model

    ocp.dims.N = n

    ocp.cost.cost_type = 'LINEAR_LS'

    ocp.cost.yref = np.zeros(2 * n)  # [x, a]
    ocp.cost.yref_e = np.zeros(1)  # [x]
    ocp.cost_y_expr = vertcat(ocp.model.x[0], ocp.model.u)
    ocp.cost_y_expr_e = ocp.model.x[0]

    ocp.cost.W = np.diag([1.0, acc_weight])
    ocp.cost.W_e = np.diag([1.0])

    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.idxbx = np.array([0, 1])
    ocp.constraints.lbx = np.array([x_min, -v_max])
    ocp.constraints.ubx = np.array([x_max, v_max])

    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbu = np.array([-a_max])
    ocp.constraints.ubu = np.array([a_max])

    ocp.constraints.x0 = np.array([x_mid, 0.])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver_iter_max = 1
    ocp.solver_options.qp_solver_cond_N = 1

    # TODO(jozbee): non-uniform
    # ocp.solver_options.tf = np.arange(n, dtype=float) * dt
    ocp.solver_options.shooting_nodes = np.arange(n + 1, dtype=float) * dt

    return ocp


class LegMPC:
    """A model predictive controller for the leg."""
    def __init__(self, x0=None):
        self.solver = AcadosOcpSolver(gen_leg_ocp(gen_leg_model()))
        self.reset(x0)
    
    def reset(self, x0=None):
        if x0 is None:
            x0 = np.array([x_mid, 0.])
        self.x0 = x0

        self.x_sol = np.zeros(n + 1)
        self.v_sol = np.zeros(n + 1)
        self.a_sol = np.zeros(n)

        self.x0_ref = x_mid
        self.a_ref = 0.
        for i in range(n):
            self.solver.cost_set(i, "yref", np.array([self.x0_ref, self.a_ref]))
        self.solver.cost_set(n, "yref", np.array([self.x0_ref]))

        # TODO(jozbee): needed for stable init?
        for i in range(n):
            self.solver.set(i, "x", np.ones(2), x_mid)
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)
        self.solver.solve()
        self.solution_status = 0
        self.solve_time = 0.
        self.cost = 0.

    def solve(self, x0, v0, x0_ref, a_ref):
        self.x0 = np.array([x0, v0])
        self.x0_ref = x0_ref
        self.a_ref = a_ref

        self.solver.constraints_set(0, "lbx", self.x0)
        self.solver.constraints_set(0, "ubx", self.x0)
        for i in range(n):
            self.solver.cost_set(i, "yref", np.array([self.x0_ref, self.a_ref]))
        self.solver.cost_set(n, "yref", np.array([self.x0_ref]))

        self.solver.solve()

        self.solution_status = self.solver.get_status()
        self.solve_time = self.solver.get_stats("time_tot")
        self.cost = self.solver.get_cost()

        for i in range(n):
            state = self.solver.get(i, "x")
            self.x_sol[i] = state[0]
            self.v_sol[i] = state[1]
            self.a_sol[i] = self.solver.get(i, "u")[0]

        return self.a_sol[0]


if __name__ == "__main__":
    solver = LegMPC()

    x0 = x_mid
    v0 = 0.01
    a_ref = 0.1
    x0_ref = x_mid
    a = solver.solve(x0, v0, x0_ref, a_ref)
    print(a)
    print(solver.solve_time)
