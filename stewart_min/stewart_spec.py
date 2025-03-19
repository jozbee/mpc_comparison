from __future__ import annotations

import dataclasses
import numpy as np
import casadi as ca
import acados_template as at

n = 40
leg_min = 1.2
leg_max = 1.8
leg_mid = 1.5
dt = 1. / 250.

center = np.array([-2.53480164e-05, -1.86958364e-04, -9.81776321e-02])
center = center.reshape(1, 3)  # for broadcasting with tops and bots
bots = np.array([
    [ 0.090902, -0.952057, -1.4     ],
    [ 0.870168,  0.397276, -1.400285],
    [ 0.779349,  0.554337, -1.4     ],
    [-0.779347,  0.553952, -1.400037],
    [-0.86997 ,  0.397276, -1.4     ],
    [-0.090113, -0.952578, -1.400363],
])
tops = np.array([
    [ 0.32637129, -0.32151598, -0.09877087],
    [ 0.44215802, -0.118274  , -0.09750311],
    [ 0.1149581 ,  0.44309603, -0.09906373],
    [-0.11874097,  0.4420145 , -0.09723438],
    [-0.44143789, -0.12203837, -0.09888594],
    [-0.32346064, -0.32440392, -0.09760777],
])


center_bots = bots - center
center_tops = tops - center
state0 = np.array([
    center[0, 0], center[0, 1], center[0, 2],
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
])


def gen_stewart_model():
    model = at.AcadosModel()
    model.name = "stewart"

    # TODO(jozbee): make cleaner looking

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    z = ca.SX.sym("z")
    phi = ca.SX.sym("phi")
    theta = ca.SX.sym("theta")
    psi = ca.SX.sym("psi")
    x_dot = ca.SX.sym("x_dot")
    y_dot = ca.SX.sym("y_dot")
    z_dot = ca.SX.sym("z_dot")
    phi_dot = ca.SX.sym("phi_dot")
    theta_dot = ca.SX.sym("theta_dot")
    psi_dot = ca.SX.sym("psi_dot")
    model.x = ca.vertcat(x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot)

    x_p = ca.SX.sym("x_p")
    y_p = ca.SX.sym("y_p")
    z_p = ca.SX.sym("z_p")
    phi_p = ca.SX.sym("phi_p")
    theta_p = ca.SX.sym("theta_p")
    psi_p = ca.SX.sym("psi_p")
    x_dot_p = ca.SX.sym("x_dot_p")
    y_dot_p = ca.SX.sym("y_dot_p")
    z_dot_p = ca.SX.sym("z_dot_p")
    phi_dot_p = ca.SX.sym("phi_dot_p")
    theta_dot_p = ca.SX.sym("theta_dot_p")
    psi_dot_p = ca.SX.sym("psi_dot_p")
    model.xdot = ca.vertcat(x_p, y_p, z_p, phi_p, theta_p, psi_p, x_dot_p, y_dot_p, z_dot_p, phi_dot_p, theta_dot_p, psi_dot_p)

    x_u = ca.SX.sym("x_u")
    y_u = ca.SX.sym("y_u")
    z_u = ca.SX.sym("z_u")
    phi_u = ca.SX.sym("phi_u")
    theta_u = ca.SX.sym("theta_u")
    psi_u = ca.SX.sym("psi_u")
    model.u = ca.vertcat(x_u, y_u, z_u, phi_u, theta_u, psi_u)

    
    model.f_expl_expr = ca.vertcat(
        x_dot + 0.5 * dt * x_u,
        y_dot + 0.5 * dt * y_u,
        z_dot + 0.5 * dt * z_u,
        phi_dot + 0.5 * dt * phi_u,
        theta_dot + 0.5 * dt * theta_u,
        psi_dot + 0.5 * dt * psi_u,
        x_u,
        y_u,
        z_u,
        phi_u,
        theta_u,
        psi_u
    )

    return model


def get_R(phi: ca.SX, theta: ca.SX, psi: ca.SX) -> ca.SX:
    R = ca.SX(3, 3)

    # first row
    R[0, 0] = ca.cos(psi) * ca.cos(theta)
    R[0, 1] = ca.sin(phi) * ca.sin(theta) * ca.cos(psi) - ca.sin(psi) * ca.cos(phi)
    R[0, 2] = ca.sin(phi) * ca.sin(psi) + ca.sin(theta) * ca.cos(phi) * ca.cos(psi)

    # second row
    R[1, 0] = ca.sin(psi) * ca.cos(theta)
    R[1, 1] = ca.sin(phi) * ca.sin(psi) * ca.sin(theta) + ca.cos(phi) * ca.cos(psi)
    R[1, 2] = -ca.sin(phi) * ca.cos(psi) + ca.sin(psi) * ca.sin(theta) * ca.cos(phi)

    # third row
    R[2, 0] = -ca.sin(theta)
    R[2, 1] = ca.sin(phi) * ca.cos(theta)
    R[2, 2] = ca.cos(phi) * ca.cos(theta)

    return R


def get_PHI(phi: ca.SX, theta: ca.SX, psi: ca.SX) -> ca.SX:
    PHI = ca.SX(3, 3)

    # first row
    PHI[0, 0] = 1
    PHI[0, 1] = 0
    PHI[0, 2] = -ca.sin(theta)

    # second row
    PHI[1, 0] = 0
    PHI[1, 1] = ca.cos(phi)
    PHI[1, 2] = ca.sin(phi)*ca.cos(theta)

    # third row
    PHI[2, 0] = 0
    PHI[2, 1] = -ca.sin(phi)
    PHI[2, 2] = ca.cos(phi)*ca.cos(theta)

    return PHI


def get_acceleration(model: at.AcadosModel) -> ca.SX:
    g = ca.vertcat(0.0, 0.0, 9.81)

    state = model.x
    control = model.u

    phi = state[3]
    theta = state[4]
    psi = state[5]

    x_u = control[0]
    y_u = control[1]
    z_u = control[2]
    acc = ca.vertcat(x_u, y_u, z_u)

    R = get_R(phi, theta, psi)

    return R.T @ (acc - g)


def get_angular_velocity(model: at.AcadosModel) -> ca.SX:
    state = model.x

    phi = state[3]
    theta = state[4]
    psi = state[5]

    phi_dot = state[9]
    theta_dot = state[10]
    psi_dot = state[11]

    PHI = get_PHI(phi, theta, psi)
    euler_dot = ca.vertcat(phi_dot, theta_dot, psi_dot)

    return PHI @ euler_dot


def get_squared_lengths(model: at.AcadosModel) -> ca.SX:
    state = model.x

    x = state[0]
    y = state[1]
    z = state[2]
    t = ca.vertcat(x, y, z)

    phi = state[3]
    theta = state[4]
    psi = state[5]
    R = get_R(phi, theta, psi)

    diffs = []
    for top, bot in zip(center_tops, center_bots):  # use center as reference
        diff = (R @ top + t) - bot
        diffs.append(diff)

    squared_lengths = []
    for diff in diffs:
        squared_length = diff.T @ diff  # compute squared Euclidean norm
        squared_lengths.append(squared_length)
        
    return ca.vertcat(*squared_lengths)


def gen_stewart_ocp(model):
    ocp = at.AcadosOcp()
    ocp.model = model

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # reference trajectory
    ocp.cost.yref = np.zeros(18)
    ocp.cost.yref_e = np.zeros(6)

    # cost weights (need to be set at runtime)
    ocp.cost.W = np.diag(np.zeros(18))
    ocp.cost.W_e = np.diag(np.zeros(6))

    # cost expressions
    ocp.model.cost_y_expr = ca.vertcat(
        get_acceleration(ocp.model),
        get_angular_velocity(ocp.model),
        get_squared_lengths(ocp.model),
        ocp.model.u,
    )
    ocp.model.cost_y_expr_e = ca.vertcat(
        # get_acceleration(ocp.model),  # uses control, so not in final cost
        # get_angular_velocity(ocp.model),  # no acc, so no omega
        get_squared_lengths(ocp.model),
    )

    # constraints
    ocp.constraints.constr_type = "BGH"
    ocp.model.con_h_expr = get_squared_lengths(ocp.model)
    ocp.constraints.lh = np.ones(6) * leg_min**2
    ocp.constraints.uh = np.ones(6) * leg_max**2

    ocp.model.con_h_expr_e = get_squared_lengths(ocp.model)
    ocp.constraints.lh_e = np.ones(6) * leg_min**2
    ocp.constraints.uh_e = np.ones(6) * leg_max**2

    # initial state
    ocp.constraints.x0 = state0

    # options
    ocp.solver_options.sq_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver_iter_max = 4
    ocp.qp_solver_cond_N = n  # no condensing, for now

    # horizon
    ocp.solver_options.N_horizon = n
    ocp.solver_options.tf = n * dt

    # TODO(jozbee): make non-uniform
    ocp.solver_options.shooting_nodes = np.arange(n + 1, dtype=float) * dt

    return ocp


@dataclasses.dataclass
class Pose:
    x: float
    y: float
    z: float
    phi: float
    theta: float
    psi: float


@dataclasses.dataclass
class PoseDot:
    x_dot: float
    y_dot: float
    z_dot: float
    phi_dot: float
    theta_dot: float
    psi_dot: float


@dataclasses.dataclass
class PoseDot2:
    x_dot2: float
    y_dot2: float
    z_dot2: float
    phi_dot2: float
    theta_dot2: float
    psi_dot2: float


@dataclasses.dataclass
class TableStats:
    """Statistics of the solution to the Stewart platform OCP."""
    time: float
    status: int
    cost: float


@dataclasses.dataclass
class TableSol:
    """A solution to the Stewart platform OCP."""
    x: np.ndarray
    u: np.ndarray
    stats: TableStats

    def pose_at(self, i: int) -> Pose:
        return Pose(*self.x[i, :6])

    def pose_dot_at(self, i: int) -> PoseDot:
        return PoseDot(*self.x[i, 6:12])
    
    def pose_dot2_at(self, i: int) -> PoseDot2:
        return PoseDot2(*self.u[i, :6])


@dataclasses.dataclass
class TableMPC:
    """A model predictive controller for the Stewart platform."""
    model: at.AcadosModel
    ocp: at.AcadosOcp
    solver: at.AcadosOcpSolver

    x0: np.ndarray = dataclasses.field(default_factory=lambda: state0)
    leg_ref: np.ndarray = dataclasses.field(default_factory=lambda: np.ones(6) * leg_mid**2)
    a_ref: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    omega_ref: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    zero_control: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(6))

    state_sol: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((n + 1, 12)))
    control_sol: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((n, 6)))

    solution_status: int = 0
    solve_time: float = 0.0
    cost: float = 0.0
    
    @classmethod
    def create_default(cls) -> TableMPC:
        model = gen_stewart_model()
        ocp = gen_stewart_ocp(model)
        solver = at.AcadosOcpSolver(ocp)
        mpc = cls(model, ocp, solver)
        mpc.set_weights()
        mpc.set_reference()
        return mpc

    def set_weights(self, w_a=1.0, w_omega=1.0, w_leg=1.0, w_control=1.0):
        W = np.diag(np.concat([
            np.ones(3) * w_a,
            np.ones(3) * w_omega,
            np.ones(6) * w_leg,
            np.ones(6) * w_control
        ]))
        W_e = np.diag(np.ones(6) * w_leg)
        for i in range(n):
            self.solver.cost_set(i, "W", W)
        self.solver.cost_set(n, "W", W_e)

    def set_reference(self, a_ref=None, omega_ref=None):
        if a_ref is not None:
            self.a_ref = a_ref
        if omega_ref is not None:
            self.omega_ref = omega_ref

        for i in range(n):
            yref = np.concatenate([
                self.a_ref, self.omega_ref, self.leg_ref, self.zero_control])
            self.solver.cost_set(i, "yref", yref)

        yref_e = np.concatenate([
            self.leg_ref])
        self.solver.cost_set(n, "yref", yref_e)

    def solve(self, x0):
        assert x0.shape == (12,)
        self.x0 = x0

        yref_e = np.concatenate([
            self.leg_ref])
        self.solver.cost_set(n, "yref", yref_e)

        self.solver.constraints_set(0, "lbx", self.x0)
        self.solver.constraints_set(0, "ubx", self.x0)

        self.solver.solve()

        self.solution_status = self.solver.get_status()
        self.solve_time = self.solver.get_stats("time_tot")
        self.cost = self.solver.get_cost()

        for i in range(n):
            state = self.solver.get(i, "x")
            self.state_sol[i] = state
            self.control_sol[i] = self.solver.get(i, "u")

        return self.control_sol[0]

    def errors(self):
        return self.solver.get_residuals(recompute=True)


if __name__ == "__main__":
    mpc = TableMPC.create_default()

    x0 = state0
    u = mpc.solve(x0)
    print(f"status={mpc.solution_status}")
    print(f"error={mpc.errors()}")
    print(f"time={mpc.solve_time}")
    print(f"cost={mpc.cost}")
    print(f"control={u}")
    print(f"state_sol={mpc.state_sol}")
    print(f"control_sol={mpc.control_sol}")
