from __future__ import annotations

import dataclasses
import numpy as np
import matplotlib.figure as mpl_fig
import matplotlib.pyplot as plt
import casadi as ca
import acados_template as at  # type: ignore

import exp_mpc.stewart_min.const as const

# mpc parameters
n = 200

state0 = np.zeros(12)


def _ca_sx(name: str) -> ca.SX:
    """Create a casadi variable without littered typing errors"""
    return ca.SX.sym(name)  # type: ignore


def gen_stewart_model():
    model = at.AcadosModel()
    model.name = "stewart"

    # TODO(jozbee): make cleaner looking

    x = _ca_sx("x")
    y = _ca_sx("y")
    z = _ca_sx("z")
    phi = _ca_sx("phi")
    theta = _ca_sx("theta")
    psi = _ca_sx("psi")
    x_dot = _ca_sx("x_dot")
    y_dot = _ca_sx("y_dot")
    z_dot = _ca_sx("z_dot")
    phi_dot = _ca_sx("phi_dot")
    theta_dot = _ca_sx("theta_dot")
    psi_dot = _ca_sx("psi_dot")
    model.x = ca.vertcat(  # type: ignore
        x,
        y,
        z,
        phi,
        theta,
        psi,
        x_dot,
        y_dot,
        z_dot,
        phi_dot,
        theta_dot,
        psi_dot,
    )

    x_p = _ca_sx("x_p")
    y_p = _ca_sx("y_p")
    z_p = _ca_sx("z_p")
    phi_p = _ca_sx("phi_p")
    theta_p = _ca_sx("theta_p")
    psi_p = _ca_sx("psi_p")
    x_dot_p = _ca_sx("x_dot_p")
    y_dot_p = _ca_sx("y_dot_p")
    z_dot_p = _ca_sx("z_dot_p")
    phi_dot_p = _ca_sx("phi_dot_p")
    theta_dot_p = _ca_sx("theta_dot_p")
    psi_dot_p = _ca_sx("psi_dot_p")
    model.xdot = ca.vertcat(
        x_p,
        y_p,
        z_p,
        phi_p,
        theta_p,
        psi_p,
        x_dot_p,
        y_dot_p,
        z_dot_p,
        phi_dot_p,
        theta_dot_p,
        psi_dot_p,
    )

    x_u = _ca_sx("x_u")
    y_u = _ca_sx("y_u")
    z_u = _ca_sx("z_u")
    phi_u = _ca_sx("phi_u")
    theta_u = _ca_sx("theta_u")
    psi_u = _ca_sx("psi_u")
    model.u = ca.vertcat(x_u, y_u, z_u, phi_u, theta_u, psi_u)

    model.f_expl_expr = ca.vertcat(  # type: ignore
        x_dot + 0.5 * const.dt * x_u,
        y_dot + 0.5 * const.dt * y_u,
        z_dot + 0.5 * const.dt * z_u,
        phi_dot + 0.5 * const.dt * phi_u,
        theta_dot + 0.5 * const.dt * theta_u,
        psi_dot + 0.5 * const.dt * psi_u,
        x_u,
        y_u,
        z_u,
        phi_u,
        theta_u,
        psi_u,
    )

    return model


def get_R(phi: ca.SX, theta: ca.SX, psi: ca.SX) -> ca.SX:
    R = ca.SX(3, 3)

    # first row
    R[0, 0] = ca.cos(psi) * ca.cos(theta)
    R[0, 1] = ca.sin(phi) * ca.sin(theta) * ca.cos(psi)
    R[0, 1] -= ca.sin(psi) * ca.cos(phi)
    R[0, 2] = ca.sin(phi) * ca.sin(psi)
    R[0, 2] += ca.sin(theta) * ca.cos(phi) * ca.cos(psi)

    # second row
    R[1, 0] = ca.sin(psi) * ca.cos(theta)
    R[1, 1] = ca.sin(phi) * ca.sin(psi) * ca.sin(theta)
    R[1, 1] += ca.cos(phi) * ca.cos(psi)
    R[1, 2] = -ca.sin(phi) * ca.cos(psi)
    R[1, 2] += ca.sin(psi) * ca.sin(theta) * ca.cos(phi)

    # third row
    R[2, 0] = -ca.sin(theta)
    R[2, 1] = ca.sin(phi) * ca.cos(theta)
    R[2, 2] = ca.cos(phi) * ca.cos(theta)

    return R


def get_R_dot2(
    phi: ca.SX,
    theta: ca.SX,
    psi: ca.SX,
    phi_dot: ca.SX,
    theta_dot: ca.SX,
    psi_dot: ca.SX,
    phi_ddot: ca.SX,
    theta_ddot: ca.SX,
    psi_ddot: ca.SX,
) -> ca.SX:
    """Second time derivative of rotation matrix.

    Derived from sympy, i.e., the form is not very readable.
    """
    R_ddot = ca.SX(3, 3)

    # first row
    R_ddot[0, 0] = (
        -psi_ddot * ca.sin(psi) * ca.cos(theta)
        - psi_dot**2 * ca.cos(psi) * ca.cos(theta)
        + 2 * psi_dot * theta_dot * ca.sin(psi) * ca.sin(theta)
        - theta_ddot * ca.sin(theta) * ca.cos(psi)
        - theta_dot**2 * ca.cos(psi) * ca.cos(theta)
    )
    R_ddot[0, 1] = (
        phi_ddot * ca.sin(phi) * ca.sin(psi)
        + phi_ddot * ca.sin(theta) * ca.cos(phi) * ca.cos(psi)
        - phi_dot**2 * ca.sin(phi) * ca.sin(theta) * ca.cos(psi)
        + phi_dot**2 * ca.sin(psi) * ca.cos(phi)
        + 2 * phi_dot * psi_dot * ca.sin(phi) * ca.cos(psi)
        - 2 * phi_dot * psi_dot * ca.sin(psi) * ca.sin(theta) * ca.cos(phi)
        + 2 * phi_dot * theta_dot * ca.cos(phi) * ca.cos(psi) * ca.cos(theta)
        - psi_ddot * ca.sin(phi) * ca.sin(psi) * ca.sin(theta)
        - psi_ddot * ca.cos(phi) * ca.cos(psi)
        - psi_dot**2 * ca.sin(phi) * ca.sin(theta) * ca.cos(psi)
        + psi_dot**2 * ca.sin(psi) * ca.cos(phi)
        - 2 * psi_dot * theta_dot * ca.sin(phi) * ca.sin(psi) * ca.cos(theta)
        + theta_ddot * ca.sin(phi) * ca.cos(psi) * ca.cos(theta)
        - theta_dot**2 * ca.sin(phi) * ca.sin(theta) * ca.cos(psi)
    )
    R_ddot[0, 2] = (
        -phi_ddot * ca.sin(phi) * ca.sin(theta) * ca.cos(psi)
        + phi_ddot * ca.sin(psi) * ca.cos(phi)
        - phi_dot**2 * ca.sin(phi) * ca.sin(psi)
        - phi_dot**2 * ca.sin(theta) * ca.cos(phi) * ca.cos(psi)
        + 2 * phi_dot * psi_dot * ca.sin(phi) * ca.sin(psi) * ca.sin(theta)
        + 2 * phi_dot * psi_dot * ca.cos(phi) * ca.cos(psi)
        - 2 * phi_dot * theta_dot * ca.sin(phi) * ca.cos(psi) * ca.cos(theta)
        + psi_ddot * ca.sin(phi) * ca.cos(psi)
        - psi_ddot * ca.sin(psi) * ca.sin(theta) * ca.cos(phi)
        - psi_dot**2 * ca.sin(phi) * ca.sin(psi)
        - psi_dot**2 * ca.sin(theta) * ca.cos(phi) * ca.cos(psi)
        - 2 * psi_dot * theta_dot * ca.sin(psi) * ca.cos(phi) * ca.cos(theta)
        + theta_ddot * ca.cos(phi) * ca.cos(psi) * ca.cos(theta)
        - theta_dot**2 * ca.sin(theta) * ca.cos(phi) * ca.cos(psi)
    )

    # second row
    R_ddot[1, 0] = (
        psi_ddot * ca.cos(psi) * ca.cos(theta)
        - psi_dot**2 * ca.sin(psi) * ca.cos(theta)
        - 2 * psi_dot * theta_dot * ca.sin(theta) * ca.cos(psi)
        - theta_ddot * ca.sin(psi) * ca.sin(theta)
        - theta_dot**2 * ca.sin(psi) * ca.cos(theta)
    )
    R_ddot[1, 1] = (
        -phi_ddot * ca.sin(phi) * ca.cos(psi)
        + phi_ddot * ca.sin(psi) * ca.sin(theta) * ca.cos(phi)
        - phi_dot**2 * ca.sin(phi) * ca.sin(psi) * ca.sin(theta)
        - phi_dot**2 * ca.cos(phi) * ca.cos(psi)
        + 2 * phi_dot * psi_dot * ca.sin(phi) * ca.sin(psi)
        + 2 * phi_dot * psi_dot * ca.sin(theta) * ca.cos(phi) * ca.cos(psi)
        + 2 * phi_dot * theta_dot * ca.sin(psi) * ca.cos(phi) * ca.cos(theta)
        + psi_ddot * ca.sin(phi) * ca.sin(theta) * ca.cos(psi)
        - psi_ddot * ca.sin(psi) * ca.cos(phi)
        - psi_dot**2 * ca.sin(phi) * ca.sin(psi) * ca.sin(theta)
        - psi_dot**2 * ca.cos(phi) * ca.cos(psi)
        + 2 * psi_dot * theta_dot * ca.sin(phi) * ca.cos(psi) * ca.cos(theta)
        + theta_ddot * ca.sin(phi) * ca.sin(psi) * ca.cos(theta)
        - theta_dot**2 * ca.sin(phi) * ca.sin(psi) * ca.sin(theta)
    )
    R_ddot[1, 2] = (
        -phi_ddot * ca.sin(phi) * ca.sin(psi) * ca.sin(theta)
        - phi_ddot * ca.cos(phi) * ca.cos(psi)
        + phi_dot**2 * ca.sin(phi) * ca.cos(psi)
        - phi_dot**2 * ca.sin(psi) * ca.sin(theta) * ca.cos(phi)
        - 2 * phi_dot * psi_dot * ca.sin(phi) * ca.sin(theta) * ca.cos(psi)
        + 2 * phi_dot * psi_dot * ca.sin(psi) * ca.cos(phi)
        - 2 * phi_dot * theta_dot * ca.sin(phi) * ca.sin(psi) * ca.cos(theta)
        + psi_ddot * ca.sin(phi) * ca.sin(psi)
        + psi_ddot * ca.sin(theta) * ca.cos(phi) * ca.cos(psi)
        + psi_dot**2 * ca.sin(phi) * ca.cos(psi)
        - psi_dot**2 * ca.sin(psi) * ca.sin(theta) * ca.cos(phi)
        + 2 * psi_dot * theta_dot * ca.cos(phi) * ca.cos(psi) * ca.cos(theta)
        + theta_ddot * ca.sin(psi) * ca.cos(phi) * ca.cos(theta)
        - theta_dot**2 * ca.sin(psi) * ca.sin(theta) * ca.cos(phi)
    )

    # third row
    R_ddot[2, 0] = -theta_ddot * ca.cos(theta) + theta_dot**2 * ca.sin(theta)
    R_ddot[2, 1] = (
        phi_ddot * ca.cos(phi) * ca.cos(theta)
        - phi_dot**2 * ca.sin(phi) * ca.cos(theta)
        - 2 * phi_dot * theta_dot * ca.sin(theta) * ca.cos(phi)
        - theta_ddot * ca.sin(phi) * ca.sin(theta)
        - theta_dot**2 * ca.sin(phi) * ca.cos(theta)
    )
    R_ddot[2, 2] = (
        -phi_ddot * ca.sin(phi) * ca.cos(theta)
        - phi_dot**2 * ca.cos(phi) * ca.cos(theta)
        + 2 * phi_dot * theta_dot * ca.sin(phi) * ca.sin(theta)
        - theta_ddot * ca.sin(theta) * ca.cos(phi)
        - theta_dot**2 * ca.cos(phi) * ca.cos(theta)
    )

    return R_ddot


def get_PHI(phi: ca.SX, theta: ca.SX, psi: ca.SX) -> ca.SX:
    PHI = ca.SX(3, 3)

    # first row
    PHI[0, 0] = 1
    PHI[0, 1] = 0
    PHI[0, 2] = -ca.sin(theta)

    # second row
    PHI[1, 0] = 0
    PHI[1, 1] = ca.cos(phi)
    PHI[1, 2] = ca.sin(phi) * ca.cos(theta)

    # third row
    PHI[2, 0] = 0
    PHI[2, 1] = -ca.sin(phi)
    PHI[2, 2] = ca.cos(phi) * ca.cos(theta)

    return PHI


def get_acceleration(model: at.AcadosModel) -> ca.SX:
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
    return R.T @ (acc + const.gravity)

    # phi_dot = state[9]
    # theta_dot = state[10]
    # psi_dot = state[11]
    # phi_ddot = control[3]
    # theta_ddot = control[4]
    # psi_ddot = control[5]
    # R_ddot = get_R_dot2(
    #     phi,
    #     theta,
    #     psi,
    #     phi_dot,
    #     theta_dot,
    #     psi_dot,
    #     phi_ddot,
    #     theta_ddot,
    #     psi_ddot,
    # )
    # return R.T @ (R_ddot @ human_displacement + acc + gravity)


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


def get_length_cost(model: at.AcadosModel) -> ca.SX:
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
    for top, bot in zip(const.tops, const.bots):  # use center as reference
        diff = (R @ top + t) - bot
        diffs.append(diff)

    leg_costs = 0
    for diff in diffs:
        squared_length = diff.T @ diff  # compute squared Euclidean norm

        leg_cost = squared_length - const.leg_mid**2

        # leg_cost = 1.0 / (squared_length - leg_min**2)
        # leg_cost += 1.0 / (leg_max**2 - squared_length)
        # # scale to approximately unity for most of the workspace
        # leg_cost /= 10.0

        leg_costs += leg_cost

    assert type(leg_costs) is ca.SX
    return leg_costs


def get_yaw_cost(model: at.AcadosModel) -> ca.SX:
    state = model.x
    psi = state[5]  # yaw
    yaw_cost = 1.0 / (psi**2 - const.max_yaw**2) ** 2
    return yaw_cost / 50.0


def gen_stewart_ocp(model):
    ocp = at.AcadosOcp()
    ocp.model = model

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # reference trajectory
    ocp.cost.yref = np.zeros(14)
    ocp.cost.yref_e = np.zeros(2)

    # cost weights (need to be set at runtime)
    ocp.cost.W = np.diag(np.zeros(14))
    ocp.cost.W_e = np.diag(np.zeros(2))

    # cost expressions
    ocp.model.cost_y_expr = ca.vertcat(
        get_acceleration(ocp.model),
        get_angular_velocity(ocp.model),
        ocp.model.u,
        # other costs, without reference
        get_length_cost(ocp.model),
        get_yaw_cost(ocp.model),
    )
    ocp.model.cost_y_expr_e = ca.vertcat(
        # get_acceleration(ocp.model),  # uses control, so not in final cost
        # get_angular_velocity(ocp.model),  # no acc, so no omega
        # other costs, without reference
        get_length_cost(ocp.model),
        get_yaw_cost(ocp.model),
    )

    # initial state
    ocp.constraints.x0 = state0

    # options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver_iter_max = 1
    ocp.solver_options.qp_solver_cond_N = n  # no condensing, for now

    # horizon
    ocp.solver_options.N_horizon = n
    ocp.solver_options.tf = n * const.dt

    # TODO(jozbee): make non-uniform
    ocp.solver_options.shooting_nodes = np.arange(n + 1, dtype=float) * const.dt

    return ocp


@dataclasses.dataclass
class Pose:
    x: float
    y: float
    z: float
    phi: float
    theta: float
    psi: float

    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def rpy(self) -> np.ndarray:
        return np.array([self.phi, self.theta, self.psi])

    def __array__(self, copy: bool = False) -> np.ndarray:
        assert type(copy) is bool
        return np.array(
            [self.x, self.y, self.z, self.phi, self.theta, self.psi]
        )


@dataclasses.dataclass
class PoseDot:
    x_dot: float
    y_dot: float
    z_dot: float
    phi_dot: float
    theta_dot: float
    psi_dot: float

    def xyz(self) -> np.ndarray:
        return np.array([self.x_dot, self.y_dot, self.z_dot])

    def rpy(self) -> np.ndarray:
        return np.array([self.phi_dot, self.theta_dot, self.psi_dot])

    def __array__(self, copy: bool = False) -> np.ndarray:
        assert type(copy) is bool
        return np.array(
            [
                self.x_dot,
                self.y_dot,
                self.z_dot,
                self.phi_dot,
                self.theta_dot,
                self.psi_dot,
            ]
        )


@dataclasses.dataclass
class PoseDot2:
    x_dot2: float
    y_dot2: float
    z_dot2: float
    phi_dot2: float
    theta_dot2: float
    psi_dot2: float

    def xyz(self) -> np.ndarray:
        return np.array([self.x_dot2, self.y_dot2, self.z_dot2])

    def rpy(self) -> np.ndarray:
        return np.array([self.phi_dot2, self.theta_dot2, self.psi_dot2])

    def __array__(self, copy: bool = False) -> np.ndarray:
        assert type(copy) is bool
        return np.array(
            [
                self.x_dot2,
                self.y_dot2,
                self.z_dot2,
                self.phi_dot2,
                self.theta_dot2,
                self.psi_dot2,
            ]
        )


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

    @classmethod
    def from_solver(cls, solver: at.AcadosOcpSolver) -> "TableSol":
        x = np.zeros((n + 1, 12))
        u = np.zeros((n, 6))

        # numpy implicitly copies here
        for i in range(n):
            x[i] = solver.get(i, "x")
            u[i] = solver.get(i, "u")
        x[n] = solver.get(n, "x")

        stats = TableStats(
            time=float(solver.get_stats("time_tot")),
            status=solver.get_status(),
            cost=solver.get_cost(),
        )

        return cls(x=x, u=u, stats=stats)


@dataclasses.dataclass
class TableMPC:
    """A model predictive controller for the Stewart platform."""

    model: at.AcadosModel
    ocp: at.AcadosOcp
    solver: at.AcadosOcpSolver

    x0: np.ndarray = dataclasses.field(default_factory=lambda: state0)
    leg_ref: np.ndarray = dataclasses.field(
        default_factory=lambda: np.ones(6) * const.leg_mid**2  # squared lengths
    )
    a_ref: np.ndarray = dataclasses.field(default_factory=lambda: const.gravity)
    omega_ref: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3)
    )
    zero_control: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(6)
    )

    state_sol: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((n + 1, 12))
    )
    control_sol: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((n, 6))
    )

    solution_status: int = 0
    solve_time: float = 0.0
    cost: float = 0.0

    @classmethod
    def create_default(cls) -> TableMPC:
        model = gen_stewart_model()
        ocp = gen_stewart_ocp(model)
        solver = at.AcadosOcpSolver(ocp)
        mpc = cls(model, ocp, solver)  # type: ignore
        # mpc.stable_init()  # cf. openpilot
        mpc.set_weights()
        mpc.set_reference()
        return mpc

    def stable_init(self):
        x0 = state0.copy()
        for i in range(n):
            self.solver.set(i, "x", np.zeros(12))
            self.solver.set(i, "u", np.zeros(6))
        self.solver.set(n, "x", np.zeros(12))
        self.solver.constraints_set(0, "lbx", x0)
        self.solver.constraints_set(0, "ubx", x0)
        self.solver.solve()

    def set_weights(
        self,
        w_acc_x=1.0,
        w_acc_y=1.0,
        w_acc_z=1.0,
        w_omega=1.0,
        w_control=1.0,
        w_leg=1.0,
        w_yaw=1.0,
    ):
        W = np.diag(
            np.concat(
                [
                    np.array([w_acc_x, w_acc_y, w_acc_z]),
                    np.ones(3) * w_omega,
                    np.ones(6) * w_control,
                    np.ones(1) * w_leg,
                    np.ones(1) * w_yaw,
                ]
            )
        )
        W_e = np.diag(np.array([w_leg, w_yaw]))
        for i in range(n):
            self.solver.cost_set(i, "W", W)
        self.solver.cost_set(n, "W", W_e)

    def set_reference(self, a_ref=None, omega_ref=None):
        if a_ref is not None:
            self.a_ref = a_ref
        if omega_ref is not None:
            self.omega_ref = omega_ref

        for i in range(n):
            yref = np.concatenate(
                [self.a_ref, self.omega_ref, self.zero_control, np.zeros(2)]
            )
            self.solver.cost_set(i, "yref", yref)

        yref_e = np.zeros(2)
        self.solver.cost_set(n, "yref", yref_e)

    def solve(self, x0):
        assert x0.shape == (12,)
        self.x0 = x0
        self.solver.constraints_set(0, "lbx", self.x0)
        self.solver.constraints_set(0, "ubx", self.x0)

        self.solver.solve()

        self.solution_status = self.solver.get_status()
        self.solve_time = float(self.solver.get_stats("time_tot"))
        self.cost = self.solver.get_cost()

        for i in range(n):
            state = self.solver.get(i, "x")
            self.state_sol[i] = state
            self.control_sol[i] = self.solver.get(i, "u")
        self.state_sol[n] = self.solver.get(n, "x")

        return self.control_sol[0].copy()

    def sim_next_state(self) -> np.ndarray:
        """Simulate the next state, using double integrator control."""
        x0 = self.state_sol[0]
        u = self.control_sol[0]

        x = np.zeros(12)
        x[6:12] = x0[6:12] + const.dt * u
        x[:6] = x0[:6] + const.dt * x[6:12]
        return x

    def get_solution(self) -> TableSol:
        """Get the last solution."""
        return TableSol.from_solver(self.solver)

    def errors(self):
        return self.solver.get_residuals(recompute=True)

    def plot_solution(self) -> mpl_fig.Figure:
        """Plot all state and control variables from the solution."""
        # Create time arrays for x-axis
        t_states = np.arange(n + 1) * const.dt
        t_controls = np.arange(n) * const.dt

        # Create figure with subplots
        fig = plt.figure(figsize=(10, 8))

        # Position plot
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(t_states, self.state_sol[:, 0], label="x")
        ax1.plot(t_states, self.state_sol[:, 1], label="y")
        ax1.plot(t_states, self.state_sol[:, 2], label="z")
        ax1.set_title("Position")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Position [m]")
        ax1.legend()
        ax1.grid(True)

        # Orientation plot
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(t_states, self.state_sol[:, 3], label="phi")
        ax2.plot(t_states, self.state_sol[:, 4], label="theta")
        ax2.plot(t_states, self.state_sol[:, 5], label="psi")
        ax2.set_title("Orientation")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Angle [rad]")
        ax2.legend()
        ax2.grid(True)

        # Linear velocity plot
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(t_states, self.state_sol[:, 6], label="x_dot")
        ax3.plot(t_states, self.state_sol[:, 7], label="y_dot")
        ax3.plot(t_states, self.state_sol[:, 8], label="z_dot")
        ax3.set_title("Linear Velocity")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Velocity [m/s]")
        ax3.legend()
        ax3.grid(True)

        # Angular velocity plot
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(t_states, self.state_sol[:, 9], label="phi_dot")
        ax4.plot(t_states, self.state_sol[:, 10], label="theta_dot")
        ax4.plot(t_states, self.state_sol[:, 11], label="psi_dot")
        ax4.set_title("Angular Velocity")
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Angular Velocity [rad/s]")
        ax4.legend()
        ax4.grid(True)

        # Linear acceleration (control) plot
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(t_controls, self.control_sol[:, 0], label="x_ddot")
        ax5.plot(t_controls, self.control_sol[:, 1], label="y_ddot")
        ax5.plot(t_controls, self.control_sol[:, 2], label="z_ddot")
        ax5.set_title("Linear Acceleration (Control)")
        ax5.set_xlabel("Time [s]")
        ax5.set_ylabel("Acceleration [m/s²]")
        ax5.legend()
        ax5.grid(True)

        # Angular acceleration (control) plot
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.plot(t_controls, self.control_sol[:, 3], label="phi_ddot")
        ax6.plot(t_controls, self.control_sol[:, 4], label="theta_ddot")
        ax6.plot(t_controls, self.control_sol[:, 5], label="psi_ddot")
        ax6.set_title("Angular Acceleration (Control)")
        ax6.set_xlabel("Time [s]")
        ax6.set_ylabel("Angular Acceleration [rad/s²]")
        ax6.legend()
        ax6.grid(True)

        fig.tight_layout()
        return fig


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
    mpc.plot_solution().show()
