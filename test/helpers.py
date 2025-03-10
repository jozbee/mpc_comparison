"""Provide helpers for testing."""
import numpy as np
import dataclasses
import matplotlib.pyplot as plt
import matplotlib.widgets as mpl_wid
import matplotlib.backend_bases as mpl_base
import matplotlib.figure as mpl_fig
import matplotlib.gridspec as mpl_gs
import typing as ty


@dataclasses.dataclass
class PassFail:
    """Prompt user to pass or fail the data test."""
    fig: mpl_fig.Figure
    success = False

    def success_callback(self, event: mpl_base.Event):
        assert isinstance(event, mpl_base.MouseEvent)
        self.success = True
        plt.close(self.fig)

    def fail_callback(self, event: mpl_base.Event):
        assert isinstance(event, mpl_base.MouseEvent)
        self.success = False
        plt.close(self.fig)


def visualize_pass_fail(
    expected: np.ndarray,
    actual: np.ndarray,
    title: ty.Optional[str] = None,
) -> bool:
    """Prompt user to pass or fail the data test."""
    fig = plt.figure(layout="constrained")
    gs = mpl_gs.GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        height_ratios=[1, 0.1],
        width_ratios=[1, 1],
        wspace=0.01,
        hspace=0.01,
    )

    # plot
    plot_ax = fig.add_subplot(gs[0, :])
    plot_ax.plot(expected, label="expected")
    plot_ax.plot(actual, label="actual")
    if isinstance(title, str):
        plot_ax.set_title(title)
    plot_ax.legend()

    # add pass/fail buttons
    pass_fail = PassFail(fig=fig)

    pass_ax = fig.add_subplot(gs[1, 1])  # type: ignore
    pass_button = mpl_wid.Button(pass_ax, "pass")
    pass_button.on_clicked(pass_fail.success_callback)

    fail_ax = fig.add_subplot(gs[1, 0])  # type: ignore
    fail_button = mpl_wid.Button(fail_ax, "fail")
    fail_button.on_clicked(pass_fail.fail_callback)

    # run the test
    plt.show()
    return pass_fail.success
