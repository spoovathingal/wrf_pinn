"""Live training monitor: incremental loss log, refreshed plot, and time-lapse GIF.

This module lets the training loop stream progress to disk *during* a run rather
than only at the end. It is designed for long (multi-day) headless runs:

- Losses are appended to ``loss_history.csv`` as they are produced, so the file
  is always current and a separate viewer can tail it.
- A ``loss_curves.png`` is re-rendered on each update so an image viewer pointed
  at it shows convergence in near real time.
- Each update also saves a numbered PNG frame and rebuilds an animated
  ``loss_curves.gif`` (via matplotlib's Pillow writer) so the whole training
  history can be replayed as a time-lapse.

The monitor is component-agnostic: it accepts a ``total`` loss plus a dictionary
of named component losses, so it adapts to whichever objective modes are active
(pde, boundary, sensor_data, flow_field_data, ...). Component series that are
identically zero across the whole run are omitted from the plot to keep it
readable.

All rendering is exception-isolated by the caller-facing ``update`` method: a
plotting failure logs a warning and never interrupts training.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files without a display.

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


LOSS_HISTORY_FILENAME = "loss_history.csv"
LOSS_PLOT_FILENAME = "loss_curves.png"
LOSS_GIF_FILENAME = "loss_curves.gif"
FRAMES_DIRNAME = "frames"

_TOTAL_COLOR = "#1b1b1b"
# A colorblind-friendly palette cycled across whichever components are present.
_COMPONENT_COLORS = ("#2166ac", "#b2182b", "#228833", "#cc6600", "#6a3d9a")


@dataclass
class LiveTrainingMonitor:
    """Stream loss history to a CSV, a refreshed PNG, and a time-lapse GIF.

    Parameters
    ----------
    output_dir:
        Directory to write ``loss_history.csv``, ``loss_curves.png``,
        ``loss_curves.gif``, and the ``frames/`` subdirectory into.
    component_names:
        Ordered names of the component losses reported each update, used for the
        CSV header and plot legend (for example ``("pde", "flow_field_data")``).
    total_epochs:
        Total planned epochs, used only to fix the plot x-axis so the curve
        grows into a stable frame instead of rescaling every update.
    gif_fps:
        Playback speed of the assembled GIF, in frames per second.
    """

    output_dir: Path
    component_names: tuple[str, ...]
    total_epochs: int | None = None
    gif_fps: float = 8.0

    _epochs: list[int] = field(default_factory=list, init=False)
    _total: list[float] = field(default_factory=list, init=False)
    _components: dict[str, list[float]] = field(default_factory=dict, init=False)
    _frame_paths: list[Path] = field(default_factory=list, init=False)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.component_names = tuple(self.component_names)
        self._components = {name: [] for name in self.component_names}

    # -- public API -------------------------------------------------------- #

    def update(self, *, epoch: int, total: float, components: dict[str, float]) -> None:
        """Record one epoch and refresh the CSV, PNG, frame, and GIF.

        Any rendering error is caught and reported as a warning so training is
        never interrupted by the monitor. The loss CSV is written first, so the
        numeric record survives even if plotting fails.
        """

        self._epochs.append(int(epoch))
        self._total.append(float(total))
        for name in self.component_names:
            self._components[name].append(float(components.get(name, float("nan"))))

        try:
            self._ensure_started()
            self._append_csv_row(epoch, total, components)
            frame_path = self._render_frame()
            self._frame_paths.append(frame_path)
            self._render_current_plot()
            self._render_gif()
        except Exception as error:  # noqa: BLE001 - monitor must never crash training
            print(f"[live-monitor] warning: update failed and was skipped: {error!r}")

    def finalize(self) -> None:
        """Do a final render pass so the outputs reflect the last epoch."""

        if not self._epochs:
            return
        try:
            self._render_current_plot()
            self._render_gif()
        except Exception as error:  # noqa: BLE001
            print(f"[live-monitor] warning: finalize failed: {error!r}")

    # -- internals --------------------------------------------------------- #

    def _ensure_started(self) -> None:
        if self._started:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir().mkdir(parents=True, exist_ok=True)
        with self._csv_path().open("w", newline="") as file:
            csv.writer(file).writerow(("epoch", "total", *self.component_names))
        self._started = True

    def _append_csv_row(
        self, epoch: int, total: float, components: dict[str, float]
    ) -> None:
        row = (epoch, total, *(components.get(name, "") for name in self.component_names))
        with self._csv_path().open("a", newline="") as file:
            csv.writer(file).writerow(row)

    def _plotted_components(self) -> list[str]:
        """Return component names that are not identically zero over the run.

        A component that stays exactly zero the whole run (an inactive objective)
        adds nothing to a log-scale plot, so it is dropped from the legend.
        """

        plotted = []
        for name in self.component_names:
            series = self._components[name]
            if any(value not in (0.0,) and value == value for value in series):
                plotted.append(name)
        return plotted

    def _build_figure(self):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.semilogy(self._epochs, self._total, label="Total", lw=2.4,
                    color=_TOTAL_COLOR, zorder=6)
        for index, name in enumerate(self._plotted_components()):
            color = _COMPONENT_COLORS[index % len(_COMPONENT_COLORS)]
            ax.semilogy(self._epochs, self._components[name], label=name, lw=1.8,
                        color=color, zorder=4)
        ax.scatter([self._epochs[-1]], [self._total[-1]], s=28,
                   color=_TOTAL_COLOR, zorder=7)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training loss history (epoch {self._epochs[-1]})")
        ax.set_xlim(self._x_lower(), self._x_upper())
        ax.grid(True, which="major", ls="-", lw=0.7, color="#c9c9c9", alpha=0.9)
        ax.grid(True, which="minor", ls="-", lw=0.5, color="#e6e6e6", alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", title="Loss component", frameon=False)
        fig.tight_layout()
        return fig

    def _render_current_plot(self) -> None:
        fig = self._build_figure()
        fig.savefig(self._plot_path(), dpi=150)
        plt.close(fig)

    def _render_frame(self) -> Path:
        fig = self._build_figure()
        frame_path = self._frames_dir() / f"frame_{len(self._epochs):06d}.png"
        fig.savefig(frame_path, dpi=110)
        plt.close(fig)
        return frame_path

    def _render_gif(self) -> None:
        # Rebuild the GIF from the saved frames each update so it always reflects
        # the full run so far and survives an early kill. Uses matplotlib's
        # Pillow-backed writer; no direct Pillow calls needed.
        if not self._frame_paths:
            return

        import matplotlib.image as mpimg

        fig = plt.figure()
        fig.set_size_inches(7.5, 5)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        writer = PillowWriter(fps=self.gif_fps)
        with writer.saving(fig, str(self._gif_path()), dpi=110):
            for frame_path in self._frame_paths:
                ax.clear()
                ax.axis("off")
                ax.imshow(mpimg.imread(frame_path))
                writer.grab_frame()
        plt.close(fig)

    def _x_lower(self) -> int:
        return self._epochs[0] if self._epochs else 0

    def _x_upper(self) -> int:
        if self.total_epochs:
            return max(self.total_epochs, self._epochs[-1])
        return max(self._epochs[-1], self._epochs[0] + 1)

    def _csv_path(self) -> Path:
        return self.output_dir / LOSS_HISTORY_FILENAME

    def _plot_path(self) -> Path:
        return self.output_dir / LOSS_PLOT_FILENAME

    def _gif_path(self) -> Path:
        return self.output_dir / LOSS_GIF_FILENAME

    def _frames_dir(self) -> Path:
        return self.output_dir / FRAMES_DIRNAME
