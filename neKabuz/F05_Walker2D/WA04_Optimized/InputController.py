import numpy as np

X_MIN, X_MAX = 0.0, 10.0
Y_MIN, Y_MAX = -1.0, 3.2

X_DEFAULT = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype = float)
Y_DEFAULT = np.array([0.0, 0.8, 0.0, -0.8, 0.0, 0.6], dtype = float)


def biased_speeds(
    n_speeds: int,
    v_min: float = -1.0,
    v_max: float = 3.0,
    p_min: float = 0.4,
    p_max: float = 0.4,
    p_zero: float = 0.15,
    p_random: float = 0.05,
):
    assert abs(p_min + p_max + p_zero + p_random - 1) < 1e-6, "Las probabilidades deben sumar 1"

    categories = np.random.choice(
        ["min", "max", "zero", "rand"],
        size = n_speeds,
        p = [p_min, p_max, p_zero, p_random]
    )

    speeds = np.zeros(n_speeds, dtype = np.float32)
    for i, c in enumerate(categories):
        if c == "min":
            speeds[i] = v_min
        elif c == "max":
            speeds[i] = v_max
        elif c == "zero":
            speeds[i] = 0.0
        else:
            speeds[i] = np.random.uniform(v_min, v_max)
    return speeds



# Honea muitzet, hola ezbaiot deitzen eztu importatzen, ta eztet RunPod barrun instalatu beharko
def run_signal_builder():
    import pandas as pd
    import sys

    from PyQt5 import QtWidgets
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

    class CurveCanvas(FigureCanvasQTAgg):
        def __init__(self, parent = None):
            fig = Figure(figsize = (7, 4), tight_layout = True)
            super().__init__(fig)
            self.setParent(parent)

            self.ax = fig.add_subplot(111)
            self.ax.set_xlim(X_MIN, X_MAX)
            self.ax.set_ylim(Y_MIN, Y_MAX)
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Speed")
            self.ax.grid(True, alpha = 0.3)

            self.x = X_DEFAULT
            self.y = Y_DEFAULT

            (self.line,) = self.ax.plot(self.x, self.y, lw = 2)
            self.scat = self.ax.scatter(self.x, self.y, s = 50, picker = 6)

            self.drag_idx = None
            self.cid_press = self.mpl_connect("button_press_event", self.on_press)
            self.cid_release = self.mpl_connect("button_release_event", self.on_release)
            self.cid_motion = self.mpl_connect("motion_notify_event", self.on_motion)
            self.cid_pick = self.mpl_connect("pick_event", self.on_pick)
            self.cid_dbl = self.mpl_connect("button_press_event", self.on_double_click)

        def redraw(self):
            self.line.set_data(self.x, self.y)
            self.scat.set_offsets(np.c_[self.x, self.y])
            self.ax.figure.canvas.draw_idle()

        def clamp_xy(self, xi, yi, idx = None):
            xi = min(max(xi, X_MIN), X_MAX)
            yi = min(max(yi, Y_MIN), Y_MAX)
            if idx is not None:
                # evita cruzar vecinos
                if idx > 0:
                    xi = max(xi, self.x[idx - 1] + 1e-6)
                if idx < len(self.x) - 1:
                    xi = min(xi, self.x[idx + 1] - 1e-6)
            return xi, yi

        def nearest_index(self, x0, y0):
            if len(self.x) == 0:
                return None
            d2 = (self.x - x0) ** 2 + (self.y - y0) ** 2
            i = int(np.argmin(d2))
            # umbral en unidades de eje (ajusta si quieres)
            thresh = 0.02 * (X_MAX - X_MIN)
            return i if abs(self.x[i] - x0) < thresh else None

        # ---------- events ----------
        def on_pick(self, event):
            # habilita arrastre al pinchar un punto
            if event.artist is self.scat and len(event.ind):
                self.drag_idx = int(event.ind[0])

        def on_press(self, event):
            # botón derecho: borrar punto más cercano
            if event.button == 3 and event.inaxes == self.ax and event.xdata is not None:
                idx = self.nearest_index(event.xdata, event.ydata)
                if idx is not None and len(self.x) > 2:
                    self.x = np.delete(self.x, idx)
                    self.y = np.delete(self.y, idx)
                    self.redraw()

        def on_double_click(self, event):
            # doble clic izquierdo: añadir punto
            if event.dblclick and event.button == 1 and event.inaxes == self.ax:
                xi, yi = event.xdata, event.ydata
                xi, yi = self.clamp_xy(xi, yi)
                # inserta manteniendo orden por x
                pos = int(np.searchsorted(self.x, xi))
                self.x = np.insert(self.x, pos, xi)
                self.y = np.insert(self.y, pos, yi)
                self.redraw()

        def on_motion(self, event):
            if self.drag_idx is None or event.inaxes != self.ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            xi, yi = self.clamp_xy(event.xdata, event.ydata, idx = self.drag_idx)
            self.x[self.drag_idx] = xi
            self.y[self.drag_idx] = yi
            self.redraw()

        def on_release(self, event):
            self.drag_idx = None

        def get_data(self, as_dataframe = True):
            idx = np.argsort(self.x)
            x_sorted, y_sorted = self.x[idx], self.y[idx]
            if as_dataframe:
                return pd.DataFrame({"t": x_sorted, "y": y_sorted})
            else:
                return list(zip(x_sorted, y_sorted))

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, title: str = "Agent speed controller"):
            super().__init__()
            self.setWindowTitle(title)
            self.canvas = CurveCanvas(self)
            toolbar = NavigationToolbar2QT(self.canvas, self)

            btn_show = QtWidgets.QPushButton("Save")
            btn_show.clicked.connect(self.save_and_close)

            btn_reset = QtWidgets.QPushButton("Reset")
            btn_reset.clicked.connect(self.reset_data)

            bottom = QtWidgets.QHBoxLayout()
            bottom.addStretch(1)
            bottom.addWidget(btn_reset)
            bottom.addWidget(btn_show)

            central = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(central)
            layout.addWidget(toolbar)
            layout.addWidget(self.canvas)
            layout.addLayout(bottom)
            self.setCentralWidget(central)
            self.resize(900, 560)

        def reset_data(self):
            self.canvas.x = X_DEFAULT
            self.canvas.y = Y_DEFAULT
            self.canvas.redraw()

        def save_and_close(self):
            self.data = self.canvas.get_data()
            self.close()

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()
    return w.data

