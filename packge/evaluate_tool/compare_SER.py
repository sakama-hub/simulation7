# -*- coding utf-8 -*-
"""module to evaluate SER between two results"""

import numpy as np
import plotly.offline as offline
from plotly import graph_objects as go
import os


def draw_graph(name_A, data_A, name_B, data_B, filename):
    """
    Draw a graph shows relation between two SER.

    input
    ---------
    name_A:
    data_A: numpy array( * 2)
    name_B:
    data_B: numpy array( * 2)
    -----------------------------------
    output
    ----------
    nothing
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
                            x=data_A[:, 0],
                            y=data_A[:, 1],
                            mode="lines",
                            name=name_A
                            )
                )
    fig.add_trace(go.Scatter(
                            y=data_B[:, 1],
                            x=data_B[:, 0],
                            mode="lines",
                            name=name_B
                            )
                )

    fig.update_layout(yaxis_type="log", title=dict(text='comparison of SER', x=0.5,
                  font=dict(size=30)), yaxis=dict(title=dict(text='SER')),
                  xaxis=dict(title=dict(text='CNR')))

    offline.plot(fig, filename='outdir/' + filename + ".html", auto_open=False)


if __name__ == "__main__":
    """A."""
    # define various parametors
    CNR = np.arange(1, 18)
    pass
