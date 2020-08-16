# -*- coding utf-8 -*-
"""
function to draw constallation graph.

NOTE
----------
16シンボルの描画用（判定領域も描画する）
"""

import plotly.offline as offline  # for offline
from plotly import graph_objects as go
import os
import numpy as np


def draw_graph(filename, shaping_points, symbols_X, symbols_T, boundary_X, boundary_T):
    """
    Draw the constallation graph.

    inputSS
    --------
    filename: filename to preserve
    shaping_points: shaping points , dict (key=symbol name, value=np.array[])
    symbols_X: feature datas of symbols with noise, np.array()
    symbols_T: label data of symbols with noise, np.array()
    boundary_X: feature data to draw boundary
    boundary_Y: label data to draw boundary
    ----------------------------------
    output
    ----------
    nothing
    """

    if not os.path.exists("images"):
        os.mkdir("images")



    fig = go.Figure()

    rgbs = [[0, 114, 188], [243, 1, 0], [147, 123, 105], [243, 188, 33],
            [0, 107, 62], [90, 255, 25], [1, 0, 102], [39, 158, 145],
            [0, 72, 100], [111, 81, 161], [218, 82, 58], [153, 93, 70],
            [255, 200, 0], [102, 28, 35], [110, 102, 54], [55, 90, 95]]

    for i, key in enumerate(shaping_points.keys()):

        fig.add_trace(go.Scatter(
                                x=symbols_X[symbols_T == key, 0],
                                y=symbols_X[symbols_T == key, 1],
                                mode="markers",
                                marker={"opacity":1.0, "color":"rgb("+str(rgbs[i][0])+","+str(rgbs[i][1])+","+str(rgbs[i][2])+")"}
                                )
                    )

        fig.add_trace(go.Scatter(
                                x=boundary_X[boundary_T == key, 0],
                                y=boundary_X[boundary_T == key, 1],
                                mode="markers",
                                marker={"opacity":0.4, "color":"rgb("+str(rgbs[i][0])+","+str(rgbs[i][1])+","+str(rgbs[i][2])+")"}
                                )
                    )

    fig.add_trace(go.Scatter(
                            x=np.array(list(shaping_points.values()))[:, 0],
                            y=np.array(list(shaping_points.values()))[:, 1],
                            name="shaping points",
                            mode="markers",
                            marker=dict(symbol="star-square-dot", size=10, color="rgb(0,0,28)")
                            )
                )
    fig.update_layout(title=dict(text='constallation shaping', x=0.5,
                      font=dict(size=30)), yaxis=dict(scaleanchor='x',
                      title=dict(text='Imaginary')),
                      xaxis=dict(title=dict(text='Real')),
                      width=600, height=550)

    offline.plot(fig, filename='outdir/'+filename+'.html', auto_open=False)
