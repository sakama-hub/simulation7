# -*- coding utf-8 -*-
"""funtion to draw constallation graph."""


import plotly.offline as offline  # for offline
from plotly import graph_objects as go
import os
from random import gauss


def draw_graph(filename, shaping, symbols):
    """input shaping:list of shaping points,
             symbols:list of symbols with noise"""

    if not os.path.exists("images"):
        os.mkdir("images")
    # create each list of shaping point i and  q
    shaping_point_i = [point[0] for point in shaping]
    shaping_point_q = [point[1] for point in shaping]
    # create each list of shaping point i and q
    symbols_i = [symbol[0] for symbol in symbols]
    symbols_q = [symbol[1] for symbol in symbols]

    max_range = max(symbols_i)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=symbols_i, y=symbols_q, mode='markers',
                             name='symbols',
                             marker=dict(color='rgba(0, 255, 0, .9)', size=4)))
    fig.add_trace(go.Scatter(x=shaping_point_i, ids=[str(i) for i in range(16)],
                             y=shaping_point_q, mode='markers',
                             name='shaping points',
                             hovertext=[str(i) for i in range(16)],
                             marker=dict(symbol="star-square-dot", size=10)))
    fig.update_layout(title=dict(text='constallation shaping', x=0.5,
                      font=dict(size=30)), yaxis=dict(scaleanchor='x',
                      title=dict(text='Imaginary'),
                      range=[-max_range-1, max_range+1]),
                      xaxis=dict(title=dict(text='Real'),
                      range=[-max_range-2, max_range+2]),
                      width=600, height=550)

    offline.plot(fig, filename='images\\'+filename+'.html', auto_open=False)


if __name__ == '__main__':

    symbols = [[gauss(0.0, 1.0), gauss(0.0, 1.0)] for _ in range(100)]
    a = [1, -1, 3, -3]
    shaping_points = []
    for i in a:
        for t in a:
            shaping_points.append([i, t])
            shaping_points_x = [point[0] for point in shaping_points]
            shaping_points_y = [point[1] for point in shaping_points]
    draw_graph('practice', shaping_points, symbols)
