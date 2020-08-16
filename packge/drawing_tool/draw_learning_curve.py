# -*- coding utf-8 -*-
"""funcution to draw leraning curve graph"""


import plotly.offline as offline  # offline
from plotly import graph_objects as go
import os


def draw_learning_curve(filename, SER, iter):
    """
    PARAMETERES
    ------------
    filename : str
    SER : list of symbol error rate
    iter : int iteration
    ----------------------------------------------
    RETURN
    --------
    nothing
    -----------------------------------------------
    NOTE
    --------
    """

    if not os.path.exists("learning_curve"):
        os.mkdir("learning_curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iter, y=SER, mode='lines+markers',
                             line=dict(color='firebrick', width=2),
                             marker=dict(symbol="star-square-dot", size=10)))
    fig.update_layout(title=dict(text='constallation shaping', x=0.5,
                      font=dict(size=30)), yaxis=dict(title=dict(text='SER')),
                      xaxis=dict(title=dict(text='iteration')))
    offline.plot(fig, filename='learning_curve\\'+filename+'.html', auto_open=False)

if __name__ == '__main__':
    filename = 'tutorial'
    SER = [10, 10, 5, 3, 3, 3, 4, 3, 2, 1, 0]
    iter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    draw_learning_curve(filename, SER, iter)
