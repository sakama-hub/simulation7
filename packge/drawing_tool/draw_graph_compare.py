import numpy as np
from plotly.subplots import make_subplots
import plotly.offline as offline
import plotly.graph_objects as go
import random
from random import gauss
import os

def add_noise(symbols, CNR, num=16):
    """INPUT.
       Symbols : array of array that includes I valuable and Q valuable,
       CNR : carrier noise ratio,
       num : the number of symbols that  modulater can express
       ----------------------------------------------------------
       OUTPUT
       symbol_with_noise : datatype is array of array. The list includes I valuable and Q valuable added noise
    """

    #  difine the power of carrer according to input
    if num == 4:
        pow_av = 2
    elif num == 8:
        pow_av = 3
    else:
        pow_av = 10

    symbol_with_noise = symbols + (5/CNR)**(1/2) * np.array([[gauss(0, 1), gauss(0, 1)] for _ in range(len(symbols))])
    return symbol_with_noise

# 格子状の点
a = [1, -1, 3, -3]

# 通常の１６QAM
shaping_points1 = {}
n=1
for i in range(len(a)):
    for t in range(len(a)):
        shaping_points1["s"+str(n)] = [a[i], a[t]]
        n += 1
# 提案手法で求めた１６QAM
shaping_points2 = {}

shaping_points2["s1"] = np.array([-3.419196, -1.760451])
shaping_points2["s2"] = np.array([-2.846742, 0.7847883])
shaping_points2["s3"] = np.array([-2.090752, -3.886134])
shaping_points2["s4"] = np.array([-1.922201, 3.296233])
shaping_points2["s5"] = np.array([-1.510536, -0.8851778])
shaping_points2["s6"] = np.array([-0.737092, 1.353173])
shaping_points2["s7"] = np.array([-0.403241, -2.29607])
shaping_points2["s8"] = np.array([0.005448286, -0.003653037])
shaping_points2["s9"] = np.array([0.5197323, 4.225205])
shaping_points2["s10"] = np.array([0.8183248, 2.12677])
shaping_points2["s11"] = np.array([0.9193622, -1.259359])
shaping_points2["s12"] = np.array([0.995495, -4.167713])
shaping_points2["s13"] = np.array([1.720408, 0.4695732])
shaping_points2["s14"] = np.array([2.526092, -2.285913])
shaping_points2["s15"] = np.array([3.286685, 2.752862])
shaping_points2["s16"] = np.array([3.779124, -0.1614526])

N = 100000 # シンボル数
symbol_T =np.array(random.choices(list(shaping_points2.keys()), k=N))

symbol_X_1 = np.array([shaping_points1[key] for key in symbol_T])
symbol_X_1 = add_noise(symbol_X_1, 12, 16)

symbol_X_2 = np.array([shaping_points2[key] for key in symbol_T])
symbol_X_2 = add_noise(symbol_X_2, 12, 16)

if not os.path.exists("images"):
    os.mkdir("images")

fig = make_subplots(rows=1, cols=2, subplot_titles=("regular 16QAM", "reshaped 16QAM"))
fig.add_trace(
                go.Scatter(
                            x=symbol_X_1[:, 0],
                            y=symbol_X_1[:, 1],
                            mode="markers",
                            name="symbols added noise",
                            marker={"color":"rgb(0, 255, 0, 9)"}
                ),
    row=1, col=1
)
fig.add_trace(
                go.Scatter(
                            x=np.array(list(shaping_points1.values()))[:, 0],
                            y=np.array(list(shaping_points1.values()))[:, 1],
                            mode="markers",
                            name="shaping points",
                            marker=dict(symbol="star-square-dot", size=15, color="rgb(255, 0, 0)")
                ),
    row=1, col=1
)

fig.add_trace(
                go.Scatter(
                            x=symbol_X_2[:, 0],
                            y=symbol_X_2[:, 1],
                            mode="markers",
                            name="symbols added noise(reshaped)",
                            marker={"color":"rgb(0, 255, 0, 9)"},
                            showlegend = False
                ),
    row=1, col=2
)
fig.add_trace(
                go.Scatter(
                            x=np.array(list(shaping_points2.values()))[:, 0],
                            y=np.array(list(shaping_points2.values()))[:, 1],
                            mode="markers",
                            name="shaping points(reshaped)",
                            marker=dict(symbol="star-square-dot", size=15, color="rgb(255, 0, 0)"),
                            showlegend = False
                ),
    row=1, col=2
)

fig.update_xaxes(title_text="real" ,row=1, col=1)
fig.update_yaxes(title_text="imaginary" ,row=1, col=1)
fig.update_xaxes(title_text="real" ,row=1, col=2)
fig.update_yaxes(title_text="imaginary" ,row=1, col=2)

fig.update_layout(title=dict(text='constellation shaping', x=0.5,
                      font=dict(size=40)))

filename = "tutolial"

offline.plot(fig, filename='images\\'+filename+'.html', auto_open=False)
