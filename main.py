# -*- coding utf-8 -*-
"""programm for confirmation that svm classfier runs correctly."""

import numpy as np
import random
from packge.classifier.euclid.euclid1 import classify_score
from packge.classifier.svm.svm import multiclassfier
from packge.ML_constalation_shaping.constallaiton_reshaping2 import constallation_reshaping
from packge.gaussian_noise.noise_1 import add_noise as add_noise1
from packge.non_gaussian_noise.noise_2 import add_noise as add_noise2
from packge.drawing_tool.draw_graph1 import draw_graph as draw_constellation
from packge.evaluate_tool.compare_SER import draw_graph
import configparser
import sys
import json
import copy
import os

# read the cofiguration file

config_name = "./" + str(sys.argv[1])
cfg = configparser.ConfigParser()
cfg.read(config_name, "UTF-8")

# define various parameters
N_test = cfg.getint("parameter", "N_test")  # the number of train data for test
N_train = cfg.getint("parameter", "N_train")  # the number of train data for svm
iteration = cfg.getint("parameter", "iteration")  # the number of training
iteration_svm = cfg.getint("parameter", "iteration_svm")  # the number of iteration of training in svm
ratio = cfg.getfloat("parameter", "ratio")  # the number of ratio of attractive and repulsive force
CNR_train = cfg.getint("parameter", "CNR_train")
learningrate_al = cfg.getfloat("parameter", "learningrate_al")
learningrate_be = cfg.getfloat("parameter", "learningrate_be")
C = cfg.getfloat("parameter", "C")
M = cfg.getint("parameter", "M")  # the number of noise symbols
step_size = cfg.getfloat("parameter", "step_size")  # the size of moving stepsize
temp_time = cfg.getint("parameter", "temp_time")  # storage time
linewidth = cfg.getint("parameter", "linewidth")  # linewidth
sam_frequency = cfg.getint("parameter", "sam_frequency")  # linewidth

uni_name = str(sys.argv[1])[2:-4]

# define the constellation
a = [1, -1, 3, -3]
shaping_points = {}

n = 0
for i in a:
    for t in a:
        n += 1
        shaping_points["s"+str(n)] = np.array([i, t])

symbol_name = list(shaping_points.keys())

new_path = "outdir/confirmation" + uni_name
if not os.path.exists(new_path):
    os.mkdir(new_path)

# initialize the objects of constellation and svm
constellation = constallation_reshaping(shaping_points, M, ratio, step_size)
classfier_svm = multiclassfier(symbol_name, N_train, C=C, learningrate_al=learningrate_al, learningrate_be=learningrate_be)

shaping_points_temp = []  # save the shaping_points and ser for backup
# learning phase
for ite in range(iteration):

    # train svm
    T_train = np.array(random.choices(symbol_name, k=N_train))
    X_train = np.array([constellation.get_shaping_points()[key] for key in T_train])
    X_train = add_noise1(X_train, CNR=CNR_train)  # add gaussian noise
    X_train = add_noise2(X_train, linewidth, sam_frequency)  # add non gaussian noise
    classfier_svm.fit(iteration_svm, X_train, T_train)

    if ite == 0:

        shaping_points_temp.append([constellation.get_shaping_points().copy(), classfier_svm.score(X_train, T_train), copy.deepcopy(classfier_svm)])

    elif not(ite % temp_time):

        print(ite)
        print("-----------------------------------")

        temp_ser = classfier_svm.score(X_train, T_train)
        if shaping_points_temp[0][1] < temp_ser:
            constellation.updata_shaping_points(shaping_points_temp[0][0].copy())
            classfier_svm = copy.deepcopy(shaping_points_temp[0][2])

            """
            # train svm
            X_train = np.array([constellation.get_shaping_points()[key] for key in T_train])
            X_train = add_noise(X_train, CNR=CNR_train)
            classfier_svm.fit(iteration_svm, X_train, T_train)
            """

        else:
            shaping_points_temp[0][0] = constellation.get_shaping_points().copy()
            shaping_points_temp[0][1] = temp_ser
            shaping_points_temp[0][2] = copy.deepcopy(classfier_svm)

    # for confirmation
    if ite % 100 == 0:
        x_boundary = np.linspace(-4, 4, 100)
        X_boundary = []
        for i in range(100):
            for t in range(100):
                X_boundary.append([x_boundary[i], x_boundary[t]])
        X_boundary = np.array(X_boundary)
        T_boundary = classfier_svm.predict(X_boundary)

        T_noise = np.array(random.choices(symbol_name, k=1600))
        X_noise = np.array([constellation.get_shaping_points()[key] for key in T_noise])
        X_noise = add_noise1(X_noise, CNR=CNR_train)  # add gaussian noise
        X_noise = add_noise2(X_noise, linewidth, sam_frequency)
        draw_constellation("confirmation"+uni_name+"/constellation_"+str(ite), constellation.get_shaping_points(), X_noise, T_noise, X_boundary, T_boundary)


    # reshape the constellation
    constellation.fit(classfier_svm.create_learning_data(X_train, T_train), classfier_svm.getlist_neighborsyb(constellation.get_shaping_points()))

# draw the constellation graph

x_boundary = np.linspace(-4, 4, 100)
X_boundary = []
for i in range(100):
    for t in range(100):
        X_boundary.append([x_boundary[i], x_boundary[t]])
X_boundary = np.array(X_boundary)
T_boundary = classfier_svm.predict(X_boundary)

T_noise = np.array(random.choices(symbol_name, k=1600))
X_noise = np.array([constellation.get_shaping_points()[key] for key in T_noise])
X_noise = add_noise1(X_noise, CNR=CNR_train)  # add gaussian noise
X_noise = add_noise2(X_noise, linewidth, sam_frequency)  # add non gaussian noise
draw_constellation("constellation_"+uni_name, constellation.get_shaping_points(), X_noise, T_noise, X_boundary, T_boundary)



# test the performance
SER_svm = []
SER_euclid = []

CNR = list(range(1, 18))
for cnr in CNR:
    T_test = np.array(random.choices(symbol_name, k=N_test))
    X_test = np.array([constellation.get_shaping_points()[key] for key in T_test])
    X_test = add_noise1(X_test, CNR=CNR_train)  # add gaussian noise
    X_test = add_noise2(X_test, linewidth, sam_frequency)  # add non gaussian noise

    SER_svm.append(classfier_svm.score(X_test, T_test))

    dataset = list(zip(X_test, T_test))
    SER_euclid.append(classify_score(shaping_points, dataset))

with open("outdir/constellation_points_"+uni_name+".json", "w") as f:
    constellation_json = {}
    for key in symbol_name:
        constellation_json[key] = constellation.get_shaping_points()[key].tolist()
    json.dump(constellation_json, f)

with open("outdir/ser_"+uni_name+".json", "w") as f:

    a = (list(zip(CNR, SER_svm)))

    json.dump(list(zip(CNR, SER_svm)), f)

SER_svm = np.array(list(zip(CNR, SER_svm)))
SER_euclid = np.array(list(zip(CNR, SER_euclid)))
draw_graph("classfier-svm", SER_svm, "classfier-euclid", SER_euclid, "comparison_ser_"+uni_name)
