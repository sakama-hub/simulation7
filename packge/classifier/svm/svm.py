# -*- coding utf-8 -*-
"""module for multiclassfier using svm.

NOTE
--------
双対表現を用いたソフトマージン最適化問題を解くアルゴリズム．
カーネル関数はガウスガーネルのみ実装済み（今後ほかのカーネル関数も実装するかも）
multiclassfierクラスにsvmクラスを組み込むことで実装している
"""

import numpy as np


def gaus_kernel(X, x=[], sgm=0.5):
    """
    A kind of kernel function.

    intput
    ------------
    X:training data(feature), numpy array( * 2) of float
    x:test data(feature), numpy array(1 * 2) of float
    sgm: innternal parametor of gauss function
    ---------------------------
    output
    -------------
    val:numpy array( * 1) of float
    """
    X = X.astype(np.float64)
    if len(x) == 0:
        K = np.exp(- sgm * (((X - X[:, None]) / sgm) ** 2).sum(axis=2))
    else:
        x = x.astype(np.float64)
        K = np.exp(- sgm * (((X - x[:, None]) / sgm) ** 2).sum(axis=2))
    return K


class multiclassfier():
    """
    class for multiclassfier.
    """

    def __init__(self, symbol_name, N, kernel=gaus_kernel, C=1.0, learningrate_al=0.001, learningrate_be=0.01):
        """
        Initialise the class objects.

        intput
        ---------
        symbol_name: list of symbolname
        N: the number of traning data
        kernel: choise of kernel function
        C: strictness of soft margine(defaul=1.0)
        learningrate_al: step size of lagrange valuable(default=0.001)
        learningrate_be: step size of penalty valuable(default=0.01)
        ---------------------------------
        output
        -------
        nothing
        ---------------------------------
        NOTE
        -----------
        ------------------------------------
        """
        self.svms = {}
        for i in range(len(symbol_name)):
            self.svms[symbol_name[i]] = SVM(N=N, kernel=kernel, C=C, learningrate_al=learningrate_al, learningrate_be=learningrate_be)

    def fit(self, iteration, X, T):
        """Fit the training date.

        input
        --------
        iteration:the number of loop of learning
        X:trainig data(feature) numpy array(N * 2)
        T:training data(label) numpy array(N * 1)
        ---------------------
        output
        --------
        nothing
        """
        for key in self.svms.keys():
            T_ = np.where(T == key, 1, -1)
            self.svms[key].fit(iteration, X, T_)

    def predict(self, x):
        """Predict the label of predicting data.

        input
        --------
        x: data for predict, numpy array( * 2)
        ----------------------------
        output
        --------
        val: the result of prediction, numpy array( * 1)
        """
        convic_deg = []
        keys = np.array(list(self.svms.keys()))
        for key in keys:
            convic_deg.append(self.svms[key].predict(x))
        convic_deg = np.array(convic_deg)

        return np.array([keys[np.argmax(convic_deg[:, i])] for i in range(len(x))])

    def score(self, x, t):
        """Calculate the SER."""
        predict_label = self.predict(x)

        return 1 - (np.count_nonzero(t == predict_label) / len(t))

    def create_learning_data(self, x, t):
        """
        Create training datas for reshaping the constallation points.

        input
        -------
        x: feature data of training data
        t: label data of training data
        -----------------------
        output
        ------
        training data: datatype is dict of dict. this dict show the number of each symbol`s mistake
        """
        # initialize the training data
        training_data = {}
        for key in self.svms.keys():
            training_data[key] = {}
            for key_ in self.svms.keys():
                training_data[key][key_] = 0

        predict_labels = self.predict(x)



        for i in range(len(t)):
            if t[i] == predict_labels[i]:
                training_data[t[i]][t[i]] += 1
            else:
                training_data[t[i]][t[i]] += 1
                training_data[t[i]][predict_labels[i]] += 1

        return training_data

    def getlist_neighborsyb(self, shaping_points):
        """
        Get list of neighbor symbols.

        input
        --------
        shaping_points: datatype is dictionary.key is a symbol label and value is point
        -----------------------------------
        output
        --------
        neighbor_symbols: dict of list. each list includes neighbor symbols labels. key is a symbol name
        """
        neighbor_symbols = {}  # initialize the dict of neighbor symbols
        coordinates = self.get_coordinates(shaping_points, self.min_distances(shaping_points))  # 各シンボル点から同心円上の座標を取得
        for key in shaping_points.keys():
            neighbor_symbol = set(self.predict(coordinates[key]))
            neighbor_symbol.discard(key)
            neighbor_symbols[key] = list(neighbor_symbol)
        return neighbor_symbols

    def min_distances(self, shaping_points):
        """
        Calculate the mimimum distance from the other points.

        PARAMETERTERS
        ----------
        shpaping_points : datatype is dictionary. key is a symbol label and value is point
        ------------------------------
        RETURN
        ----------
        min_distances : datatype is dictionary. key is a symbol label and value is distance(float)
        ------------------------------
        NOTE
        this function is used in function 'getlist_neighborsyb()'
        """
        min_distances = {}
        for key in shaping_points.keys():
            min_distance = 100  #
            for key_ in shaping_points.keys():
                if key != key_:
                    distance = np.linalg.norm(shaping_points[key] - shaping_points[key_])
                    if min_distance > distance:
                        min_distance = distance
            min_distances[key] = min_distance
        return min_distances

    def get_coordinates(self, shaping_points, min_distances):
        """
        Calculate the points arround circle.

        PARAMERTERS
        -----------
        shpaping_points : datatype is dictionary. key is a symbol label and value is point
        min_distances : datatype is dictionary. key is a symbol label and value is distance(float) from closest symbol
        ------------------------------------------------
        RETURN
        -----------
        coordinates : datatype is dictionary of list of array. each array includes coordinates at a circumference with the axel 3 as a center
        -------------------------------------------------
        NOTE
        -----------
        this function is used in function 'getlist_neighborsyb()'
        """
        # initialize coordinates and add neighbor symbol labels
        coordinates = {}
        for key in min_distances.keys():
            coordinates[key] = np.array([[shaping_points[key][0]+min_distances[key]*np.cos(np.radians(degree)), shaping_points[key][1]+min_distances[key]*np.sin(np.radians(degree))] for degree in np.arange(0, 360, 1)])
        return coordinates


class SVM():
    """class for saport vector machine."""

    def __init__(self, N, kernel, C=1, learningrate_al=0.0001, learningrate_be=0.001):
        """Initialize the objects.

        input
        ----------
        N:the number of trainig data
        kernel:choise of kernel funtion
        C:strictness of soft margine
        learningrate_al:step size of lagrange valuable(default=0.001)
        learningrate_be:step size of penalty valuable(default=0.001)
        """
        self.N = N
        self.X = None
        self.T = None
        self.alpha = np.zeros(N)
        self.beta = 0.01
        self.learningrate_al = learningrate_al
        self.learningrate_be = learningrate_be
        self.kernel = kernel
        self.C = C

    def fit(self, iteration, X, T):
        """Fit the training data.

        input
        -------
        iteration:the number of loop
        X:training data(feature)
        T:trainig data(label)
        --------------------
        output
        -------
        nothing
        """
        self.alpha = np.zeros(self.N)  # initialize the valuable every learning
        self.beta = 0.01  # initiialize the valuable every learning
        self.X = X  # updata the traninig set
        self.T = T  # updata the traning set

        margine_array = np.eye(self.N) / self.C

        for _ in range(iteration):
            delta = np.ones(self.N) - T * ((self.alpha * T).dot(self.kernel(X) + margine_array)) - self.beta * T * self.alpha.dot(T)
            self.alpha += self.learningrate_al * delta
            self.beta += self.learningrate_be * self.alpha.dot(T) ** 2 / 2

    def predict(self, x):
        """
        Predict the label of predicting data.

        input
        --------
        x: data for prediction, numpy array( * 2)
        ----------------------------
        output
        -------
        val: result of prediction, numpy array( * 1)
        """
        index = self.alpha > 0
        alpha_ = self.alpha[index]
        X_ = self.X[index]
        T_ = self.T[index]

        b = (T_*(np.ones(len(X_)) - alpha_/self.C) - (alpha_ * T_)*(self.kernel(X_)).sum(axis=1)).mean()

        return ((alpha_ * T_) * (self.kernel(X_, x))).sum(axis=1) + b
