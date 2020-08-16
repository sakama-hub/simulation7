# -*- coding utf-8 -*-
"""The module includes machine learning algorithms to find out optical constallation shaping points"""

import numpy as np
import sympy

class constallation_rashaping:
    """
    Class for reshaping the points of symbol
    """

    def __init__(self, shaping_points, M, step_size=0.01):
        """
        PARAMERTERS
        ----------
        step_size : float (default 0.01)
                    parametor to control moving distance of symbol
        shaping_points : dictionary(key:symbol label, value:coordinate(numpy.array))
                         value to initialize the shaping points
        M : the number of symbols
        -------------------------------------------
        RETURN
        ----------
        nothing
        -------------------------------------------
        NOTE
        ----------
        """
        self.step_size = step_size
        self.shaping_points = shaping_points
        self.M = M
        # define the average power of trasmission symbols
        if M == 2:
            self.power_ave = 1
        elif M == 4:
            self.power_ave = 2
        elif M == 8:
            self.power_ave = 3
        elif M == 16:
            self.power_ave = 10
        elif M == 32:
            try:
                raise Exception("シンボル数32はまだ準備できてないです")
            except Exception as e:
                print(e)
        elif M == 64:
            try:
                raise Exception("シンボル数64はまだ準備できてないです")
            except Exception as e:
                print(e)

    def fit(self, classified_result, neighbor_symbols):
        """Fit the shaping points to optimal one

        PARAMERTERS
        ----------
        classified_result : dict of dict(key:symbol labels, value:the nunmber of mistakes)
                            you have to get from classifier
        neighbor_symbols : dict of list
                           you have to get from classifier
        -------------------------------------------
        RETURN
        ----------
        nothing
        -------------------------------------------
        NOTE
        ----------
        this function updatas the shaping points to optimal shaping points
        you can change the size of attractive force
        """
        # move the shaping points to optimal with attractive and repulsive force model
        new_shaping_points = self.shaping_points.copy()
        for key in self.shaping_points.keys():
            for key_ in neighbor_symbols[key]:
                # arise repulsive force
                if classified_result[key][key_]:
                    new_shaping_points[key] = new_shaping_points[key] + self.step_size * (classified_result[key][key_]/classified_result[key][key]) * self.unit_vector(self.shaping_points[key_], self.shaping_points[key])
                # arise attractive force
                else:
                    new_shaping_points[key] = new_shaping_points[key] + 0.01*self.step_size * self.unit_vector(self.shaping_points[key], self.shaping_points[key_])

        # normalization the new shaping points
        expr = 0
        x = sympy.Symbol('x')
        for key in self.shaping_points.keys():
            new_shaping_points[key] = new_shaping_points[key] + x * self.unit_vector(np.array([0.0, 0.0]), new_shaping_points[key])
            expr += new_shaping_points[key][0]**2 + new_shaping_points[key][1]**2
        expr = expr/self.M - self.power_ave
        normalization_num = float(max(sympy.solve(expr)))
        for key in self.shaping_points.keys():
            for i in range(2):
                new_shaping_points[key][i] = float(new_shaping_points[key][i].subs(x, normalization_num))
        # update shaping points
        self.shaping_points = new_shaping_points.copy()

    def get_shaping_points(self):
        """Return latest shaping points"""

        return self.shaping_points

    def unit_vector(self, v1, v2):
        """
        Return unit vector from v1 to v2
        Use in class method 'fit()'
        -------
        NOTE
        --------
        v1 nad v2 are have to been numpy array
        """

        return (v2 - v1)/np.linalg.norm(v2-v1)

if __name__ == "__main__":
    shaping_points = {'s1':np.array([1.0,1.0]),'s2':np.array([1.0,-1.0]),'s3':np.array([-1.0,-1.0]),'s4':np.array([-1.0,1.0])}
    classified_result = {'s1':{'s1':4,'s2':1,'s3':0,'s4':1},'s2':{'s1':0,'s2':4,'s3':0,'s4':0},'s3':{'s1':0,'s2':0,'s3':4,'s4':0},
                    's4':{'s1':0,'s2':0,'s3':0,'s4':4}}
    neighbor_symbols = {'s1':['s2', 's4'],'s2':['s1', 's3'],'s3':['s2', 's4'],'s4':['s1', 's3'],}
    step_size = 0.5

    shape = constallation_rashaping(step_size, shaping_points, 4)
    print(shape.get_shaping_points())
    shape.fit(classified_result, neighbor_symbols)
    print(shape.get_shaping_points())
