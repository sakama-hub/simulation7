# -*- coding utf-8 -*-
"""The module includes euclid classifier algorithms."""

import numpy as np


def create_data_for_learning(shaping_points, symbols):
    """
    PARAMERTERS
    -------
    shaping_points : datatype is dictionary. key is a symbol label and value is point
    symbols : datatype is list of list. a list includes a symbol point with noise and correct symbol label [[point],label]
    ----------------------------------------------------
    RETURN
    -------
    classified_result : datatype is dict of dict. this dict shows the number of individual symbol mistake
    ----------------------------------------------------
    NOTE
    ------
    you must create shaping_points's list and symbols' list with numpy.array.
    this euclid classifier is different from original euclid classifier in getlist neighbors function
    """

    # create dict of dict to retain classified data
    classified_result = {}
    for key in shaping_points.keys():
        classified_result[key] = {}
        for key_ in shaping_points.keys():
            classified_result[key][key_] = 0

    # classify each symbol and add data to classified result
    for symbol in symbols:
        euclid_distance = {}
        for key in shaping_points.keys():
            euclid_distance[key] = np.linalg.norm(shaping_points[key] - symbol[0])
        classified_result[symbol[1]][min(euclid_distance, key=euclid_distance.get)] += 1
        if symbol[1] != min(euclid_distance, key=euclid_distance.get):
            classified_result[symbol[1]][symbol[1]] += 1

    return classified_result


def classify(shaping_points, symbol_points):
    """
    PARAMERTERS
    --------
    shaping_points : datatype is dictionary. key is a symbol label and value is point
    symbol_points : datatype si list of numpy.array. a numpy.array includes two elements that shwo the shaping
    -------------------------------------------------------
    RETURN
    --------
    symbol_labels : datatype is list　or str of symbol label predicted
    -------------------------------------------------------
    NOTE
    --------
    you must create shaping_points's list and symbols' list with numpy.array.
    """
    if symbol_points.ndim == 1:
        euclid_distance = {}
        for key in shaping_points.keys():
            euclid_distance[key] = np.linalg.norm(shaping_points[key] - symbol_points)
        return min(euclid_distance, key=euclid_distance.get)
    else:
        symbol_labels = []
        for point in symbol_points:
            euclid_distance = {}
            for key in shaping_points.keys():
                euclid_distance[key] = np.linalg.norm(shaping_points[key] - point)
            symbol_labels.append(min(euclid_distance, key=euclid_distance.get))
        return symbol_labels


def classify_score(shaping_points, symbols):
    """
    PARAMETERTERS
    --------
    shpaping_points : datatype is dictionary. key is a symbol label and value is point
    symbols : datatype is list of list. a list includes a symbol point with noise and correct symbol label [[point],label]
    ----------------------------------------------------------
    RETURN
    --------
    score : datatype is fluent. this represents percentage of mistakes
    ----------------------------------------------------------
    NOTE
    --------
    """
    # initialize the number of mistakes
    num_mistake = 0
    for symbol in symbols:
        euclid_distance = {}
        for key in shaping_points.keys():
            euclid_distance[key] = np.linalg.norm(shaping_points[key] - symbol[0])
        if symbol[1] != min(euclid_distance, key=euclid_distance.get):
            num_mistake += 1

    return num_mistake / len(symbols)


def getlist_neighborsyb(shaping_points):
    """
    PARAMETERTERS
    --------
    shpaping_points : datatype is dictionary. key is a symbol label and value is point
    ----------------------------------------------------------
    RETURN
    --------
    neighbor_symbols : datatype is dict of list. each list includes neighbor symbol labels
    ----------------------------------------------------------
    NOTE
    --------
    """
    # initialize neighbor symbols dictionaly of list
    neighbor_symbols = {}
    coordinates = get_coordinates(shaping_points, min_distances(shaping_points))
    # add neighbor symbols labels to list
    for key in shaping_points.keys():
        print(coordinates[key])
        neighbor_symbol = set([classify(shaping_points, coordinate) for coordinate in coordinates[key]])
        neighbor_symbol.discard(key)
        neighbor_symbols[key] = list(neighbor_symbol)
    return neighbor_symbols


def min_distances(shaping_points):
    """
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

def get_coordinates(shaping_points, min_distances):
    """
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


if __name__ == "__main__":
    shaping_points = {'s1': np.array([1, 1]), 's2': np.array([-1, -1]), 's3':np.array([1, -1]), 's4':np.array([-1, 1]), 's5': np.array([2, 2]), 's6': np.array([1, 0])}
    symbols = [[np.array([0.5, 0.5]), 's1'], [np.array([0.5, 0.5]), 's2'], [np.array([-0.5, -0.5]), 's1'], [np.array([-0.5, -0.5]), 's2']]
    min_distances_ = {'s1': 1, 's2': 2, 's3': 3, 's4': 4}

    # print(create_data_for_learning(shaping_points, symbols))
    # print(classify(shaping_points, np.array([symbol[0] for symbol in symbols])))
    # print(classify(shaping_points, symbols[0][0]))
    # print(classify_score(shaping_points, symbols))
    # print(min_distances(shaping_points))
    # print(get_coordinates(shaping_points, min_distances_))
    print(getlist_neighborsyb(shaping_points))
