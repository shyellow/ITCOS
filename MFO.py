import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''' Population initialization function '''


def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub


'''Boundary checking function'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''Calculate the fitness function'''


def CalculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''Fitness ranking'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''Sort the positions based on their fitness values'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


'''Butterfly Algorithm'''


def MFO(pop, dim, lb, ub, MaxIter, fun):    # pop: Population size dim: Dimension lb, ub: Lower and upper bounds MaxIter: Maximum number of iterations fun: Fitness function
    a = 2;
    X, lb, ub = initial(pop, dim, ub, lb)  # Initialize the population
    fitness = CalculateFitness(X, fun)  # Calculate the fitness value
    fitnessS, sortIndex = SortFitness(fitness)  # Sort the fitness values
    Xs = SortPosition(X, sortIndex)
    GbestScore = copy.copy(fitnessS[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(Xs[0, :])
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):

        Flame_no = round(pop - t * ((pop - 1) / MaxIter))
        a = -1 + t * (-1) / MaxIter
        for i in range(pop):
            for j in range(dim):
                if i <= Flame_no:
                    distance_to_flame = np.abs(Xs[i, j] - X[i, j])
                    b = 1
                    r = (a - 1) * random.random() + 1

                    X[i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[i, j]
                else:
                    distance_to_flame = np.abs(Xs[i, j] - X[i, j])
                    b = 1
                    r = (a - 1) * random.random() + 1
                    X[i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * math.pi) + Xs[Flame_no, j]

        X = BorderCheck(X, ub, lb, pop, dim)  # Boundary detection
        fitness = CalculateFitness(X, fun)  # Calculate the fitness value
        fitnessS, sortIndex = SortFitness(fitness)  # Sort the fitness values
        Xs = SortPosition(X, sortIndex)  # Population ranking
        if fitnessS[0] <= GbestScore:  # Update the global optimum
            GbestScore = copy.copy(fitnessS[0])
            GbestPositon[0, :] = copy.copy(Xs[0, :])
        Curve[t] = GbestScore
        X[-1, :] = copy.copy(GbestPositon)
        fitness = CalculateFitness(X, fun)  # Calculate the fitness value
        fitnessS, sortIndex = SortFitness(fitness)  # Sort the fitness values
        Xs = SortPosition(X, sortIndex)  # Population ranking

    return GbestScore, GbestPositon, Curve
