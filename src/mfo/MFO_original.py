"""Moth-flame optimizer used in the study. Each coordinate draws from the Python RNG."""
import numpy as np
import random
import math
import copy

def initial(pop, dim, ub, lb):
    """Draw the initial moth positions uniformly within the local bounding box."""
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub


def BorderCheck(X, ub, lb, pop, dim):
    """Clamp each proposed coordinate to its search bounds."""
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


def CalculateFitness(X, fun):
    """Evaluate each moth in the original population order."""
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


def SortFitness(Fit):
    """Sort fitness and return the corresponding population indices."""
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


def SortPosition(X, index):
    """Reorder coordinates by fitness rank."""
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def MFO(pop, dim, lb, ub, MaxIter, fun):
    """Search with logarithmic spirals and retain the best moth and per-iteration score."""
    X, lb, ub = initial(pop, dim, ub, lb)
    fitness = CalculateFitness(X, fun)
    fitnessS, sortIndex = SortFitness(fitness)
    Xs = SortPosition(X, sortIndex)
    GbestScore = copy.copy(fitnessS[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(Xs[0, :])
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):

        # Decrease the flame count over time while preserving the original rank ties.
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

        X = BorderCheck(X, ub, lb, pop, dim)
        fitness = CalculateFitness(X, fun)
        fitnessS, sortIndex = SortFitness(fitness)
        Xs = SortPosition(X, sortIndex)
        if fitnessS[0] <= GbestScore:
            GbestScore = copy.copy(fitnessS[0])
            GbestPositon[0, :] = copy.copy(Xs[0, :])
        Curve[t] = GbestScore
        # Insert the incumbent before ranking the next set of flames.
        X[-1, :] = copy.copy(GbestPositon)
        fitness = CalculateFitness(X, fun)
        fitnessS, sortIndex = SortFitness(fitness)
        Xs = SortPosition(X, sortIndex)

    return GbestScore, GbestPositon, Curve
