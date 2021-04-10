#!/usr/bin/python3

from which_pyqt import PYQT_VER
from copy import deepcopy
import traceback

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities().copy()
        count = 0
        start_time = time.time()
        route = [cities[0]]
        cities.pop(0)
        while time.time() - start_time < time_allowance:
            if len(cities) == 0:
                break
            best = 0
            m = math.inf
            for i in range(len(cities)):
                new_route = [route[-1], cities[i]]
                short = TSPSolution(new_route)
                if short.cost < m:
                    m = short.cost
                    best = i
            route.append(cities[best])
            cities.pop(best)
        bssf = TSPSolution(route)
        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    def branchAndBound(self, time_allowance=60.0):
        try:
            SCORE = 0
            DEST = 1
            CITIES = 2
            MATRIX = 3
            PATH = 4
            COST = 5

            # initialization is O(n) space, O(1) time
            bssf = self.greedy(time_allowance=time_allowance)['soln']
            cities = self._scenario.getCities()
            self.cities = cities

            heap = []
            bssfNumUpdates = 0
            maxQueueLen = 1
            numStatesCreated = 1
            numPruned = 0
            numSolutions = 0
            self.lowestCost = bssf.cost
            # getting a reduced matrix is O(n^2) time and space
            startingMatrix, lowerBound = self.initializeMatrix(cities)
            starting = tuple((lowerBound,
                              cities[0],
                              cities[1:],
                              startingMatrix,
                              [cities[0]._index],
                              lowerBound))
            heapq.heappush(heap, starting)
            startTime = time.time()
            while time.time() - startTime < time_allowance:
                #Check if heap is empty
                if not len(heap):
                    break

                # pop is O(logn) time to re-heapify and O(1) space
                nextCity = heapq.heappop(heap)

                if nextCity[5] < self.lowestCost:
                    for city in nextCity[2]:
                        if self._scenario._edge_exists[nextCity[1]._index][city._index]:
                            # Reduction is O(n^2) time & space
                            reducedTuple = self.reduceMatrix(city, nextCity[3], nextCity)
                            if not len(reducedTuple[CITIES]):
                                # this is O(n) to get the cities
                                route = self.getRoute(reducedTuple[PATH])
                                bssf = TSPSolution(route)

                                if bssf.cost < self.lowestCost:
                                    self.lowestCost = min(bssf.cost, self.lowestCost)
                                    bssfNumUpdates += 1
                                    numSolutions += 1

                            else:
                                if reducedTuple[COST] < self.lowestCost:
                                    # O(logn) time to add to heap
                                    heapq.heappush(heap, reducedTuple)
                                    numStatesCreated += 1
                                else:
                                    #Prune node
                                    numPruned += 1
                                    numStatesCreated += 1
                else:
                    numPruned += 1
                    numStatesCreated += 1

                maxQueueLen = max(len(heap), maxQueueLen)

            endTime = time.time()
            results = {}
            results['cost'] = self.lowestCost
            results['time'] = endTime - startTime
            results['count'] = numSolutions
            results['soln'] = bssf
            results['max'] = maxQueueLen
            results['total'] = numStatesCreated
            results['pruned'] = numPruned
            return results
        except Exception as error:
            print(traceback.format_exc())
            raise (error)

    def getPriority(self, score, visited):
        # Change here to try different priorities
        return score / len(visited)

    def reduceMatrix(self, dest, matrix, myTuple):
        SCORE = 0
        DEST = 1
        CITIES = 2
        MATRIX = 3
        PATH = 4
        COST = 5

        # O(1) time, O(n^2) space to copy the matrix
        tupleCopy = deepcopy(myTuple)
        matrix = matrix.copy()
        reductionCost = 0
        costToCity = matrix[tupleCopy[DEST]._index][dest._index]

        # Reduce Matrix using algorithm from class:
        # Set row and column to inf
        matrix[tupleCopy[DEST]._index] = np.inf
        matrix[:, dest._index] = np.inf
        # Set used path to inf
        matrix[dest._index][tupleCopy[DEST]._index] = np.inf
        # O(n^2) time to reduce matrix
        # No added space complexity, matrix already exists

        for row in range(matrix.shape[0]):
            # Reduce row
            rowMin = np.min(matrix[row])
            if np.isinf(rowMin):
                continue
            matrix[row] = matrix[row] - rowMin
            reductionCost += rowMin

        for col in range(matrix.shape[1]):
            #Reduce Column
            colMin = np.min(matrix[:, col])
            if np.isinf(colMin):
                continue
            matrix[:, col] = matrix[:, col] - colMin
            reductionCost += colMin

        remainingCities = tupleCopy[CITIES]
        remainingCities = self.removeCity(remainingCities, dest)

        cost = tupleCopy[COST] + costToCity + reductionCost

        newTuple = ((self.getPriority(tupleCopy[COST], tupleCopy[PATH]), dest, remainingCities, matrix,
                     tupleCopy[PATH] + [dest._index], cost))
        return newTuple

    def initializeMatrix(self, cities):
        matrix = np.full((len(cities), len(cities)), fill_value=np.inf)
        for fromIndex, city in enumerate(cities):
            for destIndex, destCity in enumerate(cities):
                if fromIndex == destIndex:
                    continue

                dist = city.costTo(destCity)
                matrix[fromIndex][destIndex] = dist

        # O(n^2) time, need to visit every cell in matrix
        reductionCost = 0

        for row in range(matrix.shape[0]):
            rowMin = np.min(matrix[row])
            matrix[row] = matrix[row] - rowMin
            reductionCost += rowMin

        for col in range(matrix.shape[1]):
            colMin = np.min(matrix[:, col])
            matrix[:, col] = matrix[:, col] - colMin
            reductionCost += colMin

        return matrix, reductionCost

    def getRoute(self, indices):
        cities = []
        for index in indices:
            cities.append(self.cities[index])
        return cities

    def removeCity(self, remaining, nextCity):
        for index, city in enumerate(remaining):
            if city._index == nextCity._index:
                indexToDelete = index
                break
        del remaining[indexToDelete]
        return remaining


    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        results = {}
        bssf = self.greedy(time_allowance=time_allowance)['soln']
        best = bssf.getListOfCities()
        route = best
        start_time = time.time()
        count = 0

        improve = True
        while time.time() - start_time < time_allowance and improve:
            improve = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue
                    new_route = route[:]
                    new_route[i:j] = route[j - 1:i - 1:-1]
                    newSol = TSPSolution(new_route)
                    if newSol._costOfRoute() < bssf._costOfRoute():
                        bssf = newSol
                        best = new_route
                        count += 1
                        improve = True
            route = best
        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results




