import heapq
import Util as util

class Road:
    def __init__(startPoint, endPoint):
        self.startPoint = startPoint
        self.endPoint = endPoint

def localConstraints(road):

def globalGoals(road):

def generateRoads(grid):
    """
    L-System algo
    initialize priority queue Q with a single entry: r(0, r0, q0)

    initialize segment list S to empty

    until Q is empty
        pop smallest r(ti, ri, qi) from Q (i.e., smallest 't')
        accepted = localConstraints(&r)
        if (accepted) {
            add segment(ri) to S
            foreach r(tj, rj, qj) produced by globalGoals(ri, qi)
            add r(ti + 1 + tj, rj, qj) to Q
    """
    pq = []
    heapq.heapify(pq)
    pq.append(firstRoad)


def generateGrid(width, height):
    """
    GenerateGrid returns a grid with generated roads depending on the width and height of the grid
    It uses a L-system algorithm to generate believable roads.
    """
    grid = [[1 for x in range(width)] for y in range(height)] 
    

