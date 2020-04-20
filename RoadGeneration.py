import heapq
import Util as util
import random
import math
import sys

def reduce(i):
    if i > 0:
        return 1
    if i < 0:
        return -1
    return 0

class Road:
    def __init__(self, startPoint, endPoint):
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.seed = random.randrange(sys.maxsize) # so that we can get a deterministic random

    def isValid(self, grid):
        if self.length() == 0:
            return False
        if self.startPoint[0] < len(grid) and self.startPoint[0] >= 0:
            if self.startPoint[1] < len(grid[0]) and self.startPoint[1] >= 0:
                if self.endPoint[0] < len(grid) and self.endPoint[0] >= 0:
                    if self.endPoint[1] < len(grid[0]) and self.endPoint[1] >= 0:
                        return True

        return False

    def length(self):
        dx = self.endPoint[0] - self.startPoint[0]
        dy = self.endPoint[1] - self.startPoint[1] 
        return math.sqrt(dx * dx + dy * dy)

    def generateCoordinates(self, grid):
        random.seed(self.seed)
        currentPt = self.startPoint
        endPt = self.endPoint
        coordinates = [currentPt]

        while currentPt != endPt:
            x = currentPt[0]
            y = currentPt[1]
            dx = endPt[0] - x
            dy = endPt[1] - y
            nextX = x + reduce(dx)
            nextY = y + reduce(dy)
            nextPt = (nextX, nextY)

            coordinates.append(nextPt)
            currentPt = nextPt

            #if already generated by another road
            if grid[x][nextY] == 0 or grid[nextX][y] == 0:
                continue
                        
            #else generate angled path
            if dx != 0 and dy != 0:
                if random.randint(0, 1) == 0:
                    coordinates.append((x, nextY))
                else:
                    coordinates.append((nextX, y))
        
        return coordinates

def localConstraints(road, grid, closeRatio, placedRoads):
    """
    The localConstraints() function evaluates a given road and modifies it 
    (snapped to local intersections, for instance), 
    and then determines if the road is acceptable or not
    """
    def getLegalNeighbors(pos, grid):
        x, y = pos[0], pos[1]
        nbrs = []

        for i in range(-1, 2):
            if i == 0:
                for j in range(-1, 2):
                    if j != 0:
                        ny = y + j
                        if ny < len(grid[0]) and ny >= 0:
                            nbrs.append((x, ny))
            else:
                nx = x + i
                if nx < len(grid) and nx >= 0:
                    nbrs.append((nx, y))

        return nbrs                

    def isClose(pos, roadPt, branchLength):
        if branchLength <= 0.9:
            return (False, None)
        for x, y in placedRoads:
            if (x, y) in roadPt: 
                continue
            dx = x - pos[0]
            dy = y - pos[1]
            if math.sqrt(dx * dx + dy * dy) < branchLength:
                 return (True, (x, y))

        return (False, None)
        # fringe = [(pos[0], pos[1], 0)]
        # expanded = set(roadPt)
        # expanded.remove(pos)

        # while fringe:
        #     pos_x, pos_y, dist = fringe.pop(0)

        #     if (pos_x, pos_y) in expanded:
        #         continue
        #     expanded.add((pos_x, pos_y))

        #     if dist > branchLength:
        #         return (False, None)

        #     # if we find another road then exit
        #     if grid[pos_x][pos_y] == 0:
        #         return (True, (pos_x, pos_y))
                
        #     # otherwise spread out from the location to its neighbours
        #     nbrs = getLegalNeighbors((pos_x, pos_y), grid)
        #     for nbr_x, nbr_y in nbrs:
        #         fringe.append((nbr_x, nbr_y, dist+1))

        # # no roads found closeby
        # return (False, None)

    gridRatio = float(len(grid)) * closeRatio
    if not road.isValid(grid):
        return False

    #if road is too short based on grid
    if road.length() <= gridRatio * 2:
        return False

    roadCoor = road.generateCoordinates(grid)
    count = -1 #-1 to not consider the first point which is always a road
    for x, y in roadCoor:
        if grid[x][y] == 0:
            count += 1

    #checks if road is making a new path
    if count > (closeRatio * len(roadCoor)):
        return False

    #checks if the road is too near another road
    countClose = 0

    # change this to checking against a list of roads instead because it's
    #computationally expensive for large grids
    for pt in roadCoor:
        if isClose(pt, roadCoor, gridRatio)[0]:
            countClose += 1

    if countClose > (float(len(roadCoor)) / 2):
        return False

    #if road can be extended to another road
    ept = isClose(road.endPoint, roadCoor, 3 * gridRatio)
    if ept[0]:
        road.endPoint = ept[1]

    return True

    

def globalGoals(road, grid, newRoadRatio, createP, createIterations):
    """    
    globalGoals() function uses a road to suggest new branching roads
    """    
    newRoads = []
    newRoadRatio = 0.80
    newLength = road.length() * newRoadRatio
    createP = 0.8

    for pt in road.generateCoordinates(grid):
        x = pt[0]
        y = pt[1]
        dx =  road.endPoint[0] - x
        dy = road.endPoint[1] - y
        newDx = reduce(dy) 
        newDy = reduce(-dx)

        for i in range(createIterations):
            if random.random() > createP:
                continue

            noise = int(newLength)
            angleNoiseX = 0
            angleNoiseY = 0

            if noise != 0:
                angleNoiseX = random.randrange(-noise, noise, 1)
                angleNoiseY = random.randrange(-noise, noise, 1)

            newEnd = (x + int(newDx * newLength + angleNoiseX), y + int(newDy * newLength + angleNoiseY))
            newRoad = Road((x, y), newEnd)

            if newRoad.isValid(grid):
                newRoads.append(newRoad)

    return newRoads

def placeSegments(road, placedRoads, grid):
    """    
    placeSegments() places the road onto the grid
    """    
    for pt in road.generateCoordinates(grid):
        x, y = pt
        grid[x][y] = 0
        placedRoads.add(pt)
   

def generateRoads(grid, closeRatio = 1/15, newRoadRatio = 0.8, createP = 0.8, createIterations = 2):
    """
    generateRoads generate roads on the grid with 0 as it's representation. 

    closeRatio is how close each road can be to each other with a lower number relaxing the limit. Ratio based on length of grid.
    newRoadRatio is the length of a new road being generated based on an existing road. 
        If less than one, shorter roads will branch of longer roads like in real life.
        
    createP is the probability in which roads are being created from a road.
    createIterations is the amount of roads generated from each position of a road. 

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
    pq = util.PriorityQueueWithFunction(lambda r: r[0]) 
    placedRoads = set()

    randomCoordinates = util.get_random_tuple(2, len(grid))
    firstRoad = Road(randomCoordinates.pop(), randomCoordinates.pop())

    while firstRoad.length() <= float(len(grid)) / 2.0: 
        randomCoordinates = util.get_random_tuple(2, len(grid))
        firstRoad = Road(randomCoordinates.pop(), randomCoordinates.pop())
        
    pq.push((0, firstRoad))

    while not pq.isEmpty():
        time, road = pq.pop()
        accepted = localConstraints(road, grid, closeRatio, placedRoads)   
        if time > len(grid):
            break

        if accepted:
            placeSegments(road, placedRoads, grid)
            for newRoad in globalGoals(road, grid, newRoadRatio, createP, createIterations):
                pq.push((time + 1, newRoad))

    return grid


def generateGrid(width, height, unpassable_n = 1):
    """
    GenerateGrid returns a grid with generated roads depending on the width and height of the grid
    It uses a L-system algorithm to generate believable roads.
    """
    grid = [[unpassable_n for x in range(width)] for y in range(height)] 
    return grid

def getFreePositions(grid):
    freePositions = []
    for i in range(0, len(grid)):
        for j in range(0, len(grid[0])):
            if grid[i][j] == 0:
                freePositions.append((i, j))
    return freePositions

def generate_grid_with_roads(size, unpassable_n = 1):
    grid = generateRoads(generateGrid(size, size, unpassable_n))
    free_positions = getFreePositions(grid)
    while len(free_positions) < ((size ** 2) / 2):
        grid = generateRoads(generateGrid(size, size, unpassable_n), 0, 0.8, 0.9, 5)
        free_positions = getFreePositions(grid)
    return grid, free_positions
