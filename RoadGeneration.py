import heapq
import Util as util
import random
import math

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

    def isValid(self, grid):
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

def localConstraints(road, grid):
    """
    The localConstraints() function evaluates a given road and modifies it 
    (snapped to local intersections, for instance), 
    and then determines if the road is acceptable or not
    """
    def getLegalNeighbors(pos, grid):
        x, y = pos[0], pos[1]
        nbrs = []

        for i in range(-1, 1):
            if i == 0:
                for j in range(-1, 1):
                    ny = y + j
                    if ny < len(grid[0]) and ny >= 0:
                        nbrs.append((x, ny))
            nx = x + i
            if nx < len(grid) and nx >= 0:
                nbrs.append((nx, y))

        return nbrs                

    def isClose(pos, roadPt, branchLength, grid):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set(roadPt)
        expanded.remove(pos)

        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if dist > branchLength:
                return (False, None)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))

            # if we find another road then exit
            if grid[pos_x][pos_y] == 0:
                return (True, (pos_x, pos_y))
                
            # otherwise spread out from the location to its neighbours
            nbrs = getLegalNeighbors((pos_x, pos_y), grid)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))

        # no roads found closeby
        return (False, None)

    gridRatio = len(grid) / 10

    if not road.isValid(grid):
        return False

    #if road is too short based on grid
    if road.length() <= gridRatio:
        return False

    #checks if the road is too near another road
    countClose = 0

    roadCoor = road.generateCoordinates(grid)
    for pt in roadCoor:
        if isClose(pt, roadCoor, gridRatio, grid)[0]:
            countClose += 1

    if countClose > (road.length() / 1.5):
        return False

    #if road can be extended to another road
    ept = isClose(road.endPoint, roadCoor, 3 * gridRatio, grid)
    if ept[0]:
        road.endPoint = ept[1]

    return True

    

def globalGoals(road, grid):
    """    
    globalGoals() function uses a road to suggest new branching roads
    """    
    newRoads = []
    newRoadRatio = 0.80
    newLength = road.length() * newRoadRatio
    createP = 0.8

    for pt in road.generateCoordinates(grid):

        x = pt[0]
        y = pt[0]
        dx =  road.endPoint[0] - x
        dy = road.endPoint[1] - y
        newDx = reduce(dy) 
        newDy = reduce(-dx)

        for i in range(3):
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

def placeSegments(road, grid):
    """    
    placeSegments() places the road onto the grid
    """    
    for pt in road.generateCoordinates(grid):
        x, y = pt
        grid[x][y] = 0
   

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
    pq = util.PriorityQueueWithFunction(lambda r: r[0]) 
    
    randomCoordinates = util.get_random_tuple(2, len(grid))
    firstRoad = Road(randomCoordinates.pop(), randomCoordinates.pop())

    while firstRoad.length() <= len(grid) / 2: 
        randomCoordinates = util.get_random_tuple(2, len(grid))
        firstRoad = Road(randomCoordinates.pop(), randomCoordinates.pop())
        
    pq.push((0, firstRoad))

    while not pq.isEmpty():
        time, road = pq.pop()
        accepted = localConstraints(road, grid)   

        if accepted:
            placeSegments(road, grid)
            for newRoad in globalGoals(road, grid):
                pq.push((time + 1, newRoad))

    return grid


def generateGrid(width, height):
    """
    GenerateGrid returns a grid with generated roads depending on the width and height of the grid
    It uses a L-system algorithm to generate believable roads.
    """
    grid = [[1 for x in range(width)] for y in range(height)] 
    return grid

def getFreePositions(grid):
    freePositions = []
    for i in range(0, len(grid) - 1):
        for j in range(0, len(grid[0]) - 1):
            if grid[i][j] == 0:
                freePositions.append((i, j))
    return freePositions

def generate_grid_with_roads(size):
    grid = generateRoads(generateGrid(size, size))
    free_positions = getFreePositions(grid)
    while len(free_positions) < ((size ** 2) / 2):
        grid = generateRoads(generateGrid(size, size))
        free_positions = getFreePositions(grid)
    return grid, free_positions