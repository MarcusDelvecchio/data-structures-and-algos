# Path with Maximum Probability LeetCode Medium
# https://leetcode.com/problems/path-with-maximum-probability/
# TC: O(n), SC: O(n)
# You are given an undirected weighted graph of n nodes (0-indexed), represented by an edge list where edges[i] = [a, b] is an undirected edge connecting the nodes a and b with a probability of success of traversing that edge succProb[i].
# Given two nodes start and end, find the path with the maximum probability of success to go from start to end and return its success probability.
def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start_node: int, end_node: int) -> float:
    neighbors = collections.defaultdict(list)
    for i in range(len(edges)):
        neighbors[edges[i][0]].append([succProb[i], edges[i][1]])
        neighbors[edges[i][1]].append([succProb[i], edges[i][0]])
    
    visited = set()
    maxProbHeap = [[-1.0, start_node]]
    while maxProbHeap:
        prob, node = heapq.heappop(maxProbHeap)
        if node in visited: continue
        if node == end_node: return -prob
        visited.add(node)
        
        for neighbor_prob, neighbor in neighbors[node]:
            heapq.heappush(maxProbHeap, [prob*neighbor_prob, neighbor])
    return 0

# Network Delay Time LeetCode Medium
# https://leetcode.com/problems/network-delay-time
# You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.
# We will send a signal from a given node k. Return the minimum time it takes for all the n nodes to receive the signal. If it is impossible for all the n nodes to receive the signal, return -1.
# TC: O(n), SC O(n)
def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
    neighbors = collections.defaultdict(list)
    for start, end, weight in times:
        neighbors[start].append([weight, end])

    closestNodes = [[0, k]]
    visited = set()
    recent = None
    while closestNodes:
        weight, node = heapq.heappop(closestNodes)
        if node in visited: continue
        visited.add(node)
        recent = weight

        for neighbor_weight, neighbor in neighbors[node]:
            heapq.heappush(closestNodes, [weight+neighbor_weight, neighbor])
    
    # check that every node has been visited
    for i in range(1, n+1):
        if i not in visited:
            return -1
    return recent

# The Maze II LeetCode Medium
# https://leetcode.com/problems/the-maze-ii/description/
# TC: O(n), SC: O(n)
# : There is a ball in a maze with empty spaces (represented as 0) and walls (represented as 1). The ball can go through the empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.
# : Given the m x n maze, the ball's start position and the destination, where start = [startrow, startcol] and destination = [destinationrow, destinationcol], return the shortest distance for the ball to stop at the destination. If the ball cannot stop at destination, return -1.
# : The distance is the number of empty spaces traveled by the ball from the start position (excluded) to the destination (included).
# : You may assume that the borders of the maze are all walls (see examples).
# this was a bit tedious just to add the logic for finidn next rightmost, leftmost, upmost and downmost cells the ball can move to
# I could have probably implemented a more dynamic loop mechanism to traverse in directions as much as posisble but is fine to have separate modular methods
def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
    rows, cols = len(maze), len(maze[0])

    def getUpmost(row, col):
        dist = 0
        while row-1 > -1 and maze[row-1][col] != 1:
            row -= 1
            dist += 1
        return [dist, [row, col]] if dist else None
    
    def getDownmost(row, col):
        dist = 0
        while row + 1 < rows and maze[row+1][col] != 1:
            row += 1
            dist += 1
        return [dist, [row, col]] if dist else None
    
    def getRightmost(row, col):
        dist = 0
        while col+1 < cols and maze[row][col+1] != 1:
            col += 1
            dist += 1
        return [dist, [row, col]] if dist else None

    def getLeftmost(row, col):
        dist = 0
        while col - 1 > -1 and maze[row][col-1] != 1:
            col -= 1
            dist += 1
        return [dist, [row, col]] if dist else None

    visited = set()
    closestNextMoves = [[0, start]]
    while closestNextMoves:
        dist, cell = heapq.heappop(closestNextMoves)
        if tuple(cell) in visited: continue
        if cell == destination:
            return dist
        visited.add(tuple(cell))

        # add all possible next cells to the queue
        neighbors = []
        left = getLeftmost(cell[0], cell[1])
        right = getRightmost(cell[0], cell[1])
        up = getUpmost(cell[0], cell[1])
        down = getDownmost(cell[0], cell[1])
        if left: neighbors.append(left)
        if right: neighbors.append(right)
        if up: neighbors.append(up)
        if down: neighbors.append(down)

        for neighbor_dist, neighbor in neighbors:
            heapq.heappush(closestNextMoves, [neighbor_dist+dist, neighbor])

        return -1

# The Maze III LeetCode Hard
# https://leetcode.com/problems/the-maze-iii/description/
# TC: O(n), SC: O(n)
def findShortestWay(self, maze: List[List[int]], ball: List[int], hole: List[int]) -> str:
    rows, cols = len(maze), len(maze[0])

    directions = {
        'd': (1, 0),
        'u': (-1, 0),
        'l': (0, -1),
        'r': (0, 1)
    }            

    heap = [[0, ball, ""]] # next moves
    visited = set()
    best_dist = None
    best_paths = []
    while heap:
        move_dist, cell, path = heapq.heappop(heap)
        if tuple(cell) in visited: continue
        visited.add(tuple(cell))
        if best_dist != None and move_dist > best_dist:
            return min(best_paths)

        for direction in directions:
            shouldStop = False
            dx, dy = 0, 0
            farthest = [cell[0] + dx, cell[1] + dy]

            # continue in the direction while possible
            while 0 <= cell[0] + dx <= rows - 1 and 0 <= cell[1] + dy <= cols - 1 and maze[cell[0] + dx][cell[1] + dy] != 1:
                farthest = [cell[0] + dx, cell[1] + dy]
                # check if this cell is the hole
                if farthest == hole:
                    travel_dist = move_dist + max(abs(farthest[0] - cell[0]), abs(farthest[1] - cell[1]))
                    if best_dist == None or travel_dist == best_dist:
                        best_dist = travel_dist
                        best_paths.append(path + direction)
                        shouldStop = True
                        break                            
                dx += directions[direction][0]
                dy += directions[direction][1]

            if shouldStop: break
            if farthest != cell:
                travel_dist = max(abs(farthest[0] - cell[0]), abs(farthest[1] - cell[1])) + move_dist
                heapq.heappush(heap, [travel_dist, farthest, path + direction])

    return "impossible" if not best_paths else min(best_paths)

# Path With Minimum Effort LeetCode Medium
# https://leetcode.com/problems/path-with-minimum-effort/description/
# TC: O(n), SC: O(n)
def minimumEffortPath(self, heights: List[List[int]]) -> int:
    rows, cols = len(heights), len(heights[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # up, down, left, right

    nextCellHeap = [[0, 0, 0]]
    visited = set()
    while nextCellHeap:
        effort, row, col = heapq.heappop(nextCellHeap)
        if (row, col) in visited: continue
        if row == rows - 1 and col == cols - 1: return effort
        visited.add((row, col))

        # consider moving to all directions
        for row_dif, col_dif in directions:
            next_row, next_col = row + row_dif, col + col_dif
            if 0 <= next_row <= rows - 1 and 0 <= next_col <= cols - 1:
                heapq.heappush(nextCellHeap, [max(effort, abs(heights[row][col] - heights[next_row][next_col])), next_row, next_col])

# Find the City With the Smallest Number of Neighbors at a Threshold Distance LeetCode Medium
# https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/description/
# TC: O(n^2), SC: O(n)
def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
    neighbors = collections.defaultdict(list)
    for src, tar, weight in edges:
        neighbors[src].append([tar, weight])
        neighbors[tar].append([src, weight])

    min_cities = float('inf')
    min_city = None
    for i in range(n):
        heap = [[0, i]]
        visited = set()
        cities_in_range = 0
        while heap:
            path_total, city = heapq.heappop(heap)
            if path_total > distanceThreshold or city in visited: continue
            else:
                visited.add(city)
                cities_in_range += 1

            for neighbor, weight in neighbors[city]:
                heapq.heappush(heap, [path_total + weight, neighbor])
        
        # check if it is min city
        if cities_in_range - 1 <= min_cities: # covers case where is equal to but greater city number
            min_cities = cities_in_range - 1
            min_city = i
    return min_city