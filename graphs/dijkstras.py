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