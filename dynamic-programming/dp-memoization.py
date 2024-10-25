# created this file for memoization problems, but there are a ton of these top-down DP memoization problems in the other dynamic-programming.py file

# Longest Increasing Path in a Matrix LeetCode Hard (Medium)
# : Given an m x n integers matrix, return the length of the longest increasing path in matrix.
# https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/
# TC: O(n*m), SC: O(n*m)
def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
    memo = {}
    def dfs(prev, r, c):
        if r == len(matrix) or r == -1 or c == len(matrix[0]) or c == -1 or matrix[r][c] <= prev: return 0 # note that this also cannot be -1
        if (r, c) in memo: return memo[(r,c)] # move this above the line above and the solution fails because prev comparison!
        cell = matrix[r][c]
        memo[(r,c)] = 1 + max(dfs(cell, r+1, c), dfs(cell, r, c-1), dfs(cell, r, c+1), dfs(cell, r-1, c))
        return memo[(r,c)]

    longest = 0
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            longest = max(longest, dfs(-1, r, c))
    return longest


# Number of Increasing Paths in a Grid LeetCode Hard (Medium)
# : You are given an m x n integer matrix grid, where you can move from a cell to any adjacent cell in all 4 directions.
# : Return the number of strictly increasing paths in the grid such that you can start from any cell and end at any cell. Since the answer may be very large, return it modulo 109 + 7.
# : Two paths are considered different if they do not have exactly the same sequence of visited cells.
# https://leetcode.com/problems/number-of-increasing-paths-in-a-grid/description/
# TC: O(n*m), SC: O(n*m)
# 13 mins because tracking down small issue
def countPaths(self, grid: List[List[int]]) -> int:
    memo = {} # memo[row][col] is the number of increasing paths starting at that cell
    def dfs(prev, r, c):
        if r == -1 or r == len(grid) or c == -1 or c == len(grid[0]) or grid[r][c] <= prev: return 0
        if (r,c) in memo: return memo[(r,c)]
        memo[(r,c)] = 1
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if not (not dr or not dc): continue
                memo[(r,c)] += dfs(grid[r][c], r+dr, c+dc)
        return memo[(r, c)]
    
    paths = 0; mod = 10 ** 9 + 7
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            paths += dfs(-1, r, c) % mod
    return paths % mod