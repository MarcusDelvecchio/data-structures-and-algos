# created this file for memoization problems, but there are a ton of these top-down DP memoization problems in the other dynamic-programming.py file

# Longest Increasing Path in a Matrix LeetCode Hard (Medium)
# : Given an m x n integers matrix, return the length of the longest increasing path in matrix.
# https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/
# TC: O(n*m), SC: O(n*m)
def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
    memo = {}
    def dfs(prev, r, c):
        if r == len(matrix) or r == -1 or c == len(matrix[0]) or c == -1 or matrix[r][c] <= prev: return 0
        if (r, c) in memo: return memo[(r,c)] # move this above the line above and the solution fails because prev comparison!
        cell = matrix[r][c]
        memo[(r,c)] = 1 + max(dfs(cell, r+1, c), dfs(cell, r, c-1), dfs(cell, r, c+1), dfs(cell, r-1, c))
        return memo[(r,c)]

    longest = 0
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            longest = max(longest, dfs(-1, r, c))
    return longest