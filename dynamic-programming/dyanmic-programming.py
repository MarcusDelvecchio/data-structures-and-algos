def minimum_coins(m, coins):
    memo = {}
    memo[0] = 0
    for i in range(1, m + 1): 
        for coin in coins:
            subproblem = i - coin
            if subproblem < 0:
                continue
                
            memo[i] = min_ignore_none(memo.get(i), memo.get(subproblem) + 1)
    return memo[m]

def min_ignore_none(a, b):
    if a == None: return b
    if b == None: return a
    return max(a,b)

# Fibonacci Number LeetCode Easy
# calculate the nth fibonacci number
# https://leetcode.com/problems/fibonacci-number/
# my solution: https://leetcode.com/problems/fibonacci-number/
# TC: O(n), SC: O(1)
def fib(self, n: int) -> int:
    if n == 0: return 0
    curr, prev_1, prev_2 = 2, 1, 0
    while curr <= n:
        new = prev_1 + prev_2
        prev_2 = prev_1
        prev_1 = new
        curr += 1
    return prev_1

# Pascal's Triangle II LeetCode Easy
# Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.
# took like 4 mins. I think this is the best example of a dynamic programming problem so far
# TC: O(n), SC: O(n)
def getRow(self, rowIndex: int) -> List[int]:
    dp = [1]
    for _ in range(rowIndex):
        new = [1]
        for i in range(len(dp)-1):
            new.append(dp[i] + dp[i+1])
        dp = new + [1]
    return dp


# say m = 20, coins = [3,5,12]
# memo looks like { 0: 0, 3: ?, 5: ?, 17: ?} -> this is why for i starts at 1?

def options(row, col):
    return (1 if row < 1 else 0) + (1 if col < 2 else 0)

#  0 1 1
#  1 2 3


# two types of dp problems:
# 1. how many ways can you get there (number of paths)
# 2. what is the shortest path

# coins: 1. => how many sets of coins exist that add to N
# coins: 2. => what is the least number of coins that can be used to add to N

# maze: 1. => how many paths exists to get to the bottom right
# coins: 2. => what is the shortest path to get to the bottom right

# Longest Increasing Subsequence LeetCode Medium
# woohoo first dp problem woohoo
# https://leetcode.com/problems/longest-increasing-subsequence/
# took a little while lol
# this is an end-to-front / bottom-up approach, see below for the other
# approach: traverse backwards in the array and for each element, determine the length of the largest sequence that *starts* at that number
# TC: O(n^2) SC: O(n)
def lengthOfLIS(self, nums: List[int]) -> int:
    memo, longest = { nums[-1]: 0 }, 1
    for i in range(len(nums) - 1, -1, -1):
        for j in range(i + 1, len(nums)):
            if nums[i] >= nums[j]: continue
            memo[i] = max(memo.get(i, 0), memo[j] + 1)
            longest = max(longest, memo[i] + 1)
        if i not in memo: memo[i] = 0
    return longest

# front-to-end, top-down approach
# approach: traverse forward in the array and for each element, determine the length of the largest sequence that *ends* at that number
# TC: O(n^2), SC: O(n)
def lengthOfLIS(self, nums: List[int]) -> int:
    n = len(nums)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# Min Cost Climbing Stairs LeetCode Easy
# https://leetcode.com/problems/min-cost-climbing-stairs/description/
# You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps. You can either start from the step with index 0, or the step with index 1. Return the minimum cost to reach the top of the floor.
# top-down dp approach
# took just under 15 mins
# TC: O(n) -> for every n we simply check the solution to the previous two steps
# SC: O(n) -> need to store an additional 'dp' list of length n
def minCostClimbingStairs(self, cost: List[int]) -> int:
    n = len(cost) + 1
    dp = [1000] * n
    dp[0], dp[1] = 0, 0
    for i in range(2, n):
        dp[i] = min(dp[i-2] + cost[i-2], dp[i-1] + cost[i-1])
    return dp[-1]

# Triangle LeetCode Medium
# Given a triangle array, return the minimum path sum from top to bottom.
# https://leetcode.com/problems/triangle/?envType=list&envId=55ac4kuc
# took 14 mins
# dp top down approach
# TC: O(n), SC: O(n)
def minimumTotal(self, triangle: List[List[int]]) -> int:
    height = len(triangle)
    dp = [[0]*len(triangle[i]) for i in range(height)]
    dp[0][0] = triangle[0][0]
    for row in range(1, height):
        for col in range(len(triangle[row])):
            curr_cost = triangle[row][col]
            if col-1 >= 0 and col < len(triangle[row]) - 1:
                dp[row][col] = min(dp[row-1][col], dp[row-1][col-1]) + curr_cost
            elif col-1 < 0:
                dp[row][col] = dp[row-1][col] + curr_cost
            elif col >= len(triangle[row]) - 1:
                dp[row][col] = dp[row-1][col-1] + curr_cost
    return min(dp[-1]) 

# Longest Common Subsequence LeetCode Medium
# A 2D DP problem
# daily problem January 25th
# Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0
# https://leetcode.com/problems/longest-common-subsequence/submissions/1156834418/?envType=daily-question&envId=2024-01-25
# took a couple hours becuase first 2D DP problem
# watched the first 14 mins of https://www.youtube.com/watch?v=Ua0GhsJSlWM and was able to implement the solution he discussed
# TC: O(n^2) SC: O(n)
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    dp = [[0]*(len(text2)+1) for _ in range(len(text1) + 1)]
    for i in range(len(text1) - 1, -1, -1):
        for j in range(len(text2) - 1, -1, -1):
            if text1[i] == text2[j]:
                dp[i][j] = dp[i+1][j+1] + 1
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]

# Out of Boundary Paths LeetCode Medium
# see notes in duplicate solution in backtracking.py file
def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
    dp = [[1]*(n+2)] + [[1] + [0]*n + [1] for _ in range(m)] + [[1]*(n+2)]
    new_dp = dp[:]

    for _ in range(maxMove):
        new_dp = [[1]*(n+2)] + [[1] + [0]*n + [1] for _ in range(m)] + [[1]*(n+2)] # idk why you need this line
        for i in range(1, m+1):
            for j in range(1, n+1):
                new_dp[i][j] = dp[i-1][j] + dp[i+1][j] + dp[i][j+1] + dp[i][j-1]
        for k in range(m+2):
            for l in range(n+2):
                if k == 0 or k == m+1 or l == 0 or l == n+1: new_dp[k][l] = 1
        dp = new_dp.copy()
    return dp[startRow+1][startColumn+1]%(10**9+7)

# Minimum Falling Path Sum LeetCode Medium
# Tabulation & Bottom-Up solution
# see memoization solution in backtracking.py
# took about 13 mins
# https://leetcode.com/problems/minimum-falling-path-sum
# my solution: https://leetcode.com/problems/minimum-falling-path-sum/solutions/4651417/python-dynamic-programming-tabulation-bottom-up-solution-o-n-time-o-1-space/
# TC: O(n), SC: O(1) -> initial matrix is adjusted in-place
def minFallingPathSum(self, matrix: List[List[int]]) -> int:
    rows, cols = len(matrix), len(matrix[0])
    for row in range(rows-2, -1, -1):
        for col in range(cols-1, -1, -1):
            left = matrix[row+1][col-1] if col > 0 else float('inf')
            below = matrix[row+1][col]
            right = matrix[row+1][col+1] if col < cols-1 else float('inf')
            matrix[row][col] = min(left, below, right) + matrix[row][col]
    return min(matrix[0])

# Count Square Submatrices with All Ones LeetCode Medium
# DP Tabulation Approach (bottom-up)
# https://leetcode.com/problems/count-square-submatrices-with-all-ones/description/
# Given a m * n matrix of ones and zeros, return how many **square** submatrices have all ones.
# took a bit because mis understood the question
# TC: O(n^2)
def countSquares(self, matrix: List[List[int]]) -> int:

    # checks if there are all zeros in a matrix between p1 (top-left) and p2 (bottom-right)
    def is_all_zeros(p1_row, p1_col, p2_row, p2_col):
        for col in range(p2_col, p1_col+1):
            if matrix[p2_row][col] == 0:
                return False
        for row in range(p2_row, p1_row+1):
            if matrix[row][p2_col] == 0: 
                return False
        return True

    rows, cols, res = len(matrix), len(matrix[0]), 0
    for row in range(rows-1, -1, -1):
        for col in range(cols-1, -1, -1):
            for dif in range(0, min(row, col)+1):
                if dif == 0 and matrix[row][col] == 1: res += 1
                elif is_all_zeros(row, col, row-dif, col-dif): res += 1
                else: break
    return res

# Generate Parentheses LeetCode Medium
# https://leetcode.com/problems/generate-parentheses/description/
# took like 35?? idk. Had issues again because I my interpretation of the relationship between the
# sub problems was off
# TC: O(???) SC: O(??)
def generateParenthesis(self, n: int) -> List[str]:
    if n == 0: return []
    dp = ["()"]
    for i in range(n-1):
        new_dp = []
        for subproblem in dp:
            for j in range(len(subproblem)):
                new_dp.append(subproblem[:j] + "()" + subproblem[j:])
        dp = new_dp
    return set(dp)

# Coin Change LeetCode Medium
# https://leetcode.com/problems/coin-change/?envType=list&envId=55ac4kuc
# You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
# Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
# TC: O(n*c) SC: O(n)
# took like 18 mins
# approach: bottom up tabulation
def coinChange(self, coins: List[int], amount: int) -> int:
    if not amount: return 0
    dp = [float('inf')]*amount
    for i in coins:
        if i > amount: continue
        dp[amount-i] = 1
    for i in range(amount-1, -1, -1):
        for c in coins:
            if i - c < 0: continue
            dp[i-c] = min(dp[i] + 1, dp[i-c])
    return dp[0] if dp[0] != float('inf') else -1