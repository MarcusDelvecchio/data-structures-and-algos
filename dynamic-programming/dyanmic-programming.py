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
def lengthOfLIS(self, nums: List[int]) -> int:
    memo, longest = { nums[-1]: 0 }, 1
    for i in range(len(nums) - 1, -1, -1):
        for j in range(i + 1, len(nums)):
            if nums[i] >= nums[j]: continue
            memo[i] = max(memo.get(i, 0), memo[j] + 1)
            longest = max(longest, memo[i] + 1)
        if i not in memo: memo[i] = 0
    return longest