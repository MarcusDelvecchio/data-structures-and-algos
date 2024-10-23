# Climbing Stairs LeetCode Easy
# : You are climbing a staircase. It takes n steps to reach the top.
# : Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# https://leetcode.com/problems/climbing-stairs/description/
# You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# TC: O(n), SC: O(n)
def climbStairs(self, n: int) -> int:
    if n < 2: return n
    dp = [0]*n
    dp[0], dp[1] = 1, 2
    for i in range(2, n):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n-1]

# Longest Arithmetic Subsequence of Given Difference LeetCode Medium
# https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/
# : Given an integer array arr and an integer difference, return the length of the longest subsequence in arr
# : which is an arithmetic sequence such that the difference between adjacent elements in the subsequence equals difference.
# TC: O(n), SC: O(n)
def longestSubsequence(self, arr: List[int], difference: int) -> int:
    vals = defaultdict(int) # set of all values we have seen so far in the list iterating forward, where vals[i] = longest subsequence ending at i
    best = 1
    for i in range(len(arr)):
        less = arr[i] - difference # the value we are looking for: current element minus difference
        vals[arr[i]] = 1 if less not in vals else vals[less] + 1 # if we have seen the target previous value for the current value, our new subsequence will be that subsequence length = 1
        best = max(best, vals[arr[i]])
    return best

# Destroy Sequential Targets LeetCode Medium
# https://leetcode.com/problems/destroy-sequential-targets/description
# TC: O(n)
# similar to Longest Arithmetic Subsequence of Given Difference
# given an integer array nums, return the integer that has the most other elements that are some multiple of space greater than it
def destroyTargets(self, nums: List[int], space: int) -> int:
    mod_vals, mod_counts = defaultdict(lambda:float('inf')), defaultdict(int)
    best_mod = -1
    for num in nums:
        mod = num % space
        mod_counts[mod] += 1
        mod_vals[mod] = min(mod_vals[mod], num)
        if mod_counts[mod] > mod_counts[best_mod] or (mod_counts[mod] == mod_counts[best_mod] and mod_vals[mod] < mod_vals[best_mod]):
            best_mod = mod
    return mod_vals[best_mod] if best_mod != -1 else min(nums)

# Longest Arithmetic Subsequence LeetCode Medium
# https://leetcode.com/problems/longest-arithmetic-subsequence/
# O(n^2) space O(n^2) time
def longestArithSeqLength(self, nums: List[int]) -> int:
    dp = [defaultdict(int) for i in range(len(nums))] # dp[i] = a dict of all arithmetic sequences that end at i, where dp[i][space] = length
    best = 0
    for i in range(len(nums)):
        for prev in range(i - 1, -1, -1):
            dif = nums[i] - nums[prev]
            if dif in dp[i]: continue # if we have already considered this difference with a later element,don't recalculate
            if dif in dp[prev]:
                dp[i][dif] = dp[prev][dif] + 1
            else:
                dp[i][dif] = 2
            best = max(best, dp[i][dif])
    return best




# Best Sightseeing Pair LeetCode Medium
# https://leetcode.com/problems/best-sightseeing-pair/description/
# Given an array of values, find the pair of values with the largest score
# the score is calculated by: adding the values and subtracting the distance between them in the array
# also included in greedy section. this question isn't very "dp" like, but has DP tag.
# TC: O(n), SC: O(1)
def maxScoreSightseeingPair(self, values: List[int]) -> int:
    
    # iterate through the array and hold on to the highest value, but reduce it by one every time
    max_score = best_spot = 0
    for value in values:
        max_score = max(max_score, value+best_spot)
        best_spot = max(best_spot, value)
        best_spot -= 1
    return max_score

# House Robber LeetCode Medium
# https://leetcode.com/problems/house-robber/
# You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, but you cannot rob two houses thgat are directly beside each other.
# Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.
# using DP tabulation with INPUT ARRAY (see two-variable) below
# TC: O(n), SC: O(1)
def rob(self, nums: List[int]) -> int:
    if len(nums) < 2: return max(nums)
    nums[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        nums[i] = max(nums[i-1], nums[i] + nums[i-2])
    return nums[-1]

# House Robber two variables
# 4 lines
def rob(self, nums: List[int]) -> int:
    sec_last = last = 0
    for i in range(len(nums)):
        sec_last, last = last, max(nums[i]+sec_last, last)

        # above is equivalent to
        # temp = last
        # last = max(nums[i]+sec_last, last)
        # sec_last = temp
    return last

# what problem is this?
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

# Jump Game LeetCode Medium
# https://leetcode.com/problems/jump-game/description/
# TC: O(n^2)(if all of DP is False) SC: O(n)
# only beats 5% because I guess there is a better greedy solution rather than DP
# see better greedy solution in greedy.py
def canJump(self, nums: List[int]) -> bool:
    dp = [False]*len(nums)
    for i in range(len(nums) -1, -1, -1):
        for j in range(nums[i]):
            if i+j+1 > len(nums)-2 or dp[i+j+1]:
                dp[i] = True
                break
    return dp[0] or len(dp) == 1

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
# Given an integer rowIndex, return the rowIndex-th (0-indexed) row of the Pascal's triangle.
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
# https://leetcode.com/problems/longest-increasing-subsequence/
# Given an integer array nums, return the length of the longest strictly increasing subsequence
# woohoo first dp problem woohoo
# took a little while lol
# this is an end-to-front / bottom-up approach, see below for the other
# approach: traverse backwards in the array and for each element, determine the length of the largest sequence that *starts* at that number (and INCLUDES that number)
# not that when we maintain our DP array, for every num in the array, we are trying to find the LIS that includes that number. So when we return a value, we will simply not just return the last computd item, becuase the LIS of that item could be 1. INstead we return the maxmimum LIS of the entire dp list
# TC: O(n^2) SC: O(n)
def lengthOfLIS(self, nums: List[int]) -> int:
    # THIS IS OLD SOLUTION SEE NEWER ONES BELOW
    dp, longest = { nums[-1]: 0 }, 1
    for i in range(len(nums) - 1, -1, -1):
        for j in range(i + 1, len(nums)):
            if nums[i] >= nums[j]: continue
            dp[i] = max(dp.get(i, 0), dp[j] + 1)
            longest = max(longest, dp[i] + 1)
        if i not in dp: dp[i] = 0
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

# same approach but forward to back. For each num look backwards at all nums and determine the length of the lis that ENDS at (and includes) the current num
def lengthOfLIS(self, nums: List[int]) -> int:
    dp = [1]*len(nums)
    for i in range(len(nums)):
        for j in range(i-1, -1, -1):
            if nums[i] <= nums[j]: continue
            dp[i] = max(dp[i], dp[j]+1)
    return max(dp)

# Min Cost Climbing Stairs LeetCode Easy
# https://leetcode.com/problems/min-cost-climbing-stairs/description/
# You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps. 
# You can either start from the step with index 0, or the step with index 1. Return the minimum cost to reach the top of the floor.
# top-down dp approach
# took just under 15 mins
# TC: O(n) -> for every n we simply check the solution to the previous two steps
# SC: O(n) -> need to store an additional 'dp' list of length n
# constant space solution BELOW (just use initial cost array)
def minCostClimbingStairs(self, cost: List[int]) -> int:
    n = len(cost) + 1
    dp = [1000] * n
    dp[0], dp[1] = 0, 0
    for i in range(2, n):
        dp[i] = min(dp[i-2] + cost[i-2], dp[i-1] + cost[i-1])
    return dp[-1]

# Min Cost Climbing Stairs LeetCode Easy
# TC: O(n), SC: O(1)
def minCostClimbingStairs(self, cost: List[int]) -> int:
    for i in range(len(cost)-3, -1, -1):
        cost[i] += min(cost[i+1], cost[i+2])
    return min(cost[0], cost[1])

# House Robber II LeetCode Medium
# https://leetcode.com/problems/house-robber-ii/description/
# You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle.
# That means the first house is the neighbor of the last one. Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police (you cannot rob adjacent houses)
# took like 15
# TC: O(n), SC: O(1)
def rob(self, nums: List[int]) -> int:
    if len(nums) < 3: return max(nums)
    # Since House[1] and House[n] are adjacent, they cannot be robbed together. Therefore, the problem becomes to rob either House[1]-House[n-1] or House[2]-House[n], depending on which choice offers more money. Now the problem has degenerated to the House Robber, which is already been solved.
    ans = 0
    for i in range(2): # perform the dp solution twice, where the i loop can also be used to shift the dp indices by 1. j either goes from 0 to len(nums)-1 or 1 to len(nums). Comparing the results
        sec_last = last = 0
        for j in range(i, len(nums)-(1-i)):
            sec_last, last = last, max(nums[j]+sec_last, last)
        ans = max(ans, last)
    return ans

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
# https://leetcode.com/problems/longest-common-subsequence/
# https://ics.uci.edu/~eppstein/161/960229.html
# Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0
# A 2D DP problem
# daily problem January 25th
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

# LCS again Oct 13th
# took like 17 mins no help rusty on 2D DP but got it after some thought
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    # dp[r][c] = the LCS between text1[:c] and text2[:r]
    dp = [[0 for _ in range(len(text1) + 1)] for _ in range(len(text2) + 1)]

    for r in range(len(text2)-1, -1, -1):
        for c in range(len(text1)-1, -1, -1):
            if text1[c] == text2[r]:
                dp[r][c] = 1 + dp[r+1][c+1]
                # dp[r][c] = max(1 + dp[r+1][c+1], dp[r+1][c], dp[r][c+1]) # we DON'T do this because if we have a mutual character for some subproblem, the LCS (for that subproblem) will ALWAYS include that character
            else:
                dp[r][c] = max(dp[r+1][c], dp[r][c+1])
    
    return dp[0][0]

    # text1 = "bbcdf", text2 = "acdc"
    # 

# how to (generally) identify the subproblem     
# decision: skip a letter in one, the other, or neither
# that is, find the (subsolution) LCS of of the three things:
# LCS(s1, s2) = max(
#   (
#       LCS(s1[1:], s2),
#       LCS(s1, s2[1:])
#   ) 
#   if s1[0] != s2[0] else 
#   (
#       LCS(s2[1:], s2[1:] + 1
#   )
# )

# subsolution 'overlap' / mutual exclusion:
# note that we only calculate the suibproblem after 'taking both' 
# IFF the characters at the beginning of the strings are the same
# because [...]

# Longest Palindromic Subsequence LeetCode Medium
# https://leetcode.com/problems/longest-palindromic-subsequence
# Given a string s, find the longest palindromic subsequence's length in s.
# observations:
# NOTE the subsequence does not need to be CONTIGUOUS
# TC: O(n^2), SC: O(n)
# took 5 mins after I figured out the trick but that was also after studying LCS
# trick!!!: to get the longest palindromic subsequence, find the LCS between s and rev(s)
def longestPalindromeSubseq(self, s: str) -> int:
    s_reverse = s[::-1]
    dp = [[0]*(len(s)+1) for _ in range(len(s)+1)]

    for i in range(len(s)-1, -1, -1):
        for j in range(len(s)-1, -1, -1):
            if s[i] == s_reverse[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]

# Maximum Length of Repeated Subarray LeetCode Medium
# Aka longest common CONTIGUOUS subsequence
# https://leetcode.com/problems/maximum-length-of-repeated-subarray/description/
# took 10 mins after doing LCS and few others
# what is the difference between LCS and LCCS?
# observations: same as LCS, but if the two characters being compared int he subproblem
#   : are not the same, we cannot skip one or the other
#   : so we do NOT look down or the right, we only look diagonally
#   : if the characters are the same, see see if the below subproblem's charcaters were the same
# TC: O(n^2), SC: O(n^2)
def findLength(self, nums1: List[int], nums2: List[int]) -> int:
    dp = [[0 for _ in range(len(nums2)+1)] for _ in range(len(nums1)+ 1)]

    best = 0
    for r in range(len(nums1)-1, -1, -1):
        for c in range(len(nums2)-1, -1, -1):
            if nums1[r] == nums2[c]:
                dp[r][c] = 1 + dp[r+1][c+1]
            # else: dp[r][c] = 0
            best = max(best, dp[r][c])

    return best

# Edit Distance LeetCode "Medium"
# https://leetcode.com/problems/edit-distance/
# : Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.
# : You have the following three operations permitted on a word:
# : Insert a character
# : Delete a character
# : Replace a character
# TC: O(n^2), SC: O(n^2)
# 2D DP Problem very similar to LCSS
# took like 20 mins but has to look at hints but very very simlar to LCSS (and did right afer LCSS)
# note I left the LCS lines in there to compare
# NOTE # dp[i+1][j+1] explained: changing either character to the other. If we do this, the characters become the same so it is esentially equal to the LCS of the remaining of both of those strings (esentially removing *both* characters)
def minDistance(self, word1: str, word2: str) -> int:
    dp = [[0]*(len(word2)+1) for _ in range(len(word1)+1)]

    # initialize the ends of the rows and the ends of the cols to edit distances between empty string and each substring
    for r in range(len(word1)):
        dp[r][-1] = len(word1)-r
    for c in range(len(word2)):
        dp[-1][c] = len(word2)-c

    # iterate forwards and up the 2D matrix
    # note commented lines are the exact lines from the LCS solution (and we also don't initialize row-ends and col-ends as we do above here, in LCS they stay zero)
    for i in range(len(word1)-1, -1, -1):
        for j in range(len(word2)-1, -1, -1):
            if word1[i] == word2[j]:
                dp[i][j] = dp[i+1][j+1]
            else:
                dp[i][j] = 1 + min(dp[i][j+1], dp[i+1][j], dp[i+1][j+1])
    return  dp[0][0]

# One Edit Distance LeetCode Medium
# note even though similar to above can be done without DP (think about it) see solution in greedy.py
# https://leetcode.com/problems/one-edit-distance/description/
# TC: O(n*3) = O(n) SC: O(n)
# note that this can probably be done in O(n) with two pointers
# took a while and got TLE. This problem is different in that it doesn't accept O(n^2) solutions
# see invalid solutions in invalid-solutions.py
# trying to write a more efficient solution but pretty tough
def isOneEditDistance(self, s: str, t: str) -> bool:
    if not s or not t: return len(s) == 1 or len(t) == 1
    dp = [len(s)-i for i in range(len(s)+1)]

    # populate 2D matrix with True/False values whether the substrings are 1 edit distance away
    for i in range(len(t)-1,-1,-1):
        new_dp = [0]*len(s) + [len(t)-i]
        for j in range(len(s)-1,-1,-1):
            if abs(i-j) > 1: continue
            if t[i] == s[j]:
                new_dp[j] = dp[j+1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j+1], dp[j+1])
        dp = new_dp
    return dp[0] == 1 

# Delete Operation for Two Strings LeetCode Medium
# https://leetcode.com/problems/delete-operation-for-two-strings/description/
# exact same problem as edit distance except for 1 difference:
# you cannot replace one char with the other so you can esentially not use the cell down 1 and right one when the characters are not the same (as you can in edit distance)
# TC: O(n^2), SC: O(n^2)
# took 6 mins because did all the similar problems: Largest Common Subsequence (LCS), Edit Distance, Longest Palindromic Subsequence, Delete Operations
# todo we can definitely do this with an O(n) time complexity (we only ever need two rows at a time)
def minDistance(self, word1: str, word2: str) -> int:
    dp = [[0]*(len(word2)+1) for _ in range(len(word1)+1)]

    # initial row-ends and col-ends to empty-string delete-differences
    for r in range(len(word1)):
        dp[r][-1] = len(word1) - r
    for c in range(len(word2)):
        dp[-1][c] = len(word2) - c
    
    # populate our dp matrix
    for i in range(len(word1)-1, -1, -1):
        for j in range(len(word2)-1, -1, -1):
            if word1[i] == word2[j]:
                dp[i][j] = dp[i+1][j+1]
            else:
                dp[i][j] = 1 + min(dp[i+1][j], dp[i][j+1])
    return dp[0][0]

# Minimum ASCII Delete Sum for Two Strings LeetCode Medium
# Given two strings s1 and s2, return the lowest ASCII sum of deleted characters to make two strings equal.
# https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/description/
# TC: O(n^2), SC: O(n^2)
# took 10 mins because doing above
def minimumDeleteSum(self, s1: str, s2: str) -> int:
    dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]

    # populate row-ends and col-ends with values to make either string equal with the other when the other is empty string
    # notice here the difference
    dp[-1][-1] = 0
    for r in range(len(s1)-1, -1, -1):
        dp[r][-1] = dp[r+1][-1] + ord(s1[r])
    for c in range(len(s2)-1, -1, -1):
        dp[-1][c] = dp[-1][c+1] + ord(s2[c])
    
    # populate dp matrix
    for i in range(len(s1)-1, -1, -1):
        for j in range(len(s2)-1, -1, -1):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i+1][j+1]
                # note that doing min(dp[r+1][c+1], ord(s1[r]) + dp[r+1][c], ord(s2[c]) + dp[r][c+1])
                # will still ALWAYS result in dp[r+1][c+1] because if we have the option to delete, doing so will ALWAYS produce a lesser score
            else:
                dp[i][j] = min(dp[i+1][j] + ord(s1[i]), dp[i][j+1] + ord(s2[j]))
    return dp[0][0]

# Uncrossed Lines LeetCode Medium
# You are given two integer arrays nums1 and nums2. We write the integers of nums1 and nums2 (in the order they are given) on two separate horizontal lines.
# We may draw connecting lines: a straight line connecting two numbers nums1[i] and nums2[j] such that: nums1[i] == nums2[j], and the line we draw does not intersect any other connecting (non-horizontal) line. Note that a connecting line cannot intersect even at the endpoints (i.e., each number can only belong to one connecting line).
# Return the maximum number of connecting lines we can draw in this way.
# simplify: given two integer arrays, returns the longest common subarray
# https://leetcode.com/problems/uncrossed-lines/description/
# this question is exact same thing as Longest Common Subsequence (LCS)
# TC: O(nums1*nums2) = O(n^2), SC: O(nums1*nums2)
# took 3 mins 30 seconds
def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
    # this is LCS
    dp = [[0]*(len(nums2)+1) for _ in range(len(nums1)+1)]

    for i in range(len(nums1)-1, -1, -1):
        for j in range(len(nums2)-1, -1, -1):
            if nums1[i] == nums2[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i][j+1], dp[i+1][j])
    return dp[0][0]

# Shortest Common Supersequence LeetCode Hard
# https://leetcode.com/problems/shortest-common-supersequence/description/
# appraoch: find the LCS (as a string, not a number) and iterate through LCS chars and populate the answer with the chars before that char in the LCS from each string
# TC: O(N*M) = O(n^2), SC: O(N*m) = O(n^2)
# took 40 mins
# nice, this seems like the most efficient approach and cannot be optimized further. see this exact solution to mine https://leetcode.com/problems/shortest-common-supersequence/solutions/3501177/day-403-easy-lcs-0ms-100-python-java-c-explained-approach/
# note that we don't use O(n^3) here because we handle the 2D DP tabulation matrix two rows at a time
# oct 19 / 2024 done in 17 mins (after doing LCS related problems above)
def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
    # find LCS between strings - this is O(n^3) because for every index we in 2D DP matrix we store the array
    dp = [[] for _ in range(len(str2)+1)]
    
    for i in range(len(str1)-1, -1, -1):
        new_dp = [[] for _ in range(len(str2)+1)]
        for j in range(len(str2)-1, -1, -1):
            if str1[i] == str2[j]:
                new_dp[j] = [str2[j]] + dp[j+1]
            else:
                new_dp[j] = max(new_dp[j+1], dp[j], key=len)
        dp = new_dp
    
    # add values from the strings before each character in the common subsequence
    lcs = dp[0] # note we do not need to reverse lcs because they are added in reverse order from end-to-start
    res = ""
    s1_idx = s2_idx = 0

    # for each character in the LCS, add all characters from both strings that come before that characterer, then add that character, then move onto the next character
    for c in lcs:
        while s1_idx < len(str1) and str1[s1_idx] != c:
            res += str1[s1_idx]
            s1_idx += 1
        while s2_idx < len(str2) and str2[s2_idx] != c:
            res += str2[s2_idx]
            s2_idx += 1
        res += c
        s1_idx += 1
        s2_idx += 1
    
    # fill in the rest of either string if they have not gotten to the end
    while s1_idx < len(str1):
            res += str1[s1_idx]
            s1_idx += 1
    while s2_idx < len(str2):
        res += str2[s2_idx]
        s2_idx += 1
    return res

# Maximize Number of Subsequences in a String LeetCode Medium
# https://leetcode.com/problems/maximize-number-of-subsequences-in-a-string/
# Given a text string and a two character string "pattern", where you can insert pattern[0] or pattern[1] into text once (not both, only one, once), return the maximum number of times pattern could
# appear as a substring of text if you were to place it in the optimized spot
# wordy but relatively simple esp thought it would be similar to those above but 1D and simpler
# TC: O(n), SC: O(1)
# 1. (the bulk of the solution boils down to this) find the amount of time pattern appears in text by default
# 2. either add the first (pattern[0]) to the beginning or the second (pattern[1]) to the end, whichever would create more subsequences, which depends on whichever other character appears more. If second appears more, we add first, otherwise second (intuitive)
# if the same, doesn't matter
# took 17 mins was writing lots of commends
def maximumSubsequenceCount(self, text: str, pattern: str) -> int:
    # don't even need a dp array!

    # work backwards, tracking the total occurances of pattern in text
    # and the overall count of pattern[0] vs patteern[1]
    # note num_of_As = count of occurances of pattern[0], anmd Bs = pattern[2] like (A,B)/(0,1)
    total, num_of_As, num_of_Bs = 0, 0, 0
    for i in range(len(text)-1, -1, -1):
        if text[i] == pattern[0]:
            total += num_of_Bs
            num_of_As += 1
        if text[i] == pattern[1]:
            num_of_Bs += 1
    
    return total + max(num_of_As, num_of_Bs)

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
# https://leetcode.com/problems/count-square-submatrices-with-all-ones/description/
# Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.
# DP Tabulation Approach (bottom-up)
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
# took like 35?? idk. Had issues again because my interpretation of the relationship between the sub problems was off
# TC: O(???) SC: O(??)
# cleaner solution below
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

# better solution less space and shorter
# TC: O(n^2)? SC: O(n)
def generateParenthesis(self, n: int) -> List[str]:
    dp = set(["()"])
    for _ in range(n-1):
        new = set()
        for el in dp:
            for idx in range(len(el)):
                new.add(el[:idx] + "()" + el[idx:])
        dp = new
    return list(dp)

# Coin Change LeetCode Medium
# https://leetcode.com/problems/coin-change/?envType=list&envId=55ac4kuc
# You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
# Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
# approach: bottom up tabulation. Create dp array where of length *amount* where dp[i] represents the number of coins required to make up that amount, between 0 and amount
# initialize the dp[coin] for all coins to 1, since it requires 1 coin to make up those amounts
# iterate downwards from amount to zero, and at each amount and for each coin, populate dp[amount-coin] to be qual to dp[amount] + 1
# TC: O(c+n*c) = O(n*c), where S = amount and n = num of coins SC: O(n)
# took like 18 mins
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

# Sequential Digits LeetCode Medium
# https://leetcode.com/problems/sequential-digits/
# An integer has sequential digits if and only if each digit in the number is one more than the previous digit.
# Return a sorted list of all the integers in the range [low, high] inclusive that have sequential digits.
# not too much of dynamic pgoramming but kinda
# this actually took like an hour I'm cheesed
# kind of a cheap way to solve the problem as well but soft
# TC: O(1) I'd say and O(1) space as well
def sequentialDigits(self, low: int, high: int) -> List[int]:
    dp, res, largest = [1,2,3,4,5,6,7,8,9], [], 0

    # get ALL 'sequential' integers from 0 - 10^9
    while dp:
        new_dp = []
        for num in dp:
            if str(num)[-1] == "9": continue
            new = int(str(num) + str(int(str(num)[-1])+1))
            largest = max(largest, new)
            new_dp.append(new)
        res.extend(new_dp)
        dp = new_dp
    
    # filter the integers for the fitted range
    return [num for num in res if num <= high and num >= low]

# Partition Array for Maximum Sum LeetCode Medium
# Given an integer array arr, partition the array into (contiguous) subarrays of length at most k. After partitioning, each subarray has their values changed to become the maximum value of that subarray.
# https://leetcode.com/problems/partition-array-for-maximum-sum/
# bottom-up dp tabulation
# took like 25 but had to watch a video etc still hurts my brain doing these tabulation problems
# but we'll get there
# see recursive memoization solution in backtracking.py
# TC: O(n), SC: O(n)
def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
    dp = [0]*len(arr)
    for i in range(len(arr)-1, -1, -1):
        window = min(len(arr) - i, k)
        for j in range(1, window+1):
            following = dp[i+j] if j+i < len(arr) else 0
            dp[i] = max(max(arr[i:i+j])*j + following, dp[i])
    return dp[0]

# Sort Integers by The Power Value LeetCode Medium
# The power of an integer x is defined as the number of steps needed to transform x into 1 using the following steps (see question)
# Return the kth integer in the range [lo, hi] sorted by the power value
# 1D tabulation solution
# https://leetcode.com/problems/sort-integers-by-the-power-value/description/
# took 27 mins but lost like 10 mins to a simple issue where I didn't put hi+1 in initial range...
# TC: O(???), SC: O(n)
def getKth(self, lo: int, hi: int, k: int) -> int:
    dp = [i for i in range(lo, hi+1)]
    while True:
        complete = []
        for i in range(len(dp)):
            if dp[i] == -1: continue
            if dp[i] == 1:
                k -= 1
                dp[i] = -1
                if k < 1: complete.append(lo+i)
            elif dp[i]%2 == 0:
                dp[i] //= 2
            else:
                dp[i] = dp[i]*3 + 1
        
        # return the smallest of all of the values that were complete in this round
        if complete: return min(complete)
 
# Minimum Path Cost in a Grid LeetCode Medium
# bottom-up tabulation dynamic programming
# The cost of a path in grid is the sum of all values of cells visited plus the sum of costs of all the moves made. Return the minimum cost of a path that starts from any cell in the first row and ends at any cell in the last row.
# https://leetcode.com/problems/minimum-path-cost-in-a-grid/description/
# TC: O(n), SC: O(1) -> matrix is manipulated in place
# took 7 mins
def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
    for row in range(len(grid)-2, -1, -1):
        for col in range(len(grid[0])-1, -1, -1):
            cheapest = float('inf')
            for n in range(len(grid[0])):
                price = grid[row+1][n] + moveCost[grid[row][col]][n]
                cheapest = min(cheapest, price)
            grid[row][col] += cheapest
    return min(grid[0])

# Unique Paths LeetCode Medium
# Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner of a matrix from the top-left.
# https://leetcode.com/problems/unique-paths/description/
# Dp tabulation solution
# took 7 mins but I had the implementation done in 5, just issues with ranges
# a 'm x n matrix' format is 'rows x cols'
def uniquePaths(self, m: int, n: int) -> int:
    dp = [[0]*n for _ in range(m)]
    dp[-1][-1] = 1
    for row in range(m-1, -1, -1):
        for col in range(n-1, -1, -1):
            if row == m-1 and col == n-1: continue
            below = dp[row+1][col] if row < m - 1 else 0
            right = dp[row][col+1] if col < n - 1 else 0
            dp[row][col] = below + right
    return dp[0][0]

# Minimum Path Sum LeetCode Medium
# took 9 mins because idk
# https://leetcode.com/problems/minimum-path-sum/description/
# Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path. You can only move either down or right at any point in time.
# TC: O(n), SC: O(1)
def minPathSum(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    for row in range(rows-1, -1, -1):
        for col in range(cols-1, -1, -1):
            if row == rows-1 and col == cols-1: continue
            below = grid[row+1][col] if row < rows-1 else float('inf')
            right = grid[row][col+1] if col < cols-1 else float('inf')
            grid[row][col] = min(below, right) + grid[row][col]
    return grid[0][0]

# Perfect Squares LeetCode Medium
# Given an integer n, return the least number of perfect square numbers that sum to n
# https://leetcode.com/problems/perfect-squares/
# perfect squares
# 1D Dynamic programming
# TC: O(n), SC: O(n)
# did below before but didn't add to this file. Exact same solution though just did it again here
# this one is a bit better I think because we are not re-calculating the squares from 0-root(n) every time. Below we are
def numSquares(self, n: int) -> int:
    squares = [i*i for i in range(int(math.sqrt(n))+1)]
    dp = [float('inf')]*(n+1)
    dp[0], dp[1] = 0, 1
    for i in range(1, n+1):
        for sq in squares:
            if sq > i: break
            dp[i] = min(dp[i], dp[i-sq] + 1)
    return dp[n]

# beats 68% - not as efficient (but virtually the same?) since we re-calculate the squares every time rather than once at the beginning
def numSquares(self, n: int) -> int:
    dp = [float('inf')] * (n+1)
    dp[0] = 0

    for i in range(1, n+1):
        for j in range(1, int(i ** 0.5) + 1):
            dp[i] = min(dp[i], dp[i - j*j] + 1)
    return dp[n]

# Largest Divisible Subset LeetCode Medium
# https://leetcode.com/problems/largest-divisible-subset/description/
# took like 40 to conceptualize the problem, watched a part of neetcode, he did recursion but then I realized
# it wouldn't be hard to do tabulation
# create DP list from from end-to front where dp[i] is equal to the longest divisible subset starting at i
# so going backwards for every nums[i], we check every nums[j] that comes AFTER and if nums[i] and if it divides nums[j], we add nums[i] to dp[j] and it becomes dp[i]
# TC: O(n^2) SC: O(n) -> would be O(n^2) becuase each dp[i] could theoretically contain the entire rest of the list, BUT the
# largest an array could be is size 32 before the intger limit is reached, so SC = Nx32 worst case which is still SC: O(n)
# see https://www.youtube.com/watch?v=LeRU6irRoW0 at 7:00 mins
def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
    nums.sort()
    dp = [[] for _ in range(len(nums))]
    for i in range(len(nums)-1, -1, -1):
        shortest = [nums[i]]
        for j in range(i+1, len(nums)):
            if nums[j]%nums[i] == 0:
                if not shortest or len(shortest) < len(dp[j]) + 1:
                    shortest = [nums[i]] + dp[j]
        dp[i] = shortest
    return max(dp, key=len)


# Invalid solution for Cherry Pickup II LeetCode Hard
# https://leetcode.com/problems/cherry-pickup-ii
# tried to do DP solution from bottom up and then traverse back down but this doesn't work
# fails to testcase where grid = [[1,0,0,3],[0,0,0,3],[0,0,3,3],[9,0,3,3]] - there is no look ahead function
# doesn't seem like theres a good tabulation solution so see the backtracking/top-down/recursive/memoization solution below
# nvm tabulation solution below as well
def cherryPickup(self, grid: List[List[int]]) -> int:
    for i in range(len(grid[0])):
        grid[-1][i] = (grid[-1][i], grid[-1][i])

    def get_best(x):
        return max()

    # iterate bottom-up in the grid, composing the most valuable paths for each robot
    for row in range(len(grid)-2, -1, -1):
        for col in range(len(grid[0])):
            grid[row][col] = (grid[row][col], grid[row][col] + max(grid[row+1][col-1][1] if col > 0 else 0, grid[row+1][col][1], grid[row+1][col+1][1] if col < len(grid[0]) - 1 else 0))

    # traverse down the grid to get the most valuable paths and to ensure there are no conflicts
    r, total, left, right = 0, 0, 0, len(grid[0]) -1
    while r < len(grid) - 1:
        # add current left and right before moving on
        total += grid[r][left][0]
        total += grid[r][right][0]

        # find the best next left and right cells
        best_left = max(left-1, left, left+1, key=lambda x: grid[r+1][x][1] if x < len(grid[0]) and x >= 0 else -1)
        best_right = max(right-1, right, right+1, key=lambda x: grid[r+1][x][1] if x < len(grid[0]) and x >= 0 else -1)

        if best_left == best_right:
            best_alt_right = max(right-1, right, right+1, key=lambda x: grid[r+1][x][1] if x < len(grid[0]) and x >= 0 and x != best_right else -1)
            best_alt_left = max(left-1, left, left+1, key=lambda x: grid[r+1][x][1] if x < len(grid[0]) and x >= 0 and x != best_right else -1)

            if grid[r+1][best_alt_right][1] > grid[r+1][best_alt_left][1]:
                best_right = best_alt_right
            else:
                best_left = best_alt_left

        right = best_right
        left = best_left
        r += 1
    total += grid[r][left][0] + grid[r][right][0]
    return total

# Cherry Pickup II LeetCode Hard
# RECURSIVE SOLUTION; see tabulation solution below as well
# https://leetcode.com/problems/cherry-pickup-ii/?envType=daily-question&envId=2024-02-11
# backtracking/top-down/recursive/memoization solution
# took like 10 mins after thinking of a memoization solution
# this actually ran FIRST try after writing the solution - no failed test cases and no syntax errors
# lost a million years trying tabulation just because I wanted to see if there was a way. But brute force was the answer
# TC: O(rows*cols*cols) aka O(row*cols^2) (key space is (row,col,col) so there is that many subproblems max)
# SC: O(rows*cols*cols) for key space and O(rows) for recusive depth, so O(rows*cols*cols) is larger and thus the SC
def cherryPickup(self, grid: List[List[int]]) -> int:
    memo = {}
    
    def solve(row, c1, c2):
        if row == len(grid): return 0
        if (row, c1, c2) in memo: return memo[(row, c1, c2)]

        # consider all possible choices for left and right
        best = 0
        for c1_next in range(c1-1, c1+2):
            if c1_next < 0 or c1_next > len(grid[0]) - 1: continue
            for c2_next in range(c2-1, c2+2):
                if c2_next < 0 or c2_next > len(grid[0]) - 1 or c1_next == c2_next: continue
                best = max(best, solve(row+1, c1_next, c2_next))

        memo[(row, c1, c2)] = best + grid[row][c1] + grid[row][c2]
        return memo[(row, c1, c2)]

    return solve(0, 0, len(grid[0])-1)

# Cherry Pickup II LeetCode Hard
# TABULATION SOLUTION - pretty complex and unintuitive. Table consists of col*col matrix for every possible location
# for both the left and right robots in a row, then when we move up to the next row we consider all of the possible
# combinations each of the cols that the robots could be at and the places they could go and overwrite the dp table
# because of this, the TC = O(col*col) rather than O(rows*cols*cols) above, because we only ever store
# the dp table for the previous row (bottom-up)
# see https://www.youtube.com/watch?v=c1stwk2TbNk at about 11:00 mins
# took about 20 mins
def cherryPickup(self, grid: List[List[int]]) -> int:
    cols = len(grid[0])
    dp = [[0]*cols for _ in range(cols)]

    for row in range(len(grid)-1, -1, -1):
        new_dp = [[0]*cols for _ in range(cols)]
        for c1 in range(cols):
            for c2 in range(cols):
                if c1 == c2:
                    new_dp[c1][c2] = 0
                else:
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            if c1+i < 0 or c1+i > cols-1 or c2+j < 0 or c2+j > cols-1: 
                                continue
                            new_dp[c1][c2] = max(new_dp[c1][c2], grid[row][c1] + grid[row][c2] + dp[c1+i][c2+j])
        dp = new_dp
    return dp[0][0] + dp[0][-1]

# Ways to Make a Fair Array LeetCode Medium
# An array is fair if the sum of the odd-indexed values equals the sum of the even-indexed values.
# Return the number of indices that you could choose such that after the removal, nums is fair.
# https://leetcode.com/problems/ways-to-make-a-fair-array/description/
# took 30 mins mostly because the confusion with the changes to even and odds 
# approach: loop through the list forwards and backwards once and build a dp array where dp[i] has 4 values pertaining to the sum of the even and odd values on the left and right of it
# IMPROVEMENT: there is really no need to loop forwards to sum the even and odd values before each item going forwards, we can just simultaneously do this as we loop through and 'remove' each element 
# core idea: when an element is removed, all of the odd and even items on the left of it stay the same, but all of the odd elements on the right become even and all of the even elements on the right become odd
# TC: O(n), SC: O(n)
def waysToMakeFair(self, nums: List[int]) -> int:
    # dp[i] = (evens_left, odds_left, evens_right, odds_right)
    dp = [[0,0,0,0] for _ in range(len(nums))]

    # loop forwards and backwards through the array to the sum of odd and even values to the left and right of each element
    for i in range(1, len(nums)):
        is_odd = i%2 != 0
        dp[i][0] = dp[i-1][0] + (nums[i-1] if is_odd else 0)
        dp[i][1] = dp[i-1][1] + (nums[i-1] if not is_odd else 0)

    for j in range(len(nums)-2, -1, -1):
        is_odd = j%2 != 0
        dp[j][2] = dp[j+1][2] + (nums[j+1] if is_odd else 0)
        dp[j][3] = dp[j+1][3] + (nums[j+1] if not is_odd else 0)
    
    # now for every possible item that can be removed, try to remove it
    res = 0
    for n in range(len(nums)):
        is_odd = n%2 != 0
        evens_left, odds_left, evens_right, odds_right = dp[n]
        evens = evens_left + odds_right
        odds = odds_left + evens_right
        if odds == evens: res += 1
    return res

# two-pass and cleaner solution (rather than 3)
# revised solution for Ways to Make a Fair Array LeetCode Medium
def waysToMakeFair(self, nums: List[int]) -> int:
    # dp[i] = (evens_right, odds_right)
    dp = [[0,0] for _ in range(len(nums))]

    # loop backwards through the array to get the sum of odd and even values to the right of each element
    for j in range(len(nums)-2, -1, -1):
        is_odd = j%2 != 0
        dp[j][0] = dp[j+1][0] + (nums[j+1] if is_odd else 0)
        dp[j][1] = dp[j+1][1] + (nums[j+1] if not is_odd else 0)
    
    # now for every possible item that can be removed, try to remove it, while also keeping track of the total sum of even and odd values on the left of that element
    res, evens_left, odds_left = 0, 0, 0
    for n in range(len(nums)):
        is_odd = n%2 != 0
        evens_right, odds_right = dp[n]
        evens = evens_left + odds_right
        odds = odds_left + evens_right
        if odds == evens: res += 1
        if is_odd: odds_left += nums[n]
        else: evens_left += nums[n]
    return res

# Knight Dialer LeetCode Medium
# https://leetcode.com/problems/knight-dialer/
# my solution here https://leetcode.com/problems/knight-dialer/solutions/4714185/python-3-dynamic-programming-tabulation-vs-memoization-solutions-tc-o-n-sc-o-1/
# took like 10 after doing memoization solution
def knightDialer(self, n: int) -> int:
    neighbors = {1: [8,6], 2: [7,9], 3:[4,8], 4: [3,9,0], 5: [], 6: [7, 1, 0], 7: [6,2], 8: [1,3], 9: [4,2], 0: [4,6]}
    vals = [1]*10
    for _ in range(1,n):
        new_vals = [0]*10
        for num in range(10):
            for neighbor in neighbors[num]:
                new_vals[num] += vals[neighbor]
        vals = new_vals
    return sum(vals)%(10**9+7)

# Minimum Cost For Tickets LeetCode Medium
# https://leetcode.com/problems/minimum-cost-for-tickets/
# TC: O(365) = O(1), SC: O(365) = O(1)
def mincostTickets(self, days: List[int], costs: List[int]) -> int:
    days = set(days)
    dp = [0] * 365
    for i in range(364, -1, -1):
        if i+1 in days:
            day = costs[0] + (dp[i+1] if i < 365 - 1 else 0)
            week = costs[1] + (dp[i+7] if i < 365 - 7 else 0)
            month = costs[2] + (dp[i+30] if i < 365 - 30 else 0)
            dp[i] = min(day, week, month)
        elif i != 364:
            dp[i] = dp[i+1]
    return dp[0]

# Unique Paths II LeetCode Medium
# https://leetcode.com/problems/unique-paths-ii/
# TC: O(n), SC: O(1)
# took 10 mins
def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    if obstacleGrid[-1][-1] == 1: return 0
    rows, cols = len(obstacleGrid), len(obstacleGrid[0])
    for i in range(rows-1, -1, -1):
        for j in range(cols-1, -1, -1):
            if i == rows-1 and j == cols - 1:
                obstacleGrid[i][j] = 1
            elif obstacleGrid[i][j] == 1:
                obstacleGrid[i][j] = 0
            else:
                right_paths = 0 if j == cols-1 else obstacleGrid[i][j+1]
                below_paths = 0 if i == rows-1 else obstacleGrid[i+1][j]
                obstacleGrid[i][j] = right_paths + below_paths
    return obstacleGrid[0][0]

# Decode Ways LeetCode Medium
# https://leetcode.com/problems/decode-ways/description/
# Given a string s containing only digits, return the number of ways to decode it, where digits can be decoded to letters from 1 -> A, ..., 26 -> Z
# this question took way longer than it should have
# and adding the default values at the beginning of the dp array caused this annoying +1 shift
# careful of edge cases
# TC: O(n), SC: O(n) (but SC: O(1) could be implemented)
# see O(1) solution below
def numDecodings(self, s: str) -> int:
    if s[0] == "0": return 0
    dp = [0]*(len(s)+1)
    dp[0] = 1 # first char is a valid isngle digit (cannot be double)
    for idx, c in enumerate(s):
        if idx > 0 and int(s[idx-1] + s[idx]) < 27 and s[idx-1] != "0":
            dp[idx+1] += dp[idx+1-2]
        if int(s[idx]) > 0:
            dp[idx+1] += dp[idx+1-1]
    return dp[-1]

# TC: O(n), SC: O(1)
def numDecodings(self, s: str) -> int:
    if s[0] == "0": return 0
    two_before = 1
    one_before = 1
    for idx, c in enumerate(s):
        curr = 0
        if idx > 0 and int(s[idx-1] + s[idx]) < 27 and s[idx-1] != "0":
            curr += two_before
        if int(s[idx]) > 0:
            curr += one_before
        two_before = one_before
        one_before = curr
    return curr

# Partition Equal Subset Sum LeetCode Medium
# https://leetcode.com/problems/partition-equal-subset-sum/description/
# Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.
# TC: O(n), SC: O(n^2)?
# top down dp
# had issues with key selection
def canPartition(self, nums: List[int]) -> bool:
    total = sum(nums)
    if total % 2 == 1:
        return False
    memo = {}

    def solve(nums_idx, target):
        if (nums_idx, target) in memo: return memo[(nums_idx, target)]

        if nums[nums_idx] == target:
            return True
        if nums_idx == len(nums)-1 or target < 0:
            return False

        # consider adding next coin to left pile
        if solve(nums_idx + 1, target-nums[nums_idx]):
            return True

        # consider adding next coin to right pile
        if solve(nums_idx + 1, target):
            return True
        
        memo[(nums_idx, target)] = False
        return False
    
    return solve(0, total//2)

# Longest Ideal Subsequence LeetCode Medium
# INCORRECT
# https://leetcode.com/problems/longest-ideal-subsequence/description/
# TC: O(n), SC: O(n) Memory Limit Exceeded (MLE) - Memoization Keyspace too large
# Great example of where top-down DP does not work. This leads to MLE error because input string is so large.
# Even though the keyspace is size len(s)*26, which is still O(n*26) = O(n), since the string can be so large, it still leads to MLE
# so perfect example of where bottom-up / tabulative DP must be employed
# study this problem. Should be able to realize this would happen. Can we tell from the constraints of the problem? "1 <= s.length <= 105"
def longestIdealString(self, s: str, k: int) -> int:
    memo = {}
    
    def solve(prev_letter, idx):
        if idx == len(s): return 0
        if (prev_letter, idx) in memo: return memo[(prev_letter, idx)]

        # consider taking the current letter (if we can)
        letter_idx = string.ascii_lowercase.index(s[idx])
        best = 0
        if prev_letter == None or abs(letter_idx - prev_letter) <= k:
            best = 1 + solve(letter_idx, idx + 1)

        # consider not taking current letter
        best = max(best, solve(prev_letter, idx + 1))

        # return greater result
        memo[(prev_letter, idx)] = best
        return best

    return solve(None, 0)