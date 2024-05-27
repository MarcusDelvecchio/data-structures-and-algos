from types import List
from collections import defaultdict, Counter

# Kids With the Greatest Number of Candies LeetCode Easy
# https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/description/?envType=study-plan-v2&envId=amazon-spring-23-high-frequency
# given an array of integers each representing a kid with that integer number of candies, return an array of boolean values representing whether or not each kid will have more candies than all of the other kids (or equal to the max)
def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
    maxx = max(candies)
    return [candies[i] + extraCandies >= maxx for i in range(len(candies))]

# Relative Ranks LeetCode Easy
# https://leetcode.com/problems/relative-ranks/description/?envType=daily-question&envId=2024-05-08
# Given an integer array score of size n, where score[i] is the score of the ith athlete in a competition. All the scores are guaranteed to be unique.
# return a list of representing the place of each score in the overall scores, and replace 1st, 2nd and 3rd rank with medal strings (see desc)
# TC: O(nlogn), SC: O(n)
# actually had some issues with this and trying to doit in 3 lines, took >15 mins
def findRelativeRanks(self, score: List[int]) -> List[str]:
    medals = {1: "Gold Medal", 2: "Silver Medal", 3: "Bronze Medal"}
    scores = {s: idx for idx, s in enumerate(sorted(score))}
    return [str(len(score) - scores[s]) if (len(score) - scores[s]) not in medals else medals[(len(score) - scores[s])] for s in score]

# Two Sum Easy
# https://leetcode.com/problems/two-sum/
# TC: O(n), SC: O(n)
def twoSum(self, nums: List[int], target: int) -> List[int]:
    find = {}
    for i in range(len(nums)):
        if nums[i] in find:
            return [find[nums[i]], i]
        find[target - nums[i]] = i

# Find the Town Judge LeetCode Easy
# daily problem  feb 22
# took like 5 mins
def findJudge(self, n: int, trust: List[List[int]]) -> int:
    if n == 1: return 1
    trusted_by, trusts, candidates = defaultdict(int), defaultdict(int), []
    for t in trust:
        trusted_by[t[1]] += 1
        trusts[t[0]] += 1

        if trusted_by[t[1]] == n - 1:
            candidates.append(t[1])
    for can in candidates:
        if not trusts[can]:
            return can
    return -1

# Time Needed to Buy Tickets LeetCode Medium
# https://leetcode.com/problems/time-needed-to-buy-tickets/
# Took 4 mins
# TC: O(n), SC: O(1)
def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
    i = time = 0
    while True:
        if i == len(tickets):
            i = 0
        if tickets[i]:
            tickets[i] -= 1
            time += 1
        if not tickets[k]:
            return time
        i += 1

# Plates Between Candles LeetCode Medium
# https://leetcode.com/problems/plates-between-candles/description/
# given a string consisting of characters "*" and "|" representing plates and candles respecitvely, given a list of queries [from, to],
# for each query, return the number of plates that exist betwen candles in the substring created by the query
# For example, s = "||**||**|*", and a query [3, 8] denotes the substring "*||**|". The number of plates between candles in this substring is 2, as each of the two plates has at least one candle in the substring to its left and right.
# TC: O(n) -> each query is O(1)
# beats 99% runtime
def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
    ans = []

    # get the number of plates to the left of every candle
    plates_to_left = [-1]*len(s)
    next_candle_to_right = [-1]*len(s)
    next_candle_to_left = [-1]*len(s)
    plates_seen = 0
    last_candle_to_left = -1
    for i in range(len(s)):
        if s[i] == "*":
            plates_seen += 1
            next_candle_to_left[i] = last_candle_to_left
        else:
            plates_to_left[i] = plates_seen
            next_candle_to_left[i] = i
            last_candle_to_left = i

    # get next candle to the right for every candle
    last_candle_to_right = -1
    for i in range(len(s)-1, -1, -1):
        if s[i] == "*":
            next_candle_to_right[i] = last_candle_to_right
        else:
            next_candle_to_right[i] = i
            last_candle_to_right = i

    for from_, to_ in queries:
        candle_right_of_left, candle_left_of_right  = next_candle_to_right[from_], next_candle_to_left[to_]
        # count the number of plates in between
        if candle_right_of_left == -1 or candle_left_of_right == -1 or candle_right_of_left >= candle_left_of_right:
            ans.append(0)
        else:
            ans.append(plates_to_left[candle_left_of_right] - plates_to_left[candle_right_of_left])
    return ans

# Degree of an Array LeetCode Easy
# https://leetcode.com/problems/degree-of-an-array/description/
# TC: O(n), SC: O(n)
# : Given a non-empty array of non-negative integers nums, the degree of this array is defined as the maximum frequency of any one of its elements.
# Your task is to find the smallest possible length of a (contiguous) subarray of nums, that has the same degree as nums.
# misread the question and didn't realize edge case when multiple items can be the most common element, so we have to look for multiple subwarrays with those elements
def findShortestSubArray(self, nums: List[int]) -> int:
    num_indices = collections.defaultdict(list)
    max_count = 0
    for idx, num in enumerate(nums):
        num_indices[num].append(idx)
        max_count = max(max_count, len(num_indices[num]))
    smallest_sub = float('inf')
    for num in num_indices:
        if len(num_indices[num]) == max_count:
            smallest_sub = min(smallest_sub, (max(num_indices[num]) - min(num_indices[num]) + 1))
    return smallest_sub

# took about an hour but O(n+m) not not true Hard solution
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    p1, p2 = 0, 0 

    new = []
    length = 0
    even = (len(nums1) + len(nums2))%2 == 0
    while length < floor((len(nums1) + len(nums2))/2 + 1):
        if p1 == len(nums1):
            if p2 == len(nums2):
                break
            new.append(nums2[p2])
            p2 += 1
        elif p2 == len(nums2) or nums1[p1] < nums2[p2]:
            new.append(nums1[p1])
            p1 += 1
        else:
            new.append(nums2[p2])
            p2 += 1
        length += 1
        
    if even:
        return (new[len(new) - 2] + new[len(new) - 1]) / 2
    else:
        return new[-1] 

# Analyze User Website Visit Pattern LeetCode Medium
# https://leetcode.com/problems/analyze-user-website-visit-pattern/description/
# : long description, problem is pretty tedious
# was trying for a largest-common substring type approach but after reading the hint realized that you could simply find all possible patterns
# for every user and then just compare them all. We can do this becuase a pattern is a fixed length fo 3 rather than being any size
# becuase of this, this limits our time complexity
# TC: O()
def mostVisitedPattern(self, username: List[str], timestamp: List[int], website: List[str]) -> List[str]:
    # find all access patterns for all users - O(n)
    user_websites = collections.defaultdict(list)
    for i in range(len(website)):
        user_websites[username[i]].append([timestamp[i], website[i]])
    
    # sorts the items by access timestamp - O(users*log*timestamp/user) == O(nlogn)
    for user in user_websites:
        user_websites[user] = sorted(user_websites[user], key=lambda x: x[0])

    # get all possible patterns for all users - O(n^3) where n is website acess data for all users or worst case all are associated with one user giving O(n^3)
    user_patterns = collections.defaultdict(list)
    for user in user_websites:
        if len(user_websites[user]) < 3: continue
        for first in range(len(user_websites[user])-2):
            for second in range(first+1, len(user_websites[user])-1):
                for third in range(second+1, len(user_websites[user])):
                    user_patterns[user].append((user_websites[user][first][1], user_websites[user][second][1], user_websites[user][third][1]))
    
    # find lowest common pattern between all users O(n)
    best = None
    best_score = 0
    for user in user_patterns:
        for pattern in user_patterns[user]:
            score = 0
            for user in user_patterns:
                if pattern in user_patterns[user]:
                    score += 1
            if score > best_score or score == best_score and pattern < best:
                best_score = score
                best = pattern
    return best

# Trapping Rain Water LeetCode Hard
# https://leetcode.com/problems/trapping-rain-water/description/
# took 34:00 mins
# wish the solution was cleaner but is what it is
def trap(self, height: List[int]) -> int:
    w, current, temp = 0, 0, 0
    cont = False

    # put all vals into a dict
    vals = defaultdict(int)
    for value in height:
        vals[value] += 1

    for i in range(len(height)):
        if cont or height[i] >= current:
            cont = False
            w += temp

            # look ahead to find if a value exists that is greater or equal to the current val
            j, found = i+1, False
            while j < len(height):
                if height[j] >= height[i]:
                    found = True
                    break
                j += 1
            if found:
                current = height[i]
            else:
                t = height[i] - 1
                while t > 0:
                    if t in vals and vals[t]:
                        current = t
                        break
                    t -= 1
                if t == 0:
                    cont = True
            temp = 0
        else:
            temp += current - height[i]
        vals[height[i]] -= 1
    return w

# note on "amortized" complexity of appending to array
# That means the whole operation of pushing n objects onto the list is O(n). If we amortize that per element, it's O(n)/n = O(1).
# https://stackoverflow.com/questions/33044883/why-is-the-time-complexity-of-pythons-list-append-method-o1

# LeetCode daily Jan 15th - Find Players With Zero or One Losses Medium
# You are given an integer array matches where matches[i] = [winner i, loser i] indicates that the player winner i defeated player loser i in a match. Return is a list of all players that have not lost any matches and a list of all players that have lost exactly one match.
# https://leetcode.com/problems/find-players-with-zero-or-one-losses/description/?envType=daily-question&envId=2024-01-15
# TC: O(nlogn) SC: O(n)
def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
    winners, losers, losses, used = [], [], defaultdict(int), {}

    # get arrays of winners and losers along with a dict for losers and thei number of losses
    for match in matches:
        winners.append(match[0])
        losers.append(match[1])
        losses[match[1]] += 1
    
    # sort the lists
    winners.sort()
    losers.sort()

    # collect the winners that are not in losers
    zero_losses = []
    for winner in winners:
        if winner not in losses and winner not in used:
            zero_losses.append(winner)
            used[winner] = True

    single_losses = [loser for loser in losers if losses[loser] == 1]
    return [zero_losses, single_losses]

# LeetCode Daily Jan 22 Set Mismatch Easy
# https://leetcode.com/problems/set-mismatch/description/?envType=daily-question&envId=2024-01-22
# given a set of len n that original contains nums from 1 to n, but has one num replaced with another num between 1 and n
# return the num that has a duplicate and the num that is missing
def findErrorNums(self, nums: List[int]) -> List[int]:
    vals, duplicate, missing = Counter(nums), None, None

    for i in range(1, len(nums) + 1):
        if i not in vals:
            missing = i
        if vals[i] == 2:
            duplicate = i 

    return [duplicate, missing]

# Best Time to Buy and Sell Stock LeetCode Easy
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
# You are given an array prices where prices[i] is the price of a given stock on the ith day.
# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
# thought this was going to be a O(n^2) solution because of the nature of needing to compare array elements with each other
# but didn't realize this was not the case because of the forward-moving solution, as elements only ever need to compare themselves with ones in front
# I actually had to watch a video for this though because I hadn't done 2 pointer solutions in a bit and was looking for a dynamic programming solution
# TC: O(n), SC: O(1)
def maxProfit(self, prices: List[int]) -> int:
    left, right, maxx = 0, 1, 0
    while right < len(prices):
        maxx = max(maxx, prices[right]-prices[left])
        if prices[right] < prices[left]:
            left = right
        right += 1
    return maxx

# ugly solution ugly question I wouldn't even bother
# Divide Array Into Arrays With Max Difference LeetCode Medium
# https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/description/?envType=daily-question&envId=2024-02-01
def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
    groups, prev, res = [], None, []

    # sort the array (nlogn)
    nums.sort()

    # divide the array into groups with values all within k
    group = [nums[0]]
    for i in range(1, len(nums)):
        if nums[i] - nums[i-1] > k:
            groups.append(group)
            group = [nums[i]]
        else:
            group.append(nums[i])
        
    if group: groups.append(group)

    # break the groups up into arrays of size 3
    for group in groups:
        curr = []
        for num in group:
            if len(curr) < 3:
                curr.append(num)
            else:
                res.append(curr)
                curr = [num]
        
        if len(curr) == 3:
            res.append(curr)

    # validate the group - check that all groups have length 3, all values are within k, and all items have been included in a group
    total = 0
    for group in res:
        total += len(group)
        if len(group) != 3:
            return []
        for num in group:
            if num - group[0] > k: return []
    if total != len(nums):
        return []
    return res

# Find Polygon With the Largest Perimiter LeetCode Medium
# https://leetcode.com/problems/find-polygon-with-the-largest-perimeter/description/?envType=daily-question&envId=2024-02-15
# took 6 mins
# TC: O(n), SC: O(1)
def largestPerimeter(self, nums: List[int]) -> int:
    nums.sort()
    total, curr, best = 0, 0, 0
    while curr < len(nums):
        if nums[curr] < total:
            best = total + nums[curr]
        total += nums[curr]
        curr += 1
    return best or -1

# Rearrange Array Elements by Sign LeetCode Medium
# took like 4 mins because was trying to come up with SC: O(1) solution but this would be pretty merked
# TC: O(n), SC: O(n)
def rearrangeArray(self, nums: List[int]) -> List[int]:
    positives, negatives, res = [], [], []
    for num in nums:
        if num < 0:
            negatives.append(num)
        else:
            positives.append(num)
    pos_p, neg_p, pos = 0,0, True
    for _ in range(len(nums)):
        if pos:
            res.append(positives[pos_p])
            pos_p += 1
        else:
            res.append(negatives[neg_p])
            neg_p += 1
        pos = not pos
    return res

# Minimum Length of String After Deleting Similar Ends LeetCode Medium
# https://leetcode.com/problems/minimum-length-of-string-after-deleting-similar-ends/solutions/4824224/beat-100-00-full-explanation-with-pictures/?envType=daily-question&envId=2024-03-05
# took like 15 mins because 1 off erros and was rushing - didn't really care to plan solution, just started coding
# TC: O(n), SC: O(1)
def minimumLength(self, s: str) -> int:
    prefix_idx = 0
    suffix_idx = len(s) - 1
    while s[prefix_idx] == s[suffix_idx]:
        # move prefix forward
        while prefix_idx < suffix_idx - 1 and s[prefix_idx + 1] == s[suffix_idx]:
            prefix_idx += 1
        
        # move suffix back
        while suffix_idx > prefix_idx + 1 and s[prefix_idx] == s[suffix_idx - 1]:
            suffix_idx -= 1
        
        if prefix_idx < suffix_idx - 1:
            prefix_idx += 1
            suffix_idx -= 1
        else:
            if prefix_idx == suffix_idx: return 1
            else: return 0
    return suffix_idx - prefix_idx + 1

#
#
#
# TLE
def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
    res = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums) + 1):
            if sum(nums[i:j]) == goal:
                res += 1
    return res

# Number of Good Ways to Split a String LeetCode Medium
# https://leetcode.com/problems/number-of-good-ways-to-split-a-string/description/
# took 16 mins but could have done quicker less minor issues
# TC: O(n), SC: O(n)
def numSplits(self, s: str) -> int:
    # iterate from left-to-right and populate a list of integers representing the total number of distinct characters to the left of each element in s
    total, elem_to_left, chars = 0, [0]*len(s), set()
    for i in range(len(s)):
        elem_to_left[i] = len(chars)
        if s[i] not in chars:
            chars.add(s[i])

    # iterate from the right-to left and compare the total distinct characters to right of each element to the stored number of distincts to the left
    chars.clear()
    for i in range(len(s) - 1, -1, -1):
        if s[i] not in chars:
            chars.add(s[i])
        
        if len(chars) == elem_to_left[i]:
            total += 1
    
    return total

# Product of Array Except Self LeetCode Medium
# https://leetcode.com/problems/product-of-array-except-self/description/
# took about 15 mins
# TC: O(n) (two-pass)
# SC: O(n)
def productExceptSelf(self, nums: List[int]) -> List[int]:
    # using prefix product and siffix product, the without-product of each element
    # will be the product of the prefix prod of the element to the left of it and the suffix prod of the element to the right of it

    # setup prefix product
    prefix_prod = [nums[0]]
    for i in range(1, len(nums)):
        prefix_prod.append(prefix_prod[i-1]*nums[i])
    
    # now iterate backwards and maintain the suffix product as well as populate the product-without for each element
    suffix_prod = 1
    res = [0]*len(nums)
    for i in range(len(nums)-1, -1, -1):
        res[i] = suffix_prod*(prefix_prod[i-1] if i > 0 else 1)
        suffix_prod *= nums[i]
    return res

# Spiral Matrix II LeetCode Medium
# https://leetcode.com/problems/spiral-matrix-ii/description/
# TC: O(n), SC: O(1)
# Given a positive integer n, generate an n x n matrix filled with elements from 1 to n^2 in a spiral order.
def generateMatrix(self, n: int) -> List[List[int]]:
    left, right, top, bottom, val = 0, n, 0, n, 1
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    while top <= bottom and left <= right:
    
        # go across the matrix
        for col in range(left, right):
            matrix[top][col] = val
            val += 1
        top += 1

        # go down the matrix
        for row in range(top, bottom):
            matrix[row][right-1] = val
            val += 1
        right -= 1

        # go left across the matrix
        for col in range(right-1, left-1, -1):
            matrix[bottom-1][col] = val
            val += 1
        bottom -= 1

        # go up the matrix
        for row in range(bottom-1, top-1, -1):
            matrix[row][left] = val
            val += 1
        left += 1
    
    return matrix


# String Encode and Decode NeetCode Medium
# https://neetcode.io/problems/string-encode-and-decode
# https://leetcode.com/problems/encode-and-decode-strings/submissions/1215609322/
# had to watch video but made sense
# TC: O(n), SC: O(n)
# see encode/decode below
def encode(self, strs: List[str]) -> str:
    return "".join([str(len(s)) + "#" + s for s in strs])

def decode(self, s: str) -> List[str]:
    res = []
    i = 0
    while i < len(s):
        curr = ""
        j = i
        while s[j] != "#":
            curr += s[j]
            j += 1
        res.append(s[j+1:j+1+int(curr)])
        i = j + 1 + int(curr)
    return res

# Spiral Matrix LeetCode Meidum
# https://leetcode.com/problems/spiral-matrix/
# TC: O(n), SC: O(1)
# took like 40 but should have been way quicker
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    rows, cols = len(matrix), len(matrix[0])
    ans, target_len = [], rows*cols
    left, right, top, bottom = 0, cols-1, 0, rows-1

    def addToAns(num: int):
        ans.append(num)
        return len(ans) == target_len

    while right >= left and top <= bottom:

        # iterate across the top (from left to right)
        for i in range(left, right+1):
            if addToAns(matrix[top][i]):
                return ans
        top += 1

        # iterate down the right (from top to bottom)
        for i in range(top, bottom+1):
            if addToAns(matrix[i][right]):
                return ans
        right -= 1

        # iterate backwards across the bottom (from right to left)
        for i in range(right, left-1, -1):
            if addToAns(matrix[bottom][i]):
                return ans
        bottom -= 1

        # iterate up the left (from bottom to top)
        for i in range(bottom, top-1, -1):
            if addToAns(matrix[i][left]):
                return ans
        left += 1
    return ans

# Longest Consecutive Sequence LeetCode Medium (used to be a Hard)
# https://leetcode.com/problems/longest-consecutive-sequence/submissions/1208792750/
# TC O(n) SC O(n)
# I glossed over this approach because I assumed that by iterating through all of the items in the sequence, it would be O(n^)
# becuase we would do this for *every* item in the sequence. But this is incorrect, we can simply iterate through all of the items
# of a sequence *starting only at the first item* (we would only do this once for every sequence, starting from the first item, making it O(n), not for EVERY item —— being O(n^2))
def longestConsecutive(self, nums: List[int]) -> int:
    items = set(nums)
    maxx = 0
    for num in nums:
        if num-1 not in items:
            end = num + 1
            while end in items:
                end += 1
            maxx = max(maxx, end - num)
    return maxx if nums else 0

# Trapping Rain Water LeetCode Hard
# https://leetcode.com/problems/trapping-rain-water/description/
# TC: O(n), SC: O(n)
# note: the amount of water stored at any point is min(highest point on left, highest point on right) - current point height
def trap(self, height: List[int]) -> int:
    # iterate forward and get prefix max for every element - the largest value that comes before every element
    prefix_max, maxx = [0]*len(height), 0
    for i in range(len(height)):
        prefix_max[i] = maxx
        maxx = max(prefix_max[i], height[i])
    
    # iterate backwards and maintain the suffix max for every element - the largest value that comes after every element
    # and add the difference between the current point and the lesser of the two: prefix max and suffix max
    total, suffix_max = 0, 0
    for j in range(len(height)-1, -1, -1):
        val = min(prefix_max[j], suffix_max) - height[j]
        total += val if val > 0 else 0
        suffix_max = max(suffix_max, height[j])
    return total

# Find All Duplicates in an Array LeetCode Medium
# https://leetcode.com/problems/find-all-duplicates-in-an-array/?envType=daily-question&envId=2024-03-25
# TC: O(n), SC: O(1)
# approach: since the list values are in range [1..n], iterate through the array and attempt to place every number in it's corresponding index
#           and take the value at that index and put it where we are. Continue to do so with the current new value, unless there is already the same value in it's spot
#           if so, move forward
# this took like 40 mins
def findDuplicates(self, nums: List[int]) -> List[int]:
    n = len(nums)
    i = 0
    while i < n:
        # if the number x isn't equal to it's index AND the number x isn't already in the x index (if there are duplicates), then move this number to that index
        if nums[i] != i+1 and nums[nums[i]-1] != nums[i]:
            temp = nums[nums[i]-1]
            nums[nums[i]-1] = nums[i]
            nums[i] = temp
        else: # else, move forward
            i += 1
    return [nums[i] for i in range(n) if nums[i] != i+1] # return all number in the list that aren't in their corresponding index

# First Missing Positive LeetCode Hard
# https://leetcode.com/problems/first-missing-positive/?envType=daily-question&envId=2024-03-26
# TC: O(n), SC: O(1)
# thought at it for 15 mins and couldn't figure it out. Started NC problem and stopped as soon as he said:
#   "for an input array of length n, the minimum possible value in the worst case is n+1"
# which def made it possible to use a trick, so then did the quesiton in 8 mins - very similar to the two problems above
# NC has a slightly dif approach where instead of swapping the nums and sending them to their corresponding index, he does the following:
# 1. converts all neg numbers to 0
# 2. for all numbers x, where 0 <= x <= n, flip index x to a negative
# 3. then iterate upwards in the indices and return the first index that hasn't bee flipped
# -- same idea but simpler than swapping. My solution is shorter though
def firstMissingPositive(self, nums: List[int]) -> int:
    # for an input array of length n, the minimum possible value in the worst case is n+1

    # iterate through array and add items to their corrsponding index or ignore
    n = len(nums)
    i = 0
    while i < n:
        idx = nums[i] - 1
        num = nums[i]
        if num > 0 and num <= n and nums[idx] != num:
            temp = nums[idx]
            nums[idx] = num
            nums[i] = temp
        else:
            i += 1
    
    for i in range(n):
        if nums[i] != i+1: return i+1
    return i+2

# Palindromic Substrings LeetCode Medium
# Given a string s, return the number of palindromic substrings in it.
# https://leetcode.com/problems/palindromic-substrings/description/
# took like 10 mins
# I also like this solution and should have done something like this - break it into a function and call it twice (second with two item center) https://leetcode.com/problems/palindromic-substrings/solutions/4703811/interview-approach-3-approach-brute-force-expand-middle-dp/
# TC: O(n^2), SC: O(1)
def countSubstrings(self, s: str) -> int:
    palindromes = 0
    for idx, char in enumerate(s):
        left = right = idx
        pal_len = 0
        while left >= 0 and right <= len(s) - 1:
            if s[left] != s[right]:
                break
            palindromes += 1
            left -= 1
            right += 1
        right = idx+1
        if right < len(s) and s[right] == s[idx]:
            palindromes += 1
            left = idx - 1
            right = right + 1
            while left >= 0 and right <= len(s) - 1:
                if s[left] != s[right]:
                    break
                palindromes += 1
                left -= 1
                right += 1
    return palindromes

# Remove Duplicates from Sorted Array II LeetCode Medium
# https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/
# : Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice. The relative order of the elements should be kept the same.
# : Return k after placing the final result in the first k slots of nums.
# TC: O(n), SC: O(1)
# this is actually a good question and struggled for a bit to find a simple way to do it
def removeDuplicates(self, nums: List[int]) -> int:
    
    curr = 1
    curr_occurance = 1

    for i in range(1, len(nums)):
        if nums[i] == nums[i-1]:
            curr_occurance += 1
        else:
            curr_occurance = 1
        
        if curr_occurance <= 2:
            nums[curr] = nums[i]
            curr += 1
    
    return curr