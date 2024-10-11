from types import List

# Minimum Difference Between Highest and Lowest of K Scores LeetCode Medium
# TC: O(nlogn), SC: O(1) aux O(n) total
def minimumDifference(self, nums: List[int], k: int) -> int:
    nums.sort()
    best = float('inf')
    for right in range(k - 1, len(nums)):
        dif = nums[right] - nums[right - k + 1]
        best = min(best, dif)
    return best

# Maximum Points You Can Obtain from Cards LeetCode Medium
# https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/description/
# TC: O(n), SC: O(n)
# also greedy but putting here. Definitely overthought it about some DP stuff and what not. But this is a very simple solution
def maxScore(self, cardPoints: List[int], k: int) -> int:
    if k == len(cardPoints): return sum(cardPoints)
    R = len(cardPoints) - 1
    score = 0; best = 0
    
    # move R backwards all the way to start as if we only take from the right
    for _ in range(k):
        score += cardPoints[R]
        R -= 1

    best = score

    # one at a time, add cards back to the right as if we would instead be taking them from the left
    for L in range(k):
        R += 1
        score += cardPoints[L]
        score -= cardPoints[R]
        best = max(best, score)
    
    return best

# Get Equal Substrings Within Budget LeetCode Medium
# https://leetcode.com/problems/get-equal-substrings-within-budget/description/
# : You are given two strings s and t of the same length and an integer maxCost.
# : You want to change s to t. Changing the ith character of s to ith character of t costs |s[i] - t[i]| (i.e., the absolute difference between the ASCII values of the characters).
# : Return the maximum length of a substring of s that can be changed to be the same as the corresponding substring of t with a cost less than or equal to maxCost. If there is no substring from s that can be changed to its corresponding substring from t, return 0.
# TC: O(n), SC: O(n)
# approach: create list of distances from each character in s to t (or t to s - no wrapping around)
# perform sliding window to find the longest subarray in that list that's total is less than maxCost
def equalSubstring(self, s: str, t: str, maxCost: int) -> int:

    # create list of distances for t[i] to s[i]
    s_distances = []
    for i in range(len(t)):
        s_distances.append(abs(ord(t[i]) - ord(s[i])))
    
    # perform sliding window to find the largest substring
    total = left = 0
    ans = 0
    for right in range(len(s_distances)):
        total += s_distances[right]

        # move left forward if above the maxCost
        while left < right and total > maxCost:
            total -= s_distances[left]
            left += 1

        if total <= maxCost:
            ans = max(ans, right - left + 1)
    return ans

# Minimum Size Subarray Sum LeetCode Medium
# https://leetcode.com/problems/minimum-size-subarray-sum/description/
# Given an array of positive integers nums and a positive integer target, return the minimal length of a subarray whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.
# TC: O(n), SC: O(1)
def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    right = 0
    summ = nums[0]
    min_len = float('inf')
    for left in range(len(nums)):
        while right < len(nums) - 1 and summ < target:
            right += 1
            summ += nums[right]
        if summ >= target:
            min_len = min(min_len, right - left + 1)           
        summ -= nums[left]
    return 0 if min_len == float('inf') else min_len


# Minimum Window Substring LeetCode Hard
# https://leetcode.com/problems/minimum-window-substring/description/
# took over 2 hours because tried DP tabulation at first but kept getting TLE (see invalid-solutions.py)
# after doing window though the solution took about 45 because of debugging issues but probably could have done in like 30
# TC: O(n), SC: O(n)
from collections import Counter
def minWindow(self, s: str, t: str) -> str:
    if s == t: return t
    remaining, p1, p2, minimum = Counter(t), 0, 0, s*2
    if s[0] in remaining: remaining[s[0]] -= 1

    # func to check if dict has any positive values (any remaining chars we need to find)
    def has_remaining(rem):
        for key in remaining: 
            if remaining[key] > 0: return True
        return False

    while p2 < len(s):
        # move right pointer right until we find a valid solution
        if has_remaining(remaining):
            p2 += 1
            if p2 < len(s) and s[p2] in remaining:
                remaining[s[p2]] -= 1
        else:
            # if the left item cannot be be left behind, move the right forward until we find something to replace the item on the left
            if remaining[s[p1]] > 0:
                while s[p2] != s[p1]:
                    p2 += 1
                    if p2 >= len(s): return minimum
                    if s[p2] in remaining: remaining[s[p2]] -= 1

            # now move the left forward as much as possible
            while not has_remaining(remaining) and p1 <= p2:
                minimum = min(minimum, s[p1:p2+1], key=len)
                if s[p1] in remaining: remaining[s[p1]] += 1
                p1 += 1
    return minimum if len(minimum) <= len(s) else ""

# shorter and clearner solution
# TC: O(n), SC: O(n)
# this only beats 18.88%, why?
def minWindow(self, s: str, t: str) -> str:
    t = Counter(t)
    curr = Counter()
    left = 0
    best = ""
    def isValid():
        for key in t:
            if curr[key] < t[key]:
                return False
        return True
        

    for right in range(len(s)):

        # expand right until all characters in the window
        if s[right] in t:
            curr[s[right]] += 1
        valid = isValid()
        if not valid:
            continue

        # shrink the window from the left as much as possible
        while left < right and (s[left] not in t or s[left] in t and curr[s[left]] > t[s[left]]):
            curr[s[left]] -= 1 # doesn't matter if the character isn't in t we reduce it in s
            left += 1
        
        # update the max
        if not best or right - left + 1 < len(best):
            best = s[left:right+1]
    return best

# Longest Substring Without Repeating Characters LeetCode Medium
# https://leetcode.com/problems/longest-substring-without-repeating-characters/
# TC: O(n), SC: O(n)
# took 8:30
# sliding window appraoach
# approach: expand the window while the items inside are unique, then contract the window to allow it to keep expanding forward, keepign track of the maximum width it becomes
def lengthOfLongestSubstring(self, s: str) -> int:
    maxx = 0
    curr = set()
    left, right = 0, 0
    while right < len(s):
        if s[right] in curr:
            # move left forward until the value we are trying to add on the right is found
            while s[left] != s[right]:
                curr.remove(s[left])
                left += 1
            left += 1 
        else:
            curr.add(s[right])
        maxx = max(right - left + 1, maxx)
        right += 1
    return maxx

# Subarray Product Less Than K LeetCode Medium
# Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.
# https://leetcode.com/problems/subarray-product-less-than-k/description/
# TC: O(n), SC: O(1)
# came up with solution in like 5 but had issues with adding subarrays between left/right, so took 15-20
def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    left, prod, res = 0, 1, 0
    # expand the window and increase the product while we can
    for right in range(len(nums)):
        prod = prod*nums[right]

        # if the product is too large shrink the window
        while left <= right and prod >= k:
            prod /= nums[left]
            left += 1

        # add all of the possible subarrays in between, which could be however many spots we could move left forward currently before it reached the right
        res += 1 + right - left
        right += 1
    return res

# todo cleaner approach available by doing 'for r in range(len(s))' rather than 'while right < len(s)'

# Permutation in String LeetCode Medium
# https://leetcode.com/problems/permutation-in-string/description/
# TC: O(n), SC: O(1)? Becuase we reassign s1 to a dict? (and also Counter is dict of max 26 keys (possible letters) regardless, so O(1) anyways)
# took 18 mins
# todo review, didn't see a solution similar to this at all
# todo better approach is FIXED sliding window, comparing the frequencies/counters of the window as we do so. Def shorter/simpler solution as well.
def checkInclusion(self, s1: str, s2: str) -> bool:
    target = len(s1)
    s1 = Counter(s1)
    left = 0
    # iterate moving the right pointer forward (expanding the window)
    for right in range(len(s2)):
        # if the right element is in s1, ensure we haven't used too many of said element
        if s2[right] in s1:

            # move left forward until that element is available again if it isn't
            while s1[s2[right]] == 0:
                if s2[left] in s1:
                    s1[s2[left]] += 1
                left += 1
            
            # check if we have new longest
            if right - left + 1 == target:
                return True

            # decrement availability of this element
            s1[s2[right]] -= 1
        
        # if the right element is not in s1 we cannot expand the window any more, so move left and right to the element after the current right
        else:
            while left <= right:
                if s2[left] in s1:
                    s1[s2[left]] += 1
                left += 1
    return False

# Sliding Window Maximum LeetCode Hard
# https://leetcode.com/problems/sliding-window-maximum/description/
# TC: O(n), SC: O(n)
# took like 40 because unfamilliar with uses of monotonically decreasing queue - had to watch NC
# beats 98%
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:

    # initialize monotonically decreasing deque and initial window
    q = deque()
    start, end = 0, 0
    for end in range(k):
        while q and nums[end] > q[-1]:
            q.pop()
        q.append(nums[end]) 
    
    # iterate forward and update the deque
    res = [q[0]]
    for end in range(k, len(nums)):
        # remove the item being left behind if it is the maximum (if it isn't, it doesn't matter)
        if nums[start] == q[0]:
            q.popleft()
        start += 1

        # add the new item to the queue and update it
        while q and nums[end] > q[-1]:
            q.pop()
        q.append(nums[end])
        res.append(q[0])
    return res

# Length of Longest Subarray With at Most K Frequency LeetCode Medium
# https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/description/?envType=daily-question&envId=2024-03-28
# TC: O(n), SC: O(n)
# took 10 mins - should have been 5
def maxSubarrayLength(self, nums: List[int], k: int) -> int:
    freq = collections.Counter()
    left = max_len = 0
    for right in range(len(nums)):
        freq[nums[right]] += 1 # update count of right pointer item
        # move left inward until it reaches same character so that max_freq is reduced back below k
        while left < right and freq[nums[right]] > k:
            freq[nums[left]] -= 1
            left += 1
        max_len = max(max_len, right-left+1)
    return max_len

# Minimum Swaps to Group All 1's Together LeetCode Medium
# https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together/description/?envType=study-plan-v2&envId=amazon-spring-23-high-frequency
# : Given a binary array data, return the minimum number of swaps required to group all 1’s present in the array together in any place in the array.
# TC: O(n), SC: O(1)
# approach: sliding window. Expand window until number of zeros inside of the window is equal to the number of ones outside of the window.
# simple but a bit tricky. Also note the variable ones / total ones can be combined into one but is more clear when two varaiables are used
# 8 mins before coming up with sliding windiow solution after looking at topic
def minSwaps(self, data: List[int]) -> int:
    total_ones = data.count(1) # total ones in the string
    best = float('inf') # smallest window
    left = 0 
    ones = 0 # number of ones in the window
    zeros = 0 # number of zeros in the window
    for right in range(len(data)):
        if data[right] == 1:
            ones += 1
        else:
            zeros += 1
        
        while zeros == total_ones - ones:
            best = min(best, zeros)
            if data[left] == 1:
                ones -= 1
            else:
                zeros -= 1
            left += 1

    return best if total_ones != 0 else 0