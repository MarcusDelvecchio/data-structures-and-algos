from types import List
from collections import Counter

# Maximum Length Substring With Two Occurrences LeetCode Easy
# https://leetcode.com/problems/maximum-length-substring-with-two-occurrences/submissions/1425701766/
# TC: O(n), SC: O(n)
def maximumLengthSubstring(self, s: str) -> int:
    counter = Counter()
    L = 0
    best = 0
    for R in range(len(s)):
        counter[s[R]] += 1
        while counter[s[R]] > 2:
            counter[s[L]] -= 1
            L += 1
        best = max(best, R-L+1)
    return best

# Fruit Into Baskets LeetCode Medium (Easy)
# https://leetcode.com/problems/fruit-into-baskets/description/
# simplified: "given an array of integers, find the longest subarray containing only two unique numbers"
# 6 mins and submitted first try. This is easier than some of the other easy questions in this file
# TC: O(n), SC: (1)
def totalFruit(self, fruits: List[int]) -> int:
    nums = Counter()
    largest = L = 0
    for R in range(len(fruits)):
        nums[fruits[R]] += 1
        while len(nums) > 2:
            nums[fruits[L]] -= 1
            if not nums[fruits[L]]:
                del nums[fruits[L]]
            L += 1
        largest = max(largest, R-L+1)
    return largest

# Defuse the Bomb LeetCode "Easy"
# https://leetcode.com/problems/defuse-the-bomb/submissions/1425952036/
# off-by-one hell
# TC: O(n), SC: O(n)
# beats 100% TC
def decrypt(self, code: List[int], k: int) -> List[int]:
    result = [0 for _ in code]
    if k > 0:
        total = sum(code[1:k+1])
        for R in range(len(code)):
            result[R] = total
            if R < len(code) - 1:
                total = total - code[R+1] + code[(R+k+1) % len(code)]
    elif k < 0:
        total = sum(code[len(code)+k:])
        for R in range(len(code)):
            result[R] = total
            total = total - code[R+k] + code[R]
    return result

# Shortest Subarray With OR at Least K I LeetCode Medium
# https://leetcode.com/problems/shortest-subarray-with-or-at-least-k-i/
# observations: as we compound OR values in an array, bits tend towards 1
# however, when we want to close our window, we cannot simply remove the bits that the leftmost value contributed to the OR
# because other values in the window could have contributed the same bits. So we must use a counter to keep track.
# TC: O(n), SC: O(n)
# NOTE that the max of nums[i] is 50 (from constraints), so our space complexity will actually be O(1) becuase there will be at most 6 bits in our counter
# NOTE also that there is a further optimization where instead of counting the total count of each bit usage, we can just keep the last position of each bit and then
# when we remove an item on the left, for each of it's bit, check if they are the last ocurrence of the bit and if so remove them.
# see this solution https://leetcode.com/problems/shortest-subarray-with-or-at-least-k-ii/solutions/4947412/python3-sliding-window-simple/
def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
    shortest = float('inf')
    bitCounts = Counter()
    L = OR = 0
    for R in range(len(nums)):
        OR |= nums[R]

        # update the bit counts
        binaryR = bin(nums[R])[2:]
        for i in range(len(binaryR)):
            idx = -(i+1)
            if binaryR[idx] == "1":
                bitCounts[i] += 1

        while L <= R and OR >= k:
            shortest = min(shortest, R-L+1)
            binaryL = bin(nums[L])[2:]
            for i in range(len(binaryL)):
                idx = -(i+1)
                if binaryL[idx] == "1":
                    bitCounts[i] -= 1
                    if bitCounts[i] == 0:
                        val = int(math.pow(2, i))
                        OR -= val
            L += 1

    return shortest if shortest != float('inf') else -1

# Minimum Difference Between Highest and Lowest of K Scores LeetCode Medium
# TC: O(nlogn), SC: O(1) aux O(n) total
def minimumDifference(self, nums: List[int], k: int) -> int:
    nums.sort()
    best = float('inf')
    for right in range(k - 1, len(nums)):
        dif = nums[right] - nums[right - k + 1]
        best = min(best, dif)
    return best

# Contains Duplicate II LeetCode Easy
# Given an integer array nums and an integer k, return true if there are two distinct 
# indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.
# simplify: given an array of nums, determine if the same two elements exist within a window of size k
# https://leetcode.com/problems/contains-duplicate-ii/description/
# TC: O(n), SC: O(n)
# more like a 'sliding forward window'
def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    indices = collections.defaultdict(list)
    for idx, num in enumerate(nums):
        if num in indices and abs(indices[num][-1] - idx) <= k:
            return True
        indices[num].append(idx)
    return False

# Repeated DNA Sequences LeetCode Medium
# https://leetcode.com/problems/repeated-dna-sequences/description/
# TC: O(n), SC: O(n)
def findRepeatedDnaSequences(self, s: str) -> List[str]:
    if len(s) < 10: return []
    counts = collections.defaultdict(int)
    ans = []
    for L in range(len(s)-9):
        counts[s[L:L+10]] += 1
        if counts[s[L:L+10]] > 1:
            ans.append(s[L:L+10])
    return set(ans)

# Arithmetic Slices LeetCode Medium
# https://leetcode.com/problems/arithmetic-slices/description/
# TC: O(n), SC: O(n)
# : An integer array is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.
# : Given an integer array nums, return the number of arithmetic subarrays of nums.
def numberOfArithmeticSlices(self, nums: List[int]) -> int:
    count = L = 0
    for R in range(len(nums)):
        if R < 2 or R - L + 1 < 3: continue
        if nums[R] - nums[R-1] != nums[R-1] - nums[R-2]:
            L = R - 1

        # calculate number of possible subarrays from the size of the streak.
        count += R - L + 1 - 2
    return count

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
# (simpler below)
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

# Minimum Size Subarray Sum LeetCode Medium
# 3 mins but doing a bunch of similar problems rn
def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    L = 0
    smallest = float('inf')
    for R in range(len(nums)):
        target -= nums[R]
        while target <= 0:
            smallest = min(smallest, R-L+1)
            target += nums[L]
            L += 1
    return smallest if smallest != float('inf') else 0


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

# Find All Anagrams in a String LeetCode Medium
# pretty hard for a medium, didn't see any solutions as short as mine
# but took a while to get it and cover the cases
# https://leetcode.com/problems/find-all-anagrams-in-a-string/description/
# TC: O(n), SC: O(n)
# : Given two strings s and p, return an array of all the start indices of p's anagrams in s. You may return the answer in any order.
# solution: https://leetcode.com/problems/find-all-anagrams-in-a-string/solutions/5929526/python-simple-sliding-window-15-lines-o-n-beats-90/
def findAnagrams(self, s: str, p: str) -> List[int]:
    pChars = set([c for c in p]) # to reference the initial chars in p
    counts = Counter(p) # counter with the number of each char we still need
    ans = []
    L = 0
    for R in range(len(s)):
        # shrink the window while it is invalid (R not in target or already used)
        while not counts[s[R]] and L <= R:
            if s[L] in pChars:
                counts[s[L]] += 1
            L += 1
        
        # account for use of R
        if s[R] in pChars:
            counts[s[R]] -= 1
        
        # if not more items needed account for subarray
        if sum(counts.values()) == 0:
            ans.append(L)
            counts[s[L]] += 1
            L += 1
    return ans

# Maximize the Confusion of an Exam LeetCode Medium
# https://leetcode.com/problems/maximize-the-confusion-of-an-exam/description/
# TC: O(n), SC: O(1)
# given an array answerKey and where answerKey[i] is either 'T' or 'F',
# we want to maximize the number of consecutive Ts OR Fs (whichever we can produce a larger consecutive subarray of)
# when only being able to change K answers in answerKey
#
# approach: like a 'doubule sliding window' where we track two possible windows and counts at the same time
def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
    False_L = False_R = True_L = True_R = True_Count = False_Count = largest = 0
    
    while True_R < len(answerKey) and False_R < len(answerKey):
        if True_R < len(answerKey) and answerKey[True_R] == "T": True_Count += 1
        if False_R < len(answerKey) and answerKey[False_R] == "F": False_Count += 1

        while True_Count > k:
            if answerKey[True_L] == "T":
                True_Count -= 1
            True_L += 1

        while False_Count > k:
            if answerKey[False_L] == "F":
                False_Count -= 1
            False_L += 1

        largest = max(largest, True_R - True_L + 1, False_R - False_L + 1)
        True_R = True_R + 1
        False_R = False_R + 1
    
    return largest 

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

# Count Substrings That Satisfy K-Constraint I LeetCode Easy
# more like a medium todo review
# https://leetcode.com/problems/count-substrings-that-satisfy-k-constraint-i/description/
# : You are given a binary string s and an integer k.
# : A binary string satisfies the k-constraint if either of the following conditions holds:
# : The number of 0's in the string is at most k.
# : The number of 1's in the string is at most k.
# : Return an integer denoting the number of substrings of s that satisfy the k-constraint.
# TC: O(n), SC: O(n)
# be careful with the 'substring' aspect of this question
# I initially thought when the R was moved all the way, we had to add all possible substrings
# within L and R, but this is not the case
def countKConstraintSubstrings(self, s: str, k: int) -> int:
    count = zeros = ones = L = 0
    prevStop = None
    for R in range(len(s)):
        if s[R] == "0": zeros += 1
        else: ones += 1

        while zeros > k and ones > k:
            if s[L] == "0": zeros -= 1
            else: ones -= 1
            L += 1

        count += R - L + 1
    return count


# simplify: given an array of elements, return true if within a window size w, there exists two elements with k or less absolute difference
# approach 1: sliding window, in every window compare every element with one another to search for abs.
# O(n*k^2)
# TLE
# TODO

# approach 2: sliding window, in every window, sort the elements and look for difference less than abs.
# O(n*klogk)
# TLE
def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    end = min(indexDiff, len(nums)-1)
    while end < len(nums):
        start = max(0, end-indexDiff)
        sorted_window = sorted(nums[start:end+1])
        for idx in range(len(sorted_window)-1):
            if abs(sorted_window[idx+1] - sorted_window[idx]) <= valueDiff:
                return True
        end += 1
    return False

# approach 3 (optimization): instead of re-sorting every time, when we slide the window over by 1, remove the element on the left and add the one on the right. TC: O(klogk + nlogk)
# not implemented, see bucket sort solution below

# this works but does not account for window size (indexDiff)
def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    M = max(nums) + 1
    buckets = [False for _ in range(len(nums))]
    for num in nums:
        bucket = math.floor(num * len(nums) / M)
        if buckets[bucket]:
            return True
        buckets[bucket] = True
    return False

# Contains Duplicate III LeetCode Hard
# bucket sort / bucketing approach
# each bucket represents a range of values. we then iterate over the array and check if there's a nearby element in the same or adjacent buckets that satisfies the condition.
# https://leetcode.com/problems/contains-duplicate-iii/description/
# simplified: given an array of elements, return true if within a window size w, there exists two elements with k or less absolute difference
# TC: O(n), SC: O(n)
# note: had difficulty determining the formula for the bucket divisor
# in the bucket sort wiki it says that bucketdivisor (M) is max(nums) + 1, but here we do it differently
# "By bucketing numbers properly, this problem becomes almost identical to Contains Duplicate II except that numbers in adjacent buckets need to be check as well."
def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
    bucketDivisor = valueDiff + 1 # determine the bucket index for each elemment. Why is this not max(nums) + 1 
    buckets = {}
    for idx, num in enumerate(nums):
        bucket = num // bucketDivisor # math.floor(num * len(nums) / bucketDivisor)
        if bucket in buckets:
            return True

        # add R to it's bucket and ensure there is not already a value there
        if bucket+1 in buckets and abs(buckets[bucket+1] - num) <= valueDiff:
            return True
        if bucket-1 in buckets and abs(buckets[bucket-1] - num) <= valueDiff:
            return True
        buckets[bucket] = num
            
        # if needed, remove the element at L from it's bucket before moving R forward
        if idx >= indexDiff:
            L_bucket = nums[idx - indexDiff] // bucketDivisor # math.floor(nums[R - indexDiff] * len(nums) / bucketDivisor)
            del buckets[L_bucket]
    return False    