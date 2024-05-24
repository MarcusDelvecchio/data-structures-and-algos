from types import List
from collections import defaultdict, Counter

# Sum of All Subset XOR Totals LeetCode Easy
# https://leetcode.com/problems/sum-of-all-subset-xor-totals/description/
# TC: O(2^n), SC: O(n)
# : The XOR total of an array is defined as the bitwise XOR of all its elements, or 0 if the array is empty.
# : Given an array nums, return the sum of all XOR totals for every subset of nums. Note that subsets with the same elements should be counted multiple times.
def subsetXORSum(self, nums: List[int]) -> int:
    if not nums: return 0
    
    def solve(idx, total):
        if idx == len(nums): return total
        with_curr = solve(idx + 1, total^nums[idx])
        without_curr = solve(idx + 1, total)
        return with_curr + without_curr

    return solve(0, 0)

# Subsets LeetCode Medium
# https://leetcode.com/problems/subsets/
# : Given an integer array nums of unique elements, return all possible 
# : subsets (the power set). The solution set must not contain duplicate subsets. Return the solution in any order.
# took 13 mins nice. Ran second try. I just forgot a 'def' beside the function definition
# interesting problem. Was just having issues thinking about the most efficient way to handle items being unique etc. and preventing duplicates
# and after reviewing the solutions realized that again, the idx prop prevents duplicates etc so see better and new solution below this one. But left here for reference
def subsets(self, nums: List[int]) -> List[List[int]]:
    res = {(): True}

    def sets(l, idx, curr, curr_hash):
        if l > len(nums):
            return
        
        for i in range(idx, len(nums)):
            if nums[i] not in curr_hash:
                curr.append(nums[i])
                res[tuple(curr)] = True
                curr_hash[nums[i]] = True
                sets(l+1, i+1, curr, curr_hash)
                del curr_hash[nums[i]]
                curr.pop()
    sets(0, 0, [], {})
    return list(res.keys())

# Permutations LeetCode Medium
# https://leetcode.com/problems/permutations/description/
# : Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
# took 7:30
# this one was easier than the ones above because due to the nature of permuations, you know that the solution will (can) simpy be O(n^2)
# because all the permuations should have the same length as the nums input and any num in nums can appear anywhere, so we have to consider all cases
# (no need to account for same-combo-different-order as we do in combinations)
def permute(self, nums: List[int]) -> List[List[int]]:
    res, used = [], {num: False for num in nums}

    def explore(path):
        if len(path) == len(nums):
            res.append(path)
            return
        
        for num in nums:
            if not used[num]:
                used[num] = True
                explore(path + [num])
                used[num] = False
    explore([])
    return res

# Palindrome Partitioning LeetCode Medium
# https://leetcode.com/problems/palindrome-partitioning/description/
# : Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
# TC: O(n * 2^n)
# SC: O(2^n) because there are 2^N possible partitions here
def partition(self, s: str) -> List[List[str]]:

    def isPalindrome(string):
        left, right = 0, len(string)-1
        while left <= right:
            if string[left] != string[right]:
                return False
            left += 1
            right -= 1
        return True

    ans = []
    def split(idx, partitions, curr_partition):
        if idx == len(s):
            if not curr_partition:
                ans.append(partitions)
            return
        curr_partition += s[idx]

        # consider continuing this partition - using the next char as part of this partition
        split(idx + 1, partitions, curr_partition)
        
        # consider ending this partition and starting a new one - using the next character as a new partition
        # note we should not end this partition unless it is a palindrome
        if isPalindrome(curr_partition):
            split(idx + 1, partitions + [curr_partition], "")

    split(0, [], "")
    return ans

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

# TC O(2^n) SC O(n)
# not as optimial as you can get it
# NOTE you can use a O(n^2) solution by calculating the difference between all elements
def beautifulSubsets(self, nums: List[int], k: int) -> int:
    nums.sort()

    def subsets_max_dif_k(idx, included):
        if idx == len(nums):
            return 1 if included else 0
        with_ = without_ = 0

        if (nums[idx] - k) not in included:
            included[nums[idx]] += 1
            with_ = subsets_max_dif_k(idx + 1, included)
            included[nums[idx]] -= 1
            if included[nums[idx]] == 0:
                del included[nums[idx]]
        
        without_ = subsets_max_dif_k(idx + 1, included)

        return with_ + without_

    return subsets_max_dif_k(0, collections.Counter())