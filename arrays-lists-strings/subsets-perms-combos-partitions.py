from types import List
from collections import defaultdict, Counter

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

# TC O(2^n) SC O(n)
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