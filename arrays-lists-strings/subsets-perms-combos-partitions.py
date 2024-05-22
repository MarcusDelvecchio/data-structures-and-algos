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