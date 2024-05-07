from types import List

# Sort Colors LeetCode Medium
# https://leetcode.com/problems/sort-colors/description/
# TC: O(n), SC: O(1)
# sort an array of integers containins zero or more 1s, 2s and 3s, so that all 1s come before all 2s and all 2s before all 3s
def sortColors(self, nums: List[int]) -> None:
    # move all 2s to the end
    k = len(nums)
    for i in range(len(nums)):
        if i == k: break
        if nums[i] != 2: continue
        for j in range(k-1, i, -1):
            if nums[j] != 2:
                temp = nums[j]
                nums[j] = nums[i]
                nums[i] = temp
                k = j
                break
    
    # move all 1s to the end (before 2s)
    end = k
    for i in range(end):
        if i == k: break
        if nums[i] != 1: continue
        for j in range(k-1, i, -1):
            if nums[j] != 1 and nums[j] != 2:
                temp = nums[j]
                nums[j] = nums[i]
                nums[i] = temp
                k = j
                break