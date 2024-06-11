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

# Relative Sort Array LeetCode Medium
# https://leetcode.com/problems/relative-sort-array/description/
# : Given two arrays arr1 and arr2, the elements of arr2 are distinct, and all elements in arr2 are also in arr1.
# : Sort the elements of arr1 such that the relative ordering of items in arr1 are the same as in arr2. Elements that do not appear in arr2 should be placed at the end of arr1 in ascending order.
# TC: O(n), SC: O(n)
def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
    res = []
    arr_1_counts = collections.Counter(arr1)
    in_arr2 = collections.Counter(arr2)
    not_in_arr2 = []
    for num in arr2:
        if num in arr_1_counts:
            res.extend([num]*arr_1_counts[num])                

    for num in arr1:
        if num not in in_arr2:
            not_in_arr2.append(num)

    return res + sorted(not_in_arr2)