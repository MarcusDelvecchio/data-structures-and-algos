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

# Maximum Gap LeetCode Medium
# https://leetcode.com/problems/maximum-gap/?envType=problem-list-v2&envId=bucket-sort
# TC: O(n*d) where n is the array size and d is the number of digits in the largest number (maximum 10,. so worst case is O(10*n))
# SC: O(n)
# simplify: given an array of nums, return the min difference between two elements if it were to be sorted, without sorting it
# approach: perform radix sort and for every radix (digit place) perform counting sort to count sort the counts of the 10 possible items those digit values can be
#
# had to learn radix sort and counting sort for this
def maximumGap(self, nums: List[int]) -> int:
    
    def countingSort(arr, radix):
        output = [0] * len(arr)
        counts = [0] * 10 # count of all of all possible values for the current digit (radix)

        for num in nums:
            counts[(num // radix) % 10] += 1

        # create the cumulative array
        for i in range(1, 10):
            counts[i] += counts[i-1]

        # fill the output array
        i = len(arr) - 1
        while i >= 0:
            index = arr[i] // radix
            output[counts[index % 10] - 1] = arr[i]
            counts[index % 10] -= 1
            i -= 1
        
        for i in range(len(arr)):
            arr[i] = output[i]
    
    max_val = max(nums)
    radix = 1
    while max_val // radix > 0:
        countingSort(nums, radix)
        radix *= 10
    
    max_dif = 0
    for i in range(1, len(nums)):
        max_dif = max(max_dif, abs(nums[i] - nums[i-1]))
    return max_dif