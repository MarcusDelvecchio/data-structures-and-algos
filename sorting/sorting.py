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
# : Given an integer array nums, return the maximum difference between two successive elements in its sorted form. If the array contains less than two elements, return 0.
# : You must write an algorithm that runs in linear time and uses linear extra space.
# https://leetcode.com/problems/maximum-gap/?envType=problem-list-v2&envId=bucket-sort
# TC: O(n*d) where n is the array size and d is the number of digits in the largest number (maximum 10,. so worst case is O(10*n))
# SC: O(n)
# simplify: given an array of nums, return the min difference between two elements if it were to be sorted, without sorting it
# approach: perform radix sort and for every radix (digit place) perform counting sort to count sort the counts of the 10 possible items those digit values can be
# had to learn radix sort and counting sort for this
def maximumGap(self, nums: List[int]) -> int:
    
    # countingSort: performs counting sort for the current digit (radix) place
    def countingSort(arr, radix):
        output = [0] * len(arr) # output array to store the sorted result for this digit place
        counts = [0] * 10 # counts array for each possible digit (0-9) at the current radix place

        # count the occurrences of digits
        for num in nums:
            counts[(num // radix) % 10] += 1

        # modifying counts array to store cumulative counts (see radix algo)
        for i in range(1, 10):
            counts[i] += counts[i-1]

        # backfilling the output array by placing elements in their correct positions using the cumulative counts
        i = len(arr) - 1
        while i >= 0:
            index = arr[i] // radix  # isolate the digit at the current radix place
            output[counts[index % 10] - 1] = arr[i]  # place it in the sorted output
            counts[index % 10] -= 1  # reduce the count of the used digit
            i -= 1
        
        # copy the sorted output back to the original array
        for i in range(len(arr)):
            arr[i] = output[i]
    
    # if the array contains fewer than 2 elements, no gap can be computed
    if len(nums) < 2:
        return 0
    
    # find the maximum number in the array to determine the number of digits (radix places) needed
    max_val = max(nums)
    
    radix = 1  # start with the least significant digit (ones place)
    
    # continue counting sort for each digit place (radix), from least significant to most significant
    while max_val // radix > 0:
        countingSort(nums, radix)
        radix *= 10  # move to the next more significant digit place
    
    # calculate the maximum gap by comparing adjacent elements in the sorted array
    max_dif = 0
    for i in range(1, len(nums)):
        max_dif = max(max_dif, abs(nums[i] - nums[i-1]))
    
    return max_dif