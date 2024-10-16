from types import List

# Binary Search LeetCode Easy
# https://leetcode.com/problems/binary-search
# TC: O(logn), SC: O(1)
def search(self, nums: List[int], target: int) -> int:
    left, right = 0, len(nums)-1
    while left <= right:
        mid = (left+right)//2
        if target > nums[mid]:
            left = mid+1
        elif target < nums[mid]:
            right = mid-1
        else:
            return mid
    return -1

# Find First and Last Position of Element in Sorted Array LeetCode Medium
# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/
# TC: O(n), SC: O(1)
# approach: perform binary search twice, once to find the earliest occurrence of the target element, and a second time to find the latest
# the two searches differ simply in what they do when they find an occurrence of the target: either shifting left to find an earlier one or right to find a later one
def searchRange(self, nums: List[int], target: int) -> List[int]:
    left, right = 0, len(nums) - 1
    
    # find earliest occurrence
    earliest = float('inf')
    while left <= right:
        mid = (left+right)//2
        if nums[mid] == target:
            earliest = min(earliest, mid)
            right = mid - 1 # we want to go earlier
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1

    # find earliest occurrence
    left, right = 0, len(nums) - 1
    latest = -1
    while left <= right:
        mid = (left+right)//2
        if nums[mid] == target:
            latest = max(latest, mid)
            left = mid + 1 # we want to go later
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return [earliest, latest] if latest != -1 else [-1, -1]

# H-Index II LeetCode Medium
# actually hard than I expected, few annoying edge cases
# : Given an array of integers citations where citations[i] is the number of citations a researcher received for their ith paper and citations is sorted in ascending order, return the researcher's h-index.
# : The h-index is defined as the maximum value of h such that the given researcher has published at least h papers that have each been cited at least h times.
# https://leetcode.com/problems/h-index-ii/
# edge case skew 7/10
# edge cases: zeros, empty list, the value returned will not always be one of the citation values ex: [11, 15]
# approach: we want to find the largest element i, where citations[i] >= (i + 1)
def hIndex(self, citations: List[int]) -> int:
    if len(citations) == 1: return int(bool(citations[0]))
    L = 0; R = len(citations) - 1; h = 0

    def isValidH(idx):
        count_with_score = len(citations) - idx
        return count_with_score >= citations[idx]

    while L <= R:
        mid = (L + R)//2
        if isValidH(mid):
            h = max(h, citations[mid])
            L = mid + 1
        else:
            h = max(h, len(citations) - mid)
            R = mid - 1
    return h

# Search a 2D Matrix LeetCode Medium
# https://leetcode.com/problems/search-a-2d-matrix
# : You are given an m x n integer matrix matrix with the following two properties:
# 1. Each row is sorted in non-decreasing order.
# 2. The first integer of each row is greater than the last integer of the previous row.
# Given an integer target, return true if target is in matrix or false otherwise.
# took 17 mins becuase off-by-one issues and dumb mistakes
# approach: approach the problem as a singular list where R and L point to indices in a normal 1D list, but every time we access a value, convert that value to the actual row/col value in the matrix
# TC: O(logn), SC: O(1)
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    m, n = len(matrix[0]), len(matrix)
    
    def getValue(idx): # given an index in the 1D array, convert it to the row/col index in the matrix and return the value
        return matrix[idx//m][idx%m]
    
    L, R = 0, m*n-1
    while L <= R:
        mid = getValue((R+L)//2) # get the value at the midpoint and update the L/R pointer based on result
        if target < mid:
            R = (R+L)//2-1
        elif target > mid:
            L = (R+L)//2+1
        else:
            return True
    return False

# Koko Eating Bananas LeetCode Medium
# 'Allocated Books' problem type
# : Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.
# : Return the minimum integer k such that she can eat all the bananas within h hours.
# https://leetcode.com/problems/koko-eating-bananas/description/
# TC: O(nlogn) SC: O(1)
def minEatingSpeed(self, piles: List[int], h: int) -> int:
    minn, maxx = 1, 0
    for pile in piles:
        maxx = max(maxx, pile)
        # note: made mistake of thinking a should be doing maxx = min(maxx, pile) here as well
        #       but this was incorrect, he can still eat slower than the speed required by minimum pile in 1h

    best = float('inf')
    while minn <= maxx:
        speed = (minn+maxx)//2
        total = 0
        for p in piles:
            total += p//speed if p%speed == 0 else p//speed+1
            if total > h: # increase the speed
                minn = speed + 1
                break
        if total <= h: # reduce the speed
            maxx = speed - 1
            best = min(best, speed) # save this speed if it is lesser than the best
    return best

# Find Minimum in Rotated Sorted Array LeetCode Medium
# observations: 1. a 'rotated' array will still have two sorted subarrays (1 if it is rotated by n)
# 2. so given an L, an R, and a mid, we can determine if the mid is in the lower or higher subarray, and move the L or R accordingly
# https://leetcode.com/problems/find-minimum-in-rotated-sorted-array
# TC: O(n), SC: O(1)
# took 17 mins
def findMin(self, nums: List[int]) -> int:
    # do binary search while min>maxx to find correct subarray
    L, R = 0, len(nums) - 1
    minn = nums[0]
    while L <= R:
        mid = (L+R)//2
        minn = min(minn, nums[mid])
        if nums[mid] > nums[R]:
            L = mid + 1
        else: # nums[mid] < nums[R] - move right down
            R = mid - 1
    return minn

# Find Peak Element LeetCode Medium
# https://leetcode.com/problems/find-peak-element/description/
# : A peak element is an element that is strictly greater than its neighbors.: 
# : Given a 0-indexed integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.
# : You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.
# : You must write an algorithm that runs in O(log n) time.
# TC: O(logn), SC: O(1)
# great question interesting concepts
def findPeakElement(self, nums: List[int]) -> int:
    ''' for any element there are 3 cases:
        1. the element on the left is greater: if so, there must be a peak on the left somewhere (could be that element)
        2. the element on the right is greater: if so, there must be a peak on the right somewhere (could be that element)
        3. this element is a peak
    '''
    
    left, right = 0, len(nums)-1
    while left <= right:
        mid_idx = (left+right)//2

        # shift right if right is greater
        if mid_idx + 1 < len(nums) and nums[mid_idx+1] > nums[mid_idx]:
            left = mid_idx + 1

        # shift right if right is greater
        elif mid_idx - 1 > -1 and nums[mid_idx-1] > nums[mid_idx]:
            right = mid_idx - 1
        
        # else is a peak
        else:
            return mid_idx
        
# Search in Rotated Sorted Array LeetCode Medium
# : Given a rotated array, return the index of the given target element, if it is included, else -1 (in logarithmic time)
# https://leetcode.com/problems/search-in-rotated-sorted-array/description/
# TC: O(logn), SC: O(1)
# observations: 1. the 'rotated' array will still have two sorted subarrays (1 if it is rotated by n)
# imagine the rotated array as a graph: see https://miro.medium.com/v2/resize:fit:517/1*1yRrcA1ge6AhezTwE3qjlw.png
# just break the quesiton into discrete cases
# 2 cases situations (midpoint in upper vs lower) x 4 subcases cases each (target greater / less than mid and is/is not in same portion)
def search(self, nums: List[int], target: int) -> int:
    L, R = 0, len(nums) -1

    while L <= R:
        mid = (L + R) // 2
        if nums[mid] == target: return mid
        elif nums[R] == target: return R
        elif nums[L] == target: return L
        elif target < nums[mid]:
            if nums[mid] > nums[L]:
                if target < nums[L]:
                    L = mid + 1
                else:
                    R = mid - 1 
            else:
                R = mid - 1          
        else:
            if nums[mid] > nums[L]:
                L = mid + 1
            else:
                if target > nums[L]:
                    R = mid - 1
                else:
                    L = mid + 1
    return -1