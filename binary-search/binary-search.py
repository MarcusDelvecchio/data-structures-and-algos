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

# Search a 2D Matrix LeetCode Medium
# https://leetcode.com/problems/search-a-2d-matrix
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