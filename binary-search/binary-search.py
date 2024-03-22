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