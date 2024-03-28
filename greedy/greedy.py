
# Maximum 69 Number LeetCode Easy
# https://leetcode.com/problems/maximum-69-number/
# took 2 mins
def maximum69Number (self, num: int) -> int:
    # remove the first occurance of 6?
    num = str(num)
    for i in range(len(num)):
        if num[i] == "6": return int(num[:i] + "9" + num[i+1:])
    return int(num)

# Minimum Sum of Four Digit Number After Splitting Digits LeetCode Easy
# Given a number, take all of it's digits and compase 2 new numbers with the smallest possible sum
# TC: O(1), would be O(nlogn) but array is always size 4
def minimumSum(self, num: int) -> int:
    nums = [c for c in str(num)]
    nums.sort()
    return int(nums[0] + nums[3]) + int(nums[1] + nums[2])

# Split a String in Balanced Strings LeetCode Easy
# 
def balancedStringSplit(self, s: str) -> int:
    res, count = 0, 0
    for i in range(len(s)):
        if s[i] == "L": count -= 1
        elif s[i] == "R": count += 1
        if count == 0: res += 1
    return res

# Maximum Units on a Truck LeetCode Easy
# TC: O(nlogn)
# cleaner solution below
def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
    boxTypes.sort(key= lambda x: x[1], reverse=True)
    boxes, total, curBox = 0, 0, 0
    while curBox < len(boxTypes) and boxes < truckSize:
        while boxTypes[curBox][0] > 0 and boxes < truckSize:
            total += boxTypes[curBox][1]
            boxTypes[curBox][0] -= 1
            boxes += 1
        curBox += 1
    return total

# TC: O(nlogn)
def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
    total = 0
    boxTypes.sort(key= lambda x: x[1], reverse=True)
    for boxes, units in boxTypes:
        while boxes and truckSize > 0:
            total += units
            boxes -= 1
            truckSize -= 1
    return total

# Longest Palindrome LeetCode Easy
#
# TC: O(n), SC: O(1) -> space complexity is actually O(n) but since characters can only be A-Z/a-z is O(1)
def longestPalindrome(self, s: str) -> int:
    counts, ans, hasCenter = collections.Counter(s), 0, False
    for c in counts.values():
        if c%2 == 1: hasCenter = True # if any char count has an odd number we can have a center value
        ans += c//2
    return ans*2 + (1 if hasCenter else 0)

# Jump Game LeetCode Medium
# https://leetcode.com/problems/jump-game/description/
# see DP solution in DP - but it is less efficient
# T: O(n), SC: O(1)
def canJump(self, nums: List[int]) -> bool:
    # approach: keep track of a window of where we can jump to and keep moving left forward by one as we expand the right window as long as we can
    left, right = 0, 0
    while left < len(nums) and left <= right:
        right = max(right, left+nums[left])
        left += 1
    return left > len(nums)-1

# Jump Game II LeetCode Medium
# https://leetcode.com/problems/jump-game-ii/description/
# took 20 mins because edge cases but came up with solution in like 5...
# TC: O(n), SC: O(n)
# approach: feels like a bad explanation and don't have time: keep a left and right pointer. The right represents the current max we can get to in the current number of jumps. We continuously
# move an 'inc' index foward and when the left reaches the inc index, the number of jumps increase
def jump(self, nums: List[int]) -> int:
    left, right, jumps = 0, 0, 0 # number of jumps
    inc = 0 # when left gets to inc, the number of jumps should be increased
    while left < len(nums) - 1:
        right = max(right, left+nums[left])
        if left == inc:
            jumps += 1
            inc = right
        left += 1
    return jumps

# Gas Station LeetCode Medium
# TC: O(n), SC: O(n) but SC can eaily be reduced to O(1)
# took like 30 and had to watch video
# this is pretty hard
# I had the idea of the net costs too
# todo this should be reviewed
def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
    nets = [gas[i]-cost[i] for i in range(len(gas))]
    tank = idx = 0
    if sum(nets) < 0: return -1
    for i in range(len(gas)):
        tank += nets[i]
        if tank < 0:
            tank = 0
            idx = i+1
    return idx

# Largest Number LeetCode Medium
# https://leetcode.com/problems/largest-number/description/
# TC: O(n), SC: O(n)
# custom sorting function is simple solution
class LargerNumKey(str):
    def __lt__(x, y):
        # Compare x+y with y+x in reverse order to get descending order
        return x+y > y+x
        
class Solution:
    def largestNumber(self, nums: List[int]) -> str:

        def compare(x, y):
            return int(x+y) < int(y+x)

        # convert to strings
        nums = [str(num) for num in nums]

        # sort items with custom sort to determine which item should come fist
        nums.sort(key=LargerNumKey)

        # combine all of the numbers
        maximum = "".join(nums)

        # return zero if the first value is "0" for any strings such as "00"
        return "0" if maximum[0] == "0" else maximum


# 
# https://leetcode.com/problems/increasing-triplet-sIncreasing Triplet Subsequence LeetCode Mediumubsequence/description/
# took 18 mins but like 5 after realizing approach
# hint/approach: for any/every element, we want to know if ANY value to the left of it is less than it and ANY value to the right is more than it
# TC: O(n), SC: O(n)
# way more clever solution below
def increasingTriplet(self, nums: List[int]) -> bool:
    largest_to_right = nums.copy()
    largest = 0
    # go right to left getting the largest value to the right of each element
    for i in range(len(nums)-2, -1, -1):
        largest = max(largest, nums[i+1])
        largest_to_right[i] = largest

    # go left to right getting the smallest value to the right of element
    smallest_to_left = nums.copy()
    smallest = float("inf")
    for i in range(1, len(nums)):
        smallest = min(smallest, nums[i-1])
        smallest_to_left[i] = smallest

    # for each element, check if the smallest on the left is less than that element and largest on right is greater
    for i in range(len(nums)):
        if smallest_to_left[i] < nums[i] < largest_to_right[i]:
            return True
    return False

# mind blowing alternative better solution but probably impossible to come up in an interview
# This is a special case of LIS. (lonest increasing subsequence)
# LIS can be solved with O(N log M) where M is the length of sequences (generally M is N). In this question, we can set M as 3 thus the problem can be solved with O(N) with the general LIS approach.
def increasingTriplet(self, nums: List[int]) -> bool:
    first = second = float('inf') 
    for n in nums: 
        if n <= first: 
            first = n
        elif n <= second:
            second = n
        else:
            return True
    return False

# Make Array Zero by Subtracting Equal Amounts LeetCode Easy
# https://leetcode.com/problems/make-array-zero-by-subtracting-equal-amounts/description/
# TC: O(n), SC: O(n)
# way simpler solution below
# took 10 mins this is too slow
def minimumOperations(self, nums: List[int]) -> int:
    # sort the items
    nums.sort()
    largest = nums[-1]
    ops = 0
    prev = -1
    reduction = 0
    if sum(nums) == 0: return 0

    # sum the items from smallest to largest and subtarct from the largest item as we go
    for i in range(len(nums)):
        val = nums[i] - reduction
        if nums[i] != 0 and nums[i] != prev:
            largest -= val
            reduction += val
            ops += 1
            if largest <= 0:
                return ops
            prev = nums[i]

# damn
def minimumOperations(self, nums: List[int]) -> int:
    if 0 in nums:
        return len(set(nums))-1
    else:
        return len(set(nums))