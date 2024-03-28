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