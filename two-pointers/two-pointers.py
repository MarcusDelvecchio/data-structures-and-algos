from types import List

# Valid Palindrome LeetCode Easy
# https://leetcode.com/problems/valid-palindrome/description/
# took 6 mins
def isPalindrome(self, s: str) -> bool:
    s = [c.lower() for c in s if c.isalnum()]
    first = 0
    last = len(s) - 1
    while first <= last:
        if s[first] != s[last]: 
            return False
        first += 1
        last -= 1
    return True

# Two Sum II - Input Array Is Sorted - LeetCode Medium
# https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/
# TC: O(n), SC: O(1)
def twoSum(self, numbers: List[int], target: int) -> List[int]:
    # set left and right pointers to start and end
    start, end = 0, len(numbers) - 1
    while start < end:
        sum_ = numbers[start] + numbers[end]
        if sum_ == target: return [start+1, end+1]
        elif sum_ > target: end -= 1
        elif sum_ < target: start += 1

# 3Sum LeetCode Medium
# https://leetcode.com/problems/3sum/description/
# TC: O(n^2) SC: O(n)
# and also remember that sorted twoSum is O(n)
# !! much faster solution below with left/right pointer 2sum instead of dictionary - is it that much faster?
def threeSum(self, nums: List[int]) -> List[List[int]]:
    nums.sort() # sort - O(nlogn)
    res = set()

    for i in range(len(nums)):
        if nums[i] > 0: break # if the lowest number is positive all vals cannot sum to 0
        if i == 0 or nums[i-1] != nums[i]:  # skip duplicates
            need = set()
            for j in range(i+1, len(nums)):  # simply perform two sum on remaining two items O(n)
                if nums[j] in need:
                    res.add((nums[i], -(nums[i] + nums[j]), nums[j]))
                else:
                    need.add(-(nums[i] + nums[j]))
    return res

# Better solution with left/right pointer 2sum instead of dictionary
# TC: O(nlogn) + O(n^2) = O(n^2)
# SC: O(n)
def threeSum(self, nums: List[int]) -> List[List[int]]:
    nums.sort() # sorting - O(nlogn)
    res = []

    for i in range(len(nums)):
        if nums[i] > 0: break # if the lowest number is positive all vals cannot sum to 0
        if i == 0 or nums[i-1] != nums[i]:  # skip duplicates
            # simply perform two sum soted on remaining two items O(n)
            left, right = i + 1, len(nums) - 1
            while left < right:
                three_sum = nums[left] + nums[right] + nums[i]
                if three_sum == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    left += 1
                    while nums[left] == nums[left-1] and left < right:
                        left += 1
                elif three_sum > 0:
                    right -= 1
                else: # nums[left] + nums[right] < -nums[i]:
                    left += 1
    return res

# 3Sum Smaller LeetCode Medium
# Given an array of n integers nums and an integer target, find the number of index triplets (with different indices) that satisfy the condition nums[i] + nums[j] + nums[k] < target
# https://leetcode.com/problems/3sum-smaller/description/
# TC: O(n^2), SC: (1)
# took 15 mins
# algo summary: 3 pointers. first pointer starts at 0 and moves fwd, second starts at first+1 and thirs starts at end
# algo: 1. while sum > k move third pointer in. 
#       2. When p3 gets to when sum < k, p3 can be reduced all the way to p2 and have accepted answers, so add p3-p2 to ans
#       3. increase p2 and repeat (p3 wil thus be forced to reduce again to make up for the increase in p2)
#       4. when p2 gets to p3, increment p1z
def threeSumSmaller(self, nums: List[int], target: int) -> int:
    nums.sort()
    ans = 0
    for i in range(len(nums)):
        left, right = i+1, len(nums)-1
        while left < right:
            threeSum = nums[i] + nums[left] + nums[right]
            while threeSum >= target and right > left:
                right -= 1
                threeSum = nums[i] + nums[left] + nums[right]
            # add all solutions for p3 being reduced to p2
            ans += right-left

            # then increase p2 (but leave p3 where it has been reduced to)
            left += 1
    return ans

# 3Sum Closest LeetCode Medium
# https://leetcode.com/problems/3sum-closest/description/
# TC: O(n^2), SC: O(n)
def threeSumClosest(self, nums: List[int], target: int) -> int:
    nums.sort()
    n = len(nums)
    best = float('inf')

    for p1 in range(n):
        p2, p3 = p1 + 1, n-1
        while p2 < p3:
            sum = nums[p1]+nums[p2]+nums[p3]
            if abs(sum-target) < abs(best-target):
                best = sum
            if sum == target:
                return target
            elif sum-target<0:
                p2 += 1
            else:
                p3 -= 1
    return best

# 3SUM // Valid Triangle Number Leetcode Medium
# https://leetcode.com/problems/valid-triangle-number/description/
# Given an integer array nums, return the number of triplets chosen from the array that can make triangles if we take them as side lengths of a triangle.
# TC: O(n^2), SC: O(n)
# took like 20 because actually different than the above problems
# this is just 3sum but where the third pointer must be greater than the sum of the first 2
def triangleNumber(self, nums: List[int]) -> int:
    nums.sort()
    ans = 0
    for p1 in range(len(nums)):
        p2, p3 = p1+1, p1+2
        while p2 < p3 and p3 < len(nums):
            first_2_sum = nums[p1] + nums[p2]
            while p2 < p3 and first_2_sum <= nums[p3]:
                p2 += 1
                first_2_sum = nums[p1] + nums[p2]
            ans += p3-p2
            p3 += 1
    return ans

# Count Subarrays Where Max Element Appears at Least K Times LeetCode Medium
# You are given an integer array nums and a positive integer k.
# Return the number of subarrays where the maximum element of nums appears at least k times in that subarray.
# took 11 mins
# TC: O(n), SC: O(1)
# sliding window/2-pointers, similar to the above questions
def countSubarrays(self, nums: List[int], k: int) -> int:
    maxx = max(nums)
    right = ans = 0
    max_count = 0 if nums[0] != maxx else 1
    for left in range(len(nums)):
        while right < len(nums)-1 and max_count < k:
            right += 1
            if nums[right] == maxx:
                max_count += 1
        
        # when there are at least k instances of the number, the right could be expanded all the way to the end, so account for all of those subarrays
        if max_count >= k:
            ans += len(nums)-1-right+1
        if nums[left] == maxx:
            max_count -= 1
    return ans

# Container With Most Water LeetCode Medium
# https://leetcode.com/problems/container-with-most-water/
# TC: O(n), SC: O(1)
# this solution is more readable but shorter on below
# took like 5 mins because studied it a few months ago
def maxArea(self, height: List[int]) -> int:
    left, right, maxx = 0, len(height) - 1, 0
    while left < right:
        maxx = max(maxx, min(height[left], height[right])*(right-left))
        if height[left] < height[right]:
            left += 1
        elif height[left] > height[right]:
            right -= 1
        elif height[left+1] > height[right-1]:
            left += 1
        else:
            right -= 1
    return maxx

# combined if statements from solution above
def maxArea(self, height: List[int]) -> int:
    left, right, maxx = 0, len(height) - 1, 0
    while left < right:
        maxx = max(maxx, min(height[left], height[right])*(right-left))
        if height[left] < height[right] or (height[left] == height[right] and height[left+1] > height[right-1]):
            left += 1
        else: #if height[left] > height[right] or (height[left] == height[right] and height[left+1] > height[right-1]):
            right -= 1
    return maxx

# Merge Operations to Turn Array Into a Palindrome LeetCode Medium
# https://leetcode.com/problems/merge-operations-to-turn-array-into-a-palindrome/description/?envType=study-plan-v2&envId=amazon-spring-23-high-frequency
# : You are given an array nums consisting of positive integers. You can perform an operaiton to combine any two adjacent elements into their sum
# : Return the minimum number of operations needed to turn the array into a palindrome.
# appraoch: use two pointers from left and right. When pointer values are not equal, combine the lesser one with it's next value and move that pointer forward
# TC: O(n), SC: O(1)
# tried to use stack at the beginning which wasn't a great decision because why use O(n) space when you can use O(1)
# (using stack) didn't realize edge case: when number of items is odd we can't simply ignore the center item becuase this oculd be used in an operation on one side to combine to get a value
# also didn't consider edge case: negative numbers. Even though they are all positive, I didn't even realize until I realized the edge case and read that it wasn't an issue 10 mins in
def minimumOperations(self, nums: List[int]) -> int:
    left, right, ops = 0, len(nums)-1, 0

    while left < right:
        if nums[left] != nums[right] and nums[left] < nums[right]:
            left += 1
            nums[left] += nums[left-1]
            ops += 1
        elif nums[left] != nums[right] and nums[left] > nums[right]:
            right -= 1
            nums[right] += nums[right+1]
            ops += 1
        else:
            left += 1
            right -= 1
    return ops

# Valid Palindrome IV LeetCode Medium
# https://leetcode.com/problems/valid-palindrome-iv/description/?envType=study-plan-v2&envId=amazon-spring-23-high-frequency
# : You are given a 0-indexed string s consisting of only lowercase English letters. In one operation, you can change any character of s to any other character.
# : Return true if you can make s a palindrome after performing exactly one or two operations, or return false otherwise.
# TC: O(n), SC: O(n) 
# approach: write a function that checks for palindrome using 2 pointers converging to the center of the string
# if there are chances remaining and the pointer's characters don't match, skip them both (equivalent to setting either one to the other)
# and recall the function with 1 less chance
def makePalindrome(self, s: str) -> bool:
    
    def checkForPalindrome(s, chances):
        if chances == -1: return False
        left, right = 0, len(s)-1
        while left < right:
            if s[left] != s[right]:
                return checkForPalindrome(s[left+1:right], chances-1)
            left += 1
            right -= 1
        return True

    return checkForPalindrome(s, 2)