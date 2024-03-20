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
# the catch/trick for this problem: 3 positive integers cannot sum to 0
# and also remember that twoSum is O(n)
# !! much faster solution below with left/right pointer 2sum instead of dictionary
def threeSum(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
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
# TC: O(n), SC: O(n)
def threeSum(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
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