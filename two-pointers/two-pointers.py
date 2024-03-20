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