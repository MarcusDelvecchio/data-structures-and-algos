# Find the Pivot Integer LeetCode Easy
# took like 5 mins
# TC: O(n), SC: O(1)
def pivotInteger(self, n: int) -> int:
    left, right = 0, n * (n + 1) // 2
    curr = 1
    while left < right:
        left += curr

        if left == right:
            return curr

        right -= curr
        curr += 1
    return -1

# Product of Array Except Self LeetCode Medium
# https://leetcode.com/problems/product-of-array-except-self/
# TC: O(n), SC: O(n)
# took 12 mins
def productExceptSelf(self, nums: List[int]) -> List[int]:
    total_prod, zeros = 1, 0
    without_zero = 0
    for num in nums:
        if num == 0:
            zeros += 1
            if zeros == 1:
                without_zero = total_prod
        else:
            without_zero*= num
        total_prod*= num
    return [total_prod//num if num != 0 else without_zero if zeros == 1 else 0 for num in nums]

# N-th Tribonacci Number LeetCode Easy
# Return the nth tribonacci number where trib(n) = trib(n-1) + trib(n-2) + trib(n-3)
# https://leetcode.com/problems/n-th-tribonacci-number/description/
# took 4 mins
# TC: O(n), SC:O(1)
def tribonacci(self, n: int) -> int:
    if n == 0: return 0
    first = 0
    second = third = 1

    num = 2
    while num < n:
        new_third = third + second + first
        first = second
        second = third
        third = new_third
        num += 1

    return third 