from typing import Optional, List

# Minimum Number of Operations to Make Array XOR Equal to K LeetCode Medium
# https://leetcode.com/problems/minimum-number-of-operations-to-make-array-xor-equal-to-k/?envType=daily-question&envId=2024-04-29
# TC: O(n), SC: O(n)
# idea: to calculate cumulative bitwise XOR, XOR all elements (one ata  time)
# then with our resuylt, we can change a single bit in any of the numbers
# and note that this single bit change can propagate through all of the operations
# to flip a bit in the final cumulative XOR
# so the question becomes, how many bit to we have to change in the cumulative XOR to get k
def minOperations(self, nums: List[int], k: int) -> int:

    # calculate cumulative XOR
    xor = nums[0]
    for i in range(1, len(nums)):
        xor ^= nums[i]

    # convert items to binary
    k_base2 = f'{k:08b}'
    xor = f'{xor:08b}'

    # backfill smaller item with zeros
    if len(k_base2) > len(xor):
        xor = '0'*(len(k_base2) - len(xor)) + xor
    elif len(k_base2) < len(xor):
        k_base2 = '0'*(len(xor) - len(k_base2)) + k_base2

    # compare differences in bits
    ans = 0
    for i in range(len(xor)):
        if k_base2[i] != xor[i]:
            ans += 1
    return ans