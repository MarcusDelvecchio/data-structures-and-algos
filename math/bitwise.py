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


# Bitwise ORs of Subarrays LeetCode Medium
# this solution is simple BUT DIFFICULT in that you must realize aspects that allow the time complexity to be limited
# https://leetcode.com/problems/bitwise-ors-of-subarrays/description/
#
# TC: O(n*w), SC: O(n*w) where w is a constant (see below)
# observations that led to the solution:
# 1. When continuously applying OR operations, once a digit becomes 1, it cannot become zero again
# 2. If I have a range of BINARY numbers, the maximum number of possible ORs results of any subset of those numbers is thus is max(bin_arr, key=len)^2-1
# 
# since any compounded OR will always be some combination of ones up to that max length
# 
# and because of the second observation, the time complexity is NOT O(n^2), but O(n*w) where w is THAT MAX
# because unlike when producing all SUBSETS, where we iterate over all elements and at each element we add that element to ALL existing subsets, which could be MAX SIZE N
# here, our existing "subsets" or in this case, previous OR results, can only be max size W
#
# Time complexity: The time complexity is O(n*w)
# n is the length of the input array
# w is the LARGEST POSSIBLE VALUE representable by the longest of all the values in arr after comverting them to binary
#
# takeaways: you would THINK with the inner loop, the solution would be O(n^n) because it may look like we are doing something like iterating over all subsets (all ORs)
# BUT because of this 'accumulative OR' property, the maximum number of existing ORs (at ANY point) will be the most number of binary numbers that can be represented by the number of digits in the largest number in arr
# so if max(bin(arr), key=len) is "1100" -> there can only be 15 possible OR values in the set MAX
# so for every n, we will only loop max 10 times when creating new OR set entries in that innner loop.
def subarrayBitwiseORs(self, arr: List[int]) -> int:
    ans = set()
    possibilitiesEndingAt = set()
    for num in arr:
        possibilitiesEndingAt = {num | OR for OR in possibilitiesEndingAt} # or the num with all existing ORs to create new ORs including this value
        possibilitiesEndingAt.add(num) # add this value only
        ans |= possibilitiesEndingAt # add all not yet included ORs to ans (take union of the two)
    return len(ans)