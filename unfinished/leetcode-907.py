# couldn't figure thius porblem out
# https://leetcode.com/problems/sum-of-subarray-minimums/?envType=daily-question&envId=2024-01-20
# after over an hour
# see https://www.youtube.com/watch?v=aX1F2-DrBkQ&list=LL&index=1&t=572s
# see https://www.youtube.com/watch?v=9yKSZ32I2S0

def sumSubarrayMins(self, arr: List[int]) -> int:
    res = 0
    def get_sublists_len_n(n):
        curr, total = 0, 0
        while curr + n < len(arr) + 1:
            total += min(arr[curr:curr+n])
            curr += 1
        return total

    for i in range(1, len(arr) + 1):
        res += get_sublists_len_n(i)
    return res % (pow(10, 9) + 7)


def sumSubarrayMins(self, arr: List[int]) -> int:
    res = 0
    for i in range(len(arr)):
        minimum = arr[i]
        for j in range(i, len(arr)):
            if arr[j] < minimum:
                minimum = arr[j]
            res += minimum

    return res % (pow(10, 9) + 7)

# attempt 3 using https://www.youtube.com/watch?v=vjxBVzVB-mE
def sumSubarrayMins(self, arr: List[int]) -> int:
    res = 0
    for i in range(len(arr)):
        # go backwards
        left = 1
        for j in range(i, -1, -1):
            if arr[j] < arr[i] or (arr[j] == arr[i] and i != j): break
            if i != j:
                left += 1
        
        # check to the right
        right = 1
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[i]: break
            right += 1
        # print(left, right)
        res += arr[i]*left*right                

    return res % (pow(10, 9) + 7)

# given every element of arr
# how many time will each element be a minimum of some subarray?