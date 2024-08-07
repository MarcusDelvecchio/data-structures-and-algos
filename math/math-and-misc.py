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

# Number of Steps to Reduce a Number in Binary Representation to One LeetCode Medium
# https://leetcode.com/problems/number-of-steps-to-reduce-a-number-in-binary-representation-to-one/description/
# TC: O(logn) -> constantly dividing, SC: (1)
# very easy took 2 mins
def numSteps(self, s: str) -> int:
    num = int(s, 2)
    steps = 0
    while num != 1:
        if num % 2 == 0:
            num //= 2
            steps += 1
        else:
            num  = (num+1)//2
            steps += 2

    return steps

# Special Array With X Elements Greater Than or Equal X LeetCode Easy
# https://leetcode.com/problems/special-array-with-x-elements-greater-than-or-equal-x/description
# : a value x makes an array 'special' if there are x values in the array that are greater or equal to x
# : given an array, return x if there is one, else -1
# approach: sort the list in decreasing order and iterate forwards, keeping track of the count of numbers we've seen
# at any point if we have 1. seen x numbers and 2. and the current number is also greater than x, but 3. the next is not, then x is a target number
# also if we are at the end of the list, condition 3 does not need to be satisfied as there are no more numbers
# TC: O(n), SC: O(1)
def specialArray(self, nums: List[int]) -> int:
    nums.sort(reverse=True)
    for i in range(len(nums)):
        count = i + 1
        if i == len(nums) - 1:
            if nums[i] >= count:
                return count
        elif nums[i] >= count and nums[i+1] < count:
            return count
    return -1

# The kth Factor of n LeetCode Medium (this solution is Easy, see below fo medium in < O(n))
# https://leetcode.com/problems/the-kth-factor-of-n/description/?envType=study-plan-v2&envId=amazon-spring-23-high-frequency
# TC: O(n), SC: O(1)
# : You are given two positive integers n and k. A factor of an integer n is defined as an integer i where n % i == 0.
# : Consider a list of all factors of n sorted in ascending order, return the kth factor in this list or return -1 if n has less than k factors.
# follow up is: how to solve this in < O(n) time?
def kthFactor(self, n: int, k: int) -> int:
    factors = set()
    for i in range(1, n+1):
        if n % i == 0:
            k -= 1
            if k == 0:
                return i
    return -1

# The kth Factor of n LeetCode Medium O(sqrt(n)*logk)
# https://leetcode.com/problems/the-kth-factor-of-n/description/?envType=study-plan-v2&envId=amazon-spring-23-high-frequency
# note there is also a way to do it in O(nsqrt(n)) see https://leetcode.com/problems/the-kth-factor-of-n/solutions/3882180/python-o-sqrt-n-with-explaination-top-95/?envType=study-plan-v2&envId=amazon-spring-23-high-frequency
# we only need to check factors up until sqrt(n) because all other factors larger than sqrt(n) may be derived from factors less than sqrt(n) -> because finding a factor gives us a second factor
# so iterater up to sqrt(n) and add every factor and it's pairing factor to a heap that maintains the k smallest factors
# if the heap grows beyond k factors, pop the largest factors (flipped-value min heap)
# and then at the end return the largest (thus, the kth overall largest) factor from the heap
# TC: O(sqrt(n)logk), SC: O(k)
def kthFactor(self, n: int, k: int) -> int:
    factors = [] # heap representing the k smallest factors
    for i in range(1, int(math.sqrt(n))+1):
        val = n
        if val % i == 0:
            heapq.heappush(factors, -i)
            if i != n//i:
                heapq.heappush(factors, n//-i)

            while len(factors) > k:
                item = heapq.heappop(factors)
    return -heapq.heappop(factors) if len(factors) == k else -1

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

# Multiply Strings LeetCode Medium
# https://leetcode.com/problems/multiply-strings/description/
# TC: O(len_num1*len_num2) = O(n*m), SC: O(1)
# approach: multiple each digit of num1 by each digit of num 2 while also accounting for each digit's position in the string by and multiplying by a factor of 10
def multiply(self, num1: str, num2: str) -> str:
    total = 0
    num1_position = 0
    for i in range(len(num1)-1, -1, -1):
        num2_position = 0
        for j in range(len(num2)-1, -1, -1):
            num1_val = int(num1[i])*10**num1_position
            num2_val = int(num2[j])*10**num2_position
            total += num1_val*num2_val
            num2_position += 1
        num1_position += 1
    return str(total)

# Pow(x, n) LeetCode Medium - Implement Power function
# https://leetcode.com/problems/powx-n/
# TC: O(n), SC: O(n)
def myPow(self, x: float, n: int) -> float:
    if n == 0: return 1
    if n < 0:
        return 1/self.myPow(x, -n)

    if n % 2 == 0:
        res = self.myPow(x, n/2)
        return res*res
    return x*self.myPow(x, n-1)

# Rotate Image LeetCode Medium
# https://leetcode.com/problems/rotate-image/description/
# TC: O(n): SC: O(1)
def rotate(self, matrix: List[List[int]]) -> None:
    # see https://assets.leetcode.com/users/images/0ab215cd-9cd8-4872-90a7-901fb660dc67_1682068950.8864057.gif
    # tanspose the matrix - swap the row and col of every cell (flip the matrix over the line y = -x)
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if row < col:
                new_row, new_col = col, row
                temp = matrix[new_row][new_col]
                matrix[new_row][new_col] = matrix[row][col]
                matrix[row][col] = temp
    
    # swap the columns - flip the matrix over a vertical line down the middle
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if col < len(matrix[0])//2:
                new_col = len(matrix[0]) - col - 1
                temp = matrix[row][new_col]
                matrix[row][new_col] = matrix[row][col]
                matrix[row][col] = temp