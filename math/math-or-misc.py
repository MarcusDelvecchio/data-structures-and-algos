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