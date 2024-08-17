from types import List
from collections import defaultdict, Counter

# Largest Odd Number in String LeetCode Easy
# https://leetcode.com/problems/largest-odd-number-in-string/description/
# took 1:30
# a odd num will always end in an odd digit
# iterate backwards in num until we find the rightmost odd and then return the entire num before that
def largestOddNumber(self, num: str) -> str:
    for i in range(len(num)-1, -1, -1):
        if int(num[i])%2==1:
            return num[:i+1]
    return ""

# Largest Perimeter Triangle LeetCode Easy
# https://leetcode.com/problems/largest-perimeter-triangle/description/
# Given an array of integers, return the largest triangle perimeter (if any) that can be produced from
# any 3 numbers in the list
def largestPerimeter(self, nums: List[int]) -> int:
    nums.sort(reverse=True)
    for largest in range(0, len(nums)-2):
        if nums[largest] < nums[largest+1] + nums[largest+2]:
            return nums[largest] + nums[largest+1] + nums[largest+2]
    return 0

# Boats to Save People LeetCode Medium
# https://leetcode.com/problems/boats-to-save-people/description/
# TC: O(nlogn)
# SC: O(1)
def numRescueBoats(self, people: List[int], limit: int) -> int:
    # sort the people based on weight
    people.sort()

    # take largest and smallest person (or just largest if it exceeds) until no more people
    boats = 0
    smallest, largest = 0, len(people)-1
    while smallest <= largest:
        if people[largest] + people[smallest] <= limit:
            boats += 1
            largest -= 1
            smallest += 1
        else:
            boats += 1
            largest -= 1
    return boats

# Best Sightseeing Pair LeetCode Medium
# https://leetcode.com/problems/best-sightseeing-pair/description/
# Given an array of values, find the pair of values with the largest score
# the score is calculated by: adding the values and subtracting the distance between them in the array
# also included in DP section. Keeping here because it is relevant to the question below
# TC: O(n), SC: O(1)
def maxScoreSightseeingPair(self, values: List[int]) -> int:
    
    # iterate through the array and hold on to the highest value, but reduce it by one every time
    max_score = best_spot = 0
    for value in values:
        max_score = max(max_score, value+best_spot)
        best_spot = max(best_spot, value)
        best_spot -= 1
    return max_score

# Maximum Number of Points with Cost LeetCode Medium
# https://leetcode.com/problems/maximum-number-of-points-with-cost/description/?envType=daily-question&envId=2024-08-17
# Given a matrix of values, output the largest sum of values you can by selecting one cell from each row, but also subtracting the distance between the columns of the cells from the adjacent rows
# i.e., you can use ANY other value in your "path sum", but you will lose points if you pick a cell too far from the cell that you picked in the previous row
# REALLY good question. Went and did the above 'Best Sightseeing Pair' problem after not realizing the optimization and got it after easily
# TC: O(n*m) -> each row is processed 3 times, once to find best score to the left and one from the right, and then to compare to the row below
# SC: O(1), the initial array is updated in place
def maxPoints(self, points: List[List[int]]) -> int:

    for row in range(len(points)-2, -1, -1):

        # create list of items representing the best score of all items to the left
        best_score_left = [0]*len(points[0])
        best_score_left[0] = points[row+1][0]
        for col in range(1, len(points[0])):
            best_score_left[col] = max(best_score_left[col-1]-1, points[row+1][col])

        # create list of items representing the best score of all items to the right
        best_score_right = [0]*len(points[0])
        best_score_right[-1] = points[row+1][-1]
        for col in range(len(points[0])-2, -1, -1):
            best_score_right[col] = max(best_score_right[col+1]-1, points[row+1][col])

        # apply the best score to each cell based on the best score to the left and right of the cell directly below it
        for col in range(len(points[0])):
            points[row][col] += max(best_score_right[col], best_score_left[col])

    return max(points[0])

# K Items With the Maximum Sum LeetCode Easy
# https://leetcode.com/problems/k-items-with-the-maximum-sum/description/
# TC: O(n), SC: O(1)
def kItemsWithMaximumSum(self, numOnes: int, numZeros: int, numNegOnes: int, k: int) -> int:
    ans = 0
    while k > 0 and numOnes > 0:
        ans += 1
        k -= 1
        numOnes -= 1
    while k > 0 and numZeros > 0:
        k -= 1
        numZeros -= 1
    while k > 0 and numNegOnes > 0:
        ans -= 1
        k -= 1
        numNegOnes -= 1
    return ans

# Maximum 69 Number LeetCode Easy
# https://leetcode.com/problems/maximum-69-number/
# took 2 mins
def maximum69Number (self, num: int) -> int:
    # remove the first occurance of 6?
    num = str(num)
    for i in range(len(num)):
        if num[i] == "6": return int(num[:i] + "9" + num[i+1:])
    return int(num)

# Minimum Sum of Four Digit Number After Splitting Digits LeetCode Easy
# Given a number, take all of it's digits and compose 2 new numbers with the smallest possible sum
# TC: O(1), would be O(nlogn) but array is always size 4
def minimumSum(self, num: int) -> int:
    nums = [c for c in str(num)]
    nums.sort()
    return int(nums[0] + nums[3]) + int(nums[1] + nums[2])

# Optimal Partition of String LeetCode Medium
# https://leetcode.com/problems/optimal-partition-of-string/description/
# : Given a string s, partition the string into one or more substrings such that the characters in each substring are unique. 
# : That is, no letter appears in a single substring more than once.
# : Return the minimum number of substrings in such a partition.
# again, can't believe this is a medium. I actually struggles with a few greedy easys
# TC: O(n), SC: O(n)
# took 2:10
def partitionString(self, s: str) -> int:
    seen, subs = set(), 1
    for c in s:
        if c in seen:
            subs += 1
            seen = set([c])
        else:
            seen.add(c)
    return subs

# Maximize Happiness of Selected Children LeetCode Medium
# https://leetcode.com/problems/maximize-happiness-of-selected-children/description/
# Given an array happiness where happiness[i] represents the happiness of child i, you want to select k children in turns. When you select a child
# the score goes up by that child's happiness value, but the remaining children's happiness is all reduced by 1. A child's happiness also cannot become negative but rather, stay at zero.
# return the largest score / happiness of all selected children you can obtain
# approach: greedy, you would think there might be some strategy for choosing smallest vs largest first, but it alwyas makes the most sense to select the largest value, becuase k is always less than n, 
# so you don't want to lose out on larger values by not selecting them until the end or potentially not selecting them at all
# approach: sort the chidlren, continuously take the largest child value until you have taken k children
# TC: O(nlogn), SC: O(1) 
def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
    happiness.sort(reverse=True)
    reduction = ans = 0
    for val in happiness:
        if not k: break
        ans += max(val - reduction, 0)
        reduction += 1
        k -= 1
    return ans

# Maximum Number of Coins You Can Get LeetCode Medium
# greedy mediums easier than LC gards
#
#
#
def maxCoins(self, piles: List[int]) -> int:
    piles.sort()
    left, right = 0, len(piles)-2
    ans = 0
    while left < right:
        ans += piles[right]
        right -= 2
        left += 1
    return ans

# Partition Array Into Three Parts With Equal Sum LeetCode Easy
# Given an array of integers arr, return true if we can partition the array into three non-empty parts with equal sums
# https://leetcode.com/problems/partition-array-into-three-parts-with-equal-sum/description/
# TC: O(n^2), SC: O(1)
# harder than expected but returning to LC so idk
def canThreePartsEqualSum(self, arr: List[int]) -> bool:
    target = sum(arr)/3
    left_sum = 0
    for left in range(len(arr)):
        left_sum += arr[left]
        if left_sum == target:
            mid_sum = 0
            for right in range(left+1, len(arr)-1): # note the -1 at the end of edge cases blow up
                mid_sum += arr[right]
                if mid_sum == target:
                    return True
    return False

# Minimum Number of Keypresses LeetCode Medium
# https://leetcode.com/problems/minimum-number-of-keypresses/description/?envType=study-plan-v2&envId=amazon-spring-23-high-frequency
# : You have a keypad with 9 buttons, numbered from 1 to 9, each mapped to lowercase English letters. You can choose which characters each button is matched to as long as:
# : All 26 lowercase English letters are mapped to.
# : Each character is mapped to by exactly 1 button.
# : Each button maps to at most 3 characters.
# : To type the first character matched to a button, you press the button once. To type the second character, you press the button twice, and so on.
# : Given a string s, return the minimum number of keypresses needed to type s using your keypad.
# TC: O(n), SC: O(1)
# approach: assign the most frequent characters to 'first slots' on a number until there are not more first slots,
# then second slots then third, tracking the excpected presses based on the frequency of items and the slots we place them in
def minimumKeypresses(self, s: str) -> int:
    # get the frequencies of each letter
    freq = collections.Counter(s)

    # create array of most frequent items
    most_freq = sorted([c for c in freq], key=lambda c: freq[c], reverse=True)

    # dispense letters across keys based on positions
    ans = 0
    first_slots = second_slots = 9
    for c in most_freq:
        if first_slots: # if there are first slots remaining, put the next most frequent character ina first slot and add char_frequency*1 to ans
            first_slots -= 1
            ans += freq[c]*1
        elif second_slots: # else if there are second slots remaining, put the next most frequent character ina second slot and add char_frequency*2 to ans
            second_slots -= 1
            ans += freq[c]*2
        else:  # put the next most frequent character in third slot and add char_frequency*3 to ans
            ans += freq[c]*3
    return ans

# Find Minimum Time to Finish All Jobs II LeetCode Medium
# https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs-ii/?envType=study-plan-v2&envId=amazon-spring-23-high-frequency
# TC: O(nlogn), SC: O(n)
# approach: sort the jobs and workers and assign largest job to worker that can do most work, second largest6 to second best worker etc. keeping track of
# what worker/job pair will take the longest
# return the largest quotient for job size / worker efficiency
# TC: O(nlogn), SC: O(1/n)
def minimumTime(self, jobs: List[int], workers: List[int]) -> int:
    jobs.sort()
    workers.sort()

    maxx = 0
    for i in range(len(jobs)):
        time_required = math.ceil(jobs[i] / workers[i])
        maxx = max(maxx, time_required)

    return maxx

# Minimize Maximum Pair Sum in Array LeetCode Medium
# this medium is really easier than some of the easy greedy problems here
# https://leetcode.com/problems/minimize-maximum-pair-sum-in-array/
# TC: O(nlog) + O(n) = O(nlogn), SC: O(1)
# took 4 mins
def minPairSum(self, nums: List[int]) -> int:
    nums.sort()
    left, right = 0, len(nums)-1
    max_pair = 0
    while left < right:
        max_pair = max(max_pair, nums[left] + nums[right])
        left += 1
        right -= 1
    return max_pair

# Minimum Number of Moves to Seat Everyone LeetCode Easy
# took 5 mins but could have been 30 seconds
# TC: O(nlogn), SC: O(1)
def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
    seats.sort()
    students.sort()
    moves = 0
    for i in range(len(seats)):
        moves += abs(seats[i] - students[i])
    return moves

# Minimum Cost to Move Chips to The Same Position LeetCode Easy 
# https://leetcode.com/problems/minimum-cost-to-move-chips-to-the-same-position/
# 7 mins
# TC: O(n), SC: O(1)
# approach: all even chiups can be moved to any even position for free and same with odd. All odds can be moved next to all evens for free.
# then whichever pile has less chips, all chips need to be moves 1 over to the next pile 
def minCostToMoveChips(self, position: List[int]) -> int:
    # we want to know the dif between num of even and num of odds
    even = odd = 0
    for pos in position:
        if pos%2 == 0:
            even += 1
        else:
            odd += 1
    return min(odd, even)

# Maximum Sum With Exactly K Elements 
# https://leetcode.com/problems/maximum-sum-with-exactly-k-elements/description/
# took 3 mins becuase tried to do 1-liner
# TC: O(n), SC: O(1)
def maximizeSum(self, nums: List[int], k: int) -> int:
    maxx = max(nums)
    ans = 0
    for i in range(k):
        ans += maxx
        maxx += 1
    return ans

# Minimum Adjacent Swaps to Make a Valid Array LeetCode Medium
# Given an integer array nums where swaps of adjacent elements are able to be performed on nums.
# Return the minimum swaps required so that (one of) the largest value is at the end
# and (one of) the smallest is at the beginning
# https://leetcode.com/problems/minimum-adjacent-swaps-to-make-a-valid-array/description/
# took 5 mins
# TC: O(n), SC: O(1)
# follow up: min swaps but not only to have largest on right and smallest on left, but to fix entire order of all elements
# this would be O(nlogn) because we'd have to sort to determine and compare where every element needs to be vs is. But if it were something like n elements with values from 0-(n-1) we wouldn't need to sort and could figure it out in O(n) time directly.
def minimumSwaps(self, nums: List[int]) -> int:
    maxx, minn = max(nums), min(nums)
    max_idx, min_idx = 0, float('inf')
    for idx, num in enumerate(nums):
        if num == maxx and idx > max_idx:
            max_idx = idx
        elif num == minn and idx < min_idx:
            min_idx = idx
    # return distance of largest from right + distance of smallest from left - 1 IF left is greater than right (becuase then we swap BOTH)
    return (len(nums)-1-max_idx) + min_idx + (-1 if min_idx > max_idx else 0)

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

# Minimum Processing Time LeetCode Medium
# https://leetcode.com/problems/minimum-processing-time/description/
# took like 8 mins
# TC: O(nlogn), SC: O(1)
def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
    tasks.sort(reverse=True)
    processorTime.sort()
    max_end = 0
    for p in range(0, len(tasks), 4):
        max_end = max(max_end, processorTime[(p+1)//4] + tasks[p])
    return max_end

# Can Place Flowers LeetCode Easy
# You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in adjacent plots.
# Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return true if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule and false otherwise.
# https://leetcode.com/problems/can-place-flowers/description/
# TC: O(n), SC: O(n)
# took 13 mins
def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    available, ans = collections.defaultdict(bool), 0
    # all spots that have no 1 before or after, mark as available
    for i in range(len(flowerbed)):
        if flowerbed[i] == 0 and (i == 0 or flowerbed[i-1] != 1) and (i == len(flowerbed) - 1 or flowerbed[i+1] != 1):
            available[i] = True
    # iterate though available and mark adjacent spots as unavailable as we do so
    for i in range(len(flowerbed)):
        if available[i]:
            ans += 1
            if i < len(flowerbed)-1:
                available[i+1] = False
    return ans >= n

# Longest Palindrome LeetCode Easy
#
# TC: O(n), SC: O(1) -> space complexity is actually O(n) but since characters can only be A-Z/a-z is O(1)
def longestPalindrome(self, s: str) -> int:
    counts, ans, hasCenter = collections.Counter(s), 0, False
    for c in counts.values():
        if c%2 == 1: hasCenter = True # if any char count has an odd number we can have a center value
        ans += c//2
    return ans*2 + (1 if hasCenter else 0)

# Minimum Suffix Flips LeetCode Medium
# https://leetcode.com/problems/minimum-suffix-flips/
# Given a binary string of length n and a second binary string of length n intitially set to all zeros
# You want to make s equal to target. In one operation, you can pick an index i where 0 <= i < n and 
# flip all bits in the inclusive range [i, n - 1]. Flip means changing '0' to '1' and '1' to '0'.
# Return the minimum number of operations needed to make s equal to target.
# took 11 mins.
# see shorter version below
# TC: O(n), SC: O(1)
def minFlips(self, target: str) -> int:
    ans = 0
    flipped = False
    for i in range(len(target)):
        if int(target[i]) == (1 if flipped else 0):
            continue
        else:
            ans += 1
            flipped = not flipped
    return ans

# shorter answer than above
def minFlips(self, target: str) -> int:
    ans = flipped = 0
    for i in range(len(target)):
        if int(target[i]) != int(flipped):
            ans += 1
            flipped = not flipped
    return ans

# Minimum Number of Swaps to Make the String Balanced LeetCode Medium
# https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/
# Given an even string of open/close brackets, return the minimum number of swaps to make s balanced.
# TC: O(n), SC: O(1)
# got lucky with this one and solved in like 5
def minSwaps(self, s: str) -> int:
    count = ans = 0
    for c in s:
        count += (1 if c == "[" else -1)
        if count < 0:
            ans += 1
            count += 2
    return ans

# Valid Palindrome II LeetCode "Easy"
# https://leetcode.com/problems/valid-palindrome-ii/description/
# Given a string s, return true if the s can be palindrome after deleting at most one character from it.
# note this is also very similar to the below problem
# approach: two pointers; Move pointers in until chars are not equal. If not equal, either could be removed (i.e., "skipped") and moved forward once for free
# so we take the two possible strings that could be made up from removing the char at either pointer and return true if either is a palindrome
# TC: O(n), SC: O(n)
# took like 25 because didn't realize this simple approach. Was using 2ptr approach but was trying some janky logic to compare the remaining
# of the two strings if either pointer char was skipped. Didn't realize you could just check at that point if the strings were palindromic
def validPalindrome(self, s: str) -> bool:
    left, right = 0, len(s)-1
    while left <= right:
        if s[left] != s[right]:
            s1 = s[:left]+s[left+1:]
            s2 = s[:right]+s[right+1:]
            return s1==s1[::-1] or s2==s2[::-1]
        right -= 1
        left += 1
    return True

# Maximum Subarray LeetCode Medium
# https://leetcode.com/problems/maximum-subarray/description/
# : Given an integer array nums, find the subarray with the largest sum, and return its sum.
# TC: O(n), SC: O(1)
# took like 10 mins to think about but 1 minute after realizing approach
# one of those questions where when you realize the approach it is easy.
# but at first I stared at this question for like 5 mins not getting it and thinking there was some
# special DP trick. But once you realize how simple it is, the question is really a Medium
# if it was an Easy I would have probably gotten it sooner because of how complicated I expected it to be
# see Kadane's algorithm
def maxSubArray(self, nums: List[int]) -> int:
    maxx, curr = -float('inf'), 0
    for num in nums:
        curr += num
        maxx = max(maxx, curr)
        if curr < 0:
            curr = 0
    return maxx

# Maximum Absolute Sum of Any Subarray LeetCode Medium
# see even more simplified version below but WAY less readable/inuitive
# very similar to above question
# TC: O(n), SC: O(1)
# see Kadane's algorithm
def maxAbsoluteSum(self, nums: List[int]) -> int:
    neg_sum = pos_sum = largest_abs = 0
    for num in nums:
        neg_sum += num
        pos_sum += num
        if neg_sum > 0:
            neg_sum = 0
        if pos_sum < 0:
            pos_sum = 0
        largest_abs = max(largest_abs, abs(pos_sum), abs(neg_sum))
    return largest_abs

# even more simplified
def maxAbsoluteSum(self, nums: List[int]) -> int:
    neg_sum = pos_sum = largest_abs = 0
    for num in nums:
        neg_sum = min(neg_sum + num, 0)
        pos_sum = max(pos_sum+num, 0)
        largest_abs = max(largest_abs, abs(pos_sum), abs(neg_sum))
    return largest_abs

# Longest Turbulent Subarray LeetCode Medium
# https://leetcode.com/problems/longest-turbulent-subarray
# TC: O(n), SC: O(1)
# similar to above
# ugly solution
def maxTurbulenceSize(self, arr: List[int]) -> int:
    if len(arr) == 1: return 1
    if len(arr) == 2: return 1 if arr[0] == arr[1] else 2
    expectingGreater = None
    max_len = curr_len = 1
    for i, num in enumerate(arr):
        if i == 0: continue
        if expectingGreater == None:
            expectingGreater = arr[i] < arr[i-1]
            if arr[i] != arr[i-1]:
                curr_len = 2
        elif arr[i-1] == arr[i]:
            curr_len = 1
            expectingGreater = None 
        elif expectingGreater == bool(arr[i] > arr[i-1]):
            curr_len += 1
            expectingGreater = not expectingGreater
        else:
            curr_len = 2
            expectingGreater = arr[i] < arr[i-1]
        max_len = max(max_len, curr_len)
    return max_len

# Maximum Product Subarray LeetCode Medium
# https://leetcode.com/problems/maximum-product-subarray/submissions/1225192552/
# Given an integer array nums, find a  subarray that has the largest product, and return the product. The test cases are generated so that the answer will fit in a 32-bit integer.
# so merked, look out for odd/even number of negatives and zeros
# approach: split array where zeros occur and then trim either the trailing/leading subarray upto/after the first/last nagative
# and then compare all subarrays
# edge cases/factors to keep in mind: negative numbers and zeros
# this is such a merked question/solution
# beats 99% runtime
# took like 40 mins can't even lie
# TC: O(n), SC: O(n)
def maxProduct(self, nums: List[int]) -> int:
    # compute product of entire list
    product = 1
    for num in nums:
        product *= num

    # if total sum is positive or length of list is 1 return total sum
    if product > 0 or len(nums) == 1:
        return product

    # split the list where zeros occur
    lists = []
    curr_list = []
    for num in nums:
        if num != 0:
            curr_list.append(num)
        elif curr_list:
            lists.append(curr_list)
            curr_list = []
    if curr_list:
        lists.append(curr_list)

    # for every negative subarray calculate the maximum value
    max_prod = 0
    for sublist in lists:
        curr_sum = 1
        for num in sublist:
            curr_sum *= num
            max_prod = max(max_prod, curr_sum)

        # if the subarray product is negative, remove the left or right negative number
        if len(sublist) == 1: continue
        if curr_sum < 0:

            # take off the subarray before (and including) the first negative
            right_product = curr_sum
            for num in sublist:
                right_product //= num
                if num < 0:
                    break

            # or take of the subarray after (and including) the last negative
            left_product = curr_sum
            for i in range(len(sublist)-1, -1, -1):
                left_product //= sublist[i]
                if sublist[i] < 0:
                    break
            
            # compare the two and return the greater
            max_prod = max(max_prod, right_product, left_product)
    return max_prod

# two-pass version that does exactly the same thing but no need to split items
# edge cases/factors to keep in mind: negative numbers and zeros
def maxProduct(self, nums: List[int]) -> int:
    currentProduct = 1
    maxProduct = float('-inf')

    # forwards
    for num in nums:
        currentProduct *= num
        maxProduct = max(maxProduct, currentProduct, num)
        if currentProduct == 0:
            currentProduct = num

    # backwards
    currentProduct = 1
    for num in nums[::-1]:
        currentProduct *= num
        maxProduct = max(maxProduct, currentProduct)
        if currentProduct == 0:
            currentProduct = num
    return maxProduct

# One Edit Distance LeetCode Medium
# https://leetcode.com/problems/one-edit-distance/description/
# TC: O(max(s, t)) = O(n), SC: O(max(s, t)) = O(n) (but it actually could easily be done in O(1) by iterating through items rather than using s[x:y] syntax (duplicating string))
# took 10 mins. I initially used 2D DP solution similar to calculate-edit-distance, but realized that was overkill
# not really greedy but putting here because similar to above
# also note you could definitely do this problem with pointers instead of storing the entire rest of the string
# for constant space TODO
# | a note on this problem
# | note that insert and replace is covered (at the point when characters are not the same) by either skipping
# | the current charcater in *one string but not the other* (to 'delete' that character), or by skipping
# | the character in *both strings* (to 'replace' one char with the other so both strings continue)
# | also note that insert is covered by the delete use case because if we need to insert a character to make them
# | the same it is the same thing as deleting that character from the other string
def isOneEditDistance(self, s: str, t: str) -> bool:
    for i in range(min(len(s), len(t))):
        if s[i] != t[i]:
            skip_s_char = s[i+1:] # if we ignore the character in s
            skip_t_char = t[i+1:] # if we ignore the character in t
            # remove char from s == t OR remove char t == s OR remove char from both (replace one with other) makes them eq
            return skip_s_char == t[i:] or skip_t_char == s[i:] or skip_s_char == skip_t_char
    return abs(len(s) - len(t)) == 1 # if no differences, the difference in length must be one for one-edit distance

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

# above solution^ using for loop
def canJump(self, nums: List[int]) -> bool:
    max_jump = 0
    for idx, num in enumerate(nums):
        if idx > max_jump:
            return False
        max_jump = max(max_jump, idx + num)
    return True

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
# todo this should be reviewed
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

# Largest Number LeetCode Medium
# https://leetcode.com/problems/largest-number/description/
# TC: O(n), SC: O(n)
# custom sorting function is simple solution
class LargerNumKey(str):
    def __lt__(x, y):
        # Compare x+y with y+x in reverse order to get descending order
        return x+y > y+x

class Solution:
    def largestNumber(self, nums: List[int]) -> str:

        def compare(x, y):
            return int(x+y) < int(y+x)

        # convert to strings
        nums = [str(num) for num in nums]

        # sort items with custom sort to determine which item should come fist
        nums.sort(key=LargerNumKey)

        # combine all of the numbers
        maximum = "".join(nums)

        # return zero if the first value is "0" for any strings such as "00"
        return "0" if maximum[0] == "0" else maximum


# 
# https://leetcode.com/problems/increasing-triplet-sIncreasing Triplet Subsequence LeetCode Mediumubsequence/description/
# took 18 mins but like 5 after realizing approach
# hint/approach: for any/every element, we want to know if ANY value to the left of it is less than it and ANY value to the right is more than it
# TC: O(n), SC: O(n)
# way more clever solution below
def increasingTriplet(self, nums: List[int]) -> bool:
    largest_to_right = nums.copy()
    largest = 0
    # go right to left getting the largest value to the right of each element
    for i in range(len(nums)-2, -1, -1):
        largest = max(largest, nums[i+1])
        largest_to_right[i] = largest

    # go left to right getting the smallest value to the right of element
    smallest_to_left = nums.copy()
    smallest = float("inf")
    for i in range(1, len(nums)):
        smallest = min(smallest, nums[i-1])
        smallest_to_left[i] = smallest

    # for each element, check if the smallest on the left is less than that element and largest on right is greater
    for i in range(len(nums)):
        if smallest_to_left[i] < nums[i] < largest_to_right[i]:
            return True
    return False

# mind blowing alternative better solution but probably impossible to come up in an interview
# This is a special case of LIS. (lonest increasing subsequence)
# LIS can be solved with O(N log M) where M is the length of sequences (generally M is N). In this question, we can set M as 3 thus the problem can be solved with O(N) with the general LIS approach.
def increasingTriplet(self, nums: List[int]) -> bool:
    first = second = float('inf') 
    for n in nums: 
        if n <= first: 
            first = n
        elif n <= second:
            second = n
        else:
            return True
    return False

# Make Array Zero by Subtracting Equal Amounts LeetCode Easy
# https://leetcode.com/problems/make-array-zero-by-subtracting-equal-amounts/description/
# TC: O(n), SC: O(n)
# way simpler solution below
# took 10 mins this is too slow
def minimumOperations(self, nums: List[int]) -> int:
    # sort the items
    nums.sort()
    largest = nums[-1]
    ops = 0
    prev = -1
    reduction = 0
    if sum(nums) == 0: return 0

    # sum the items from smallest to largest and subtarct from the largest item as we go
    for i in range(len(nums)):
        val = nums[i] - reduction
        if nums[i] != 0 and nums[i] != prev:
            largest -= val
            reduction += val
            ops += 1
            if largest <= 0:
                return ops
            prev = nums[i]

# damn
def minimumOperations(self, nums: List[int]) -> int:
    if 0 in nums:
        return len(set(nums))-1
    else:
        return len(set(nums))

# Score After Flipping Matrix LeetCode Medium
# TC: O(n), SC: O(1)
# interesting problem
# : You are given an m x n binary matrix grid.: 
# : A move consists of choosing any row or column and toggling each value in that row or column (i.e., changing all 0's to 1's, and all 1's to 0's).
# : Every row of the matrix is interpreted as a binary number, and the score of the matrix is the sum of these numbers.
# : Return the highest possible score after making any number of moves (including zero moves).
def matrixScore(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    # flip rows so that the leftmost digits are all 1s
    for row in range(rows):
        if grid[row][0] == 0:
            for col in range(cols):
                grid[row][col] = int(not grid[row][col])

    # flip the columns as to produce the greatest number of ones in each column
    for col in range(cols):
        col_ones = 0
        for row in range(rows):
            if grid[row][col] == 1:
                col_ones += 1
        if rows - col_ones > col_ones: # if less ones than zeros flip the col
            for row in range(rows):
                grid[row][col] = int(not grid[row][col])

    # calculate the score
    score = 0
    for row in range(rows):
        fac = 0
        for col in range(cols-1, -1, -1):
            if grid[row][col] == 1:
                score += 2**fac
            fac += 1
    return score

# Remove K Digits LeetCode Medium - pretty tough greedy
# Given string num representing a non-negative integer num, and an integer k, return the smallest possible integer after removing k digits from num.
# https://leetcode.com/problems/remove-k-digits
# TC: O(n), SC: O(n)
#
# took a while and had to look at principles involved in solution. But makes sense after you understand them
# 1. If you need to remove all digits (k is as big as the number), just return "0" because you won't have anything left.
# 2. Go through each digit of the number and if the current digit is smaller than the last digit in your "new number," throw away the last digit until it's not bigger anymore.
# 3. Keep doing this until you've thrown away k digits or you've gone through all the digits.
# 4. If you still have to throw away more digits (k > 0) after going through all the digits, just remove them from the end.
# 5. The result is the smallest number you can make after all these operations. Handle any leading zeros.
def removeKdigits(self, num: str, k: int) -> str:
    if k == len(num): return "0"
    ans = collections.deque([])
    for c in num:
        while k and ans and ans[-1] > c:
            ans.pop()
            k -= 1
        ans.append(c)
    while k:
        ans.pop()
        k -= 1
    while ans and ans[0] == "0":
        ans.popleft()
    return "".join(ans) if ans else "0"

# Hand of Straights LeetCode Medium
# https://leetcode.com/problems/hand-of-straights/description/
# Given an integer array hand where hand[i] is the value written on the i-th card and an integer groupSize, return true if all of the cards can be arranged into groups of sizeGroupsize
# where all of the cards in the group are consecutive numbers (ex: 1,2,3)
# TC: O(n), SC: O()
# took 8 mins
# careful of edge cases, there can be duplicate numbers!
# also realized that I thought I needed to sort but we actually didn't
def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
    if len(hand)%groupSize != 0: return False
    # hand.sort() we don't need to sort!!
    counts = collections.Counter(hand)
    for card in hand:
        if counts[card] == 0:
            continue
        else:
            counts[card] -= 1
        # check if the next groupSize cards are in hand and if any are left
        for i in range(1, groupSize):
            if counts[card+i] == 0:
                return False
            else:
                counts[card+i] -= 1
    return True

# Merge Triplets to Form Target Triplet LeetCode Medium
# You are given a 2D integer array triplets, where triplets[i] = [ai, bi, ci] describes the ith triplet. You are also given an integer array target = [x, y, z] that describes the triplet you want to obtain.
# To obtain target, you may apply the following operation on triplets any number of times (possibly zero): choose any two triplets and update either triplet equal to become [max(ai, aj), max(bi, bj), max(ci, cj)]
# questions seems complex but as with many greedy problems, there is a little pattern/idea that can be used as a simple approach to solving the problem
# the question basically boils down to: given the target triplet [i,j,k], do there exist triplets containing any of the i,j,k values (in the correct indices) where the other two values in the triplet do *not exceed* the other two valules in the target?
# intuitively, this makes sense. If we want to apply the operation on a triplet that may have an i, j, or k that we want, we cannot use that triplet if any of it's values exceed our target
# took 8 mins
# my solution: https://leetcode.com/problems/merge-triplets-to-form-target-triplet/solutions/4958317/python-6-lines-simple-solution-o-n-time-o-1-space/
# TC: O(n), SC: O(1)
def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
    ans = [False, False, False]
    for i,j,k in triplets:
        if i == target[0] and j <= target[1] and k <= target[2]:    # if first value matches target and second/third do not exceed
            ans[0] = True
        if j == target[1] and i <= target[0] and k <= target[2]: # if second value matches target and first/third do not exceed
            ans[1] = True
        if k == target[2] and i <= target[0] and j <= target[1]: # if third value matches target and first/second do not exceed
            ans[2] = True
    return ans[0] and ans[1] and ans[2]

# shorter solution
def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
    ans = [False, False, False]
    for i,j,k in triplets:
        ans[0] = ans[0] or (i == target[0] and j <= target[1] and k <= target[2])
        ans[1] = ans[1] or (j == target[1] and i <= target[0] and k <= target[2])
        ans[2] = ans[2] or (k == target[2] and i <= target[0] and j <= target[1])
    return ans[0] and ans[1] and ans[2]

# Partition Labels LeetCode Medium
# https://leetcode.com/problems/partition-labels/description/
# TC: O(n), SC: O(n)
# took < 15 mins
# You are given a string s. We want to partition the string into as many parts as possible so that each letter appears in at most one part.
def partitionLabels(self, s: str) -> List[int]:
    indices = collections.defaultdict(collections.deque)

    # create dictionary of {character: [list of indices that character appears]}
    for idx, c in enumerate(s):
        indices[c].append(idx)
    
    # iterate forward through string expanding the minimum index we must traverse to so that all occurances of all of the characters in the substr can be covered
    min_idx = substr_len = 0
    ans = []
    for idx, c in enumerate(s):
        substr_len += 1
        # if we are not at the minimum index yet (or haven't set it if a new substr) we should just check if it should be further expanded or single character can be it's own substr
        if not min_idx or idx < min_idx:
            # check if we should increase the minimum index
            if len(indices[c]) > 1:
                min_idx = max(min_idx, indices[c][-1])
                while indices[c] and indices[c][0] <= idx: # pop all lesser indices from the char: indices queue
                    indices[c].popleft()
            # else if there is no more occurances of the same digit and this is the first letter of the current substring, it can be it's own substr (single character) 
            elif not min_idx:
                ans.append(substr_len)
                substr_len = 0
        else: # if we have reached the min idx we can add the length to our ans and start a new substr
            ans.append(substr_len)
            min_idx = 0
            substr_len = 0
    return ans

# Valid Parenthesis String LeetCode Medium
# https://leetcode.com/problems/valid-parenthesis-string/
# Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid. '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".
# TC: O(n), SC: O(1)
# beats 99% runtime
# be careful of edge cases here *( notable invalid testcase
# took 20
def checkValidString(self, s: str) -> bool:
    min_open = max_open = 0
    for c in s:
        if c == "(":
            min_open += 1
            max_open += 1
        elif c == ")":
            min_open -= 1
            max_open -= 1
        elif c == "*":
            max_open += 1
            min_open -= 1
        
        # keep our values in range (note the importance of the second clause here)
        if max_open < 0:
            return False
        if min_open < 0:
            min_open = 0
    return 0 >= min_open # and 0 <= max_open'

# Minimum Cost to Hire K Workers LeetCode Hard
# https://leetcode.com/problems/minimum-cost-to-hire-k-workers/description/
# TC: O(nlogn), SC: O(n)
# todo add notes on approach when reviewed
def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
    res = float('inf')

    # for each worker, calculate their quality / wage ratio
    ratios = []
    for i in range(len(quality)):
        ratio = wage[i]/quality[i]
        ratios.append((ratio, quality[i]))
    ratios.sort(key=lambda x: x[0])
    
    heap = []
    total_quality = 0
    for ratio, q in ratios:
        total_quality += q
        heapq.heappush(heap, -q)

        if len(heap) > k:
            total_quality += heapq.heappop(heap)

        if len(heap) == k:
            res = min(res, total_quality*ratio)
    
    return res

# Minimum Increment to Make Array Unique LeetCode Medium
# https://leetcode.com/problems/minimum-increment-to-make-array-unique/description/?envType=daily-question&envId=2024-06-14
# TC: O(nlogn) (sorting) SC: O(n)
def minIncrementForUnique(self, nums: List[int]) -> int:
    nums.sort()
    nums.append(float('inf'))
    seen = set()
    moves = waiting = 0
    prev = None
    for i in range(len(nums)):
        num = nums[i]
        # increment any waiting numbers to the gaps between this num and the previous num
        if prev != None and waiting > 0:
            room = num - prev - 1
            cur = prev + 1
            while cur < num and waiting:
                moves += waiting                    
                waiting -= 1
                cur += 1
        
        # add current num to seen if not already seen
        if num not in seen:
            seen.add(num)
            prev = num
            moves += waiting
        else:
            waiting += 1
    return moves

# Maximum Distance in Arrays LeetCode Medium
# https://leetcode.com/problems/maximum-distance-in-arrays/description/
# TC: O(n) where n is the number of arrays
# SC: O(n) where n is the number of arrays
def maxDistance(self, arrays: List[List[int]]) -> int:
    largest, second_largest, smallest, second_smallest = [-float('inf'), None], [-float('inf'), None], [float('inf'), None], [float('inf'), None]

    # go through and find the largest, second largest, smallest, second smalllest values in out of all of the arrays
    for idx, arr in enumerate(arrays):
        if arr[-1] > largest[0]:
            second_largest = largest
            largest = [arr[-1], idx]
        elif arr[-1] > second_largest[0]:
            second_largest = [arr[-1], idx]
        
        if arr[0] < smallest[0]:
            second_smallest = smallest
            smallest = [arr[0], idx]
        elif arr[0] < second_smallest[0]:
            second_smallest = [arr[0], idx]

    # return the dsiatnce between the two items that are not from the same array
    # else, either use the largest and the second smallest, or the smallest and the second largest
    if largest[1] != smallest[1]:
        return largest[0] - smallest[0]
    else:
        return max(largest[0] - second_smallest[0], second_largest[0] - smallest[0])