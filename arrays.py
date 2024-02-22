
# Find the Town Judge LeetCode Easy
# daily problem  feb 22
# took like 5 mins
def findJudge(self, n: int, trust: List[List[int]]) -> int:
    if n == 1: return 1
    trusted_by, trusts, candidates = defaultdict(int), defaultdict(int), []
    for t in trust:
        trusted_by[t[1]] += 1
        trusts[t[0]] += 1

        if trusted_by[t[1]] == n - 1:
            candidates.append(t[1])
    for can in candidates:
        if not trusts[can]:
            return can
    return -1

# took about an hour but O(n+m) not not true Hard solution
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    p1, p2 = 0, 0 

    new = []
    length = 0
    even = (len(nums1) + len(nums2))%2 == 0
    while length < floor((len(nums1) + len(nums2))/2 + 1):
        if p1 == len(nums1):
            if p2 == len(nums2):
                break
            new.append(nums2[p2])
            p2 += 1
        elif p2 == len(nums2) or nums1[p1] < nums2[p2]:
            new.append(nums1[p1])
            p1 += 1
        else:
            new.append(nums2[p2])
            p2 += 1
        length += 1
        
    if even:
        return (new[len(new) - 2] + new[len(new) - 1]) / 2
    else:
        return new[-1] 

# Trapping Rain Water LeetCode Hard
# https://leetcode.com/problems/trapping-rain-water/description/
# took 34:00 mins
# wish the solution was cleaner but is what it is
def trap(self, height: List[int]) -> int:
    w, current, temp = 0, 0, 0
    cont = False

    # put all vals into a dict
    vals = defaultdict(int)
    for value in height:
        vals[value] += 1

    for i in range(len(height)):
        if cont or height[i] >= current:
            cont = False
            w += temp

            # look ahead to find if a value exists that is greater or equal to the current val
            j, found = i+1, False
            while j < len(height):
                if height[j] >= height[i]:
                    found = True
                    break
                j += 1
            if found:
                current = height[i]
            else:
                t = height[i] - 1
                while t > 0:
                    if t in vals and vals[t]:
                        current = t
                        break
                    t -= 1
                if t == 0:
                    cont = True
            temp = 0
        else:
            temp += current - height[i]
        vals[height[i]] -= 1
    return w

# note on "amortized" complexity of appending to array
# That means the whole operation of pushing n objects onto the list is O(n). If we amortize that per element, it's O(n)/n = O(1).
# https://stackoverflow.com/questions/33044883/why-is-the-time-complexity-of-pythons-list-append-method-o1

# LeetCode daily Jan 15th - Find Players With Zero or One Losses Medium
# You are given an integer array matches where matches[i] = [winner i, loser i] indicates that the player winner i defeated player loser i in a match. Return is a list of all players that have not lost any matches and a list of all players that have lost exactly one match.
# https://leetcode.com/problems/find-players-with-zero-or-one-losses/description/?envType=daily-question&envId=2024-01-15
# TC: O(nlogn) SC: O(n)
def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
    winners, losers, losses, used = [], [], defaultdict(int), {}

    # get arrays of winners and losers along with a dict for losers and thei number of losses
    for match in matches:
        winners.append(match[0])
        losers.append(match[1])
        losses[match[1]] += 1
    
    # sort the lists
    winners.sort()
    losers.sort()

    # collect the winners that are not in losers
    zero_losses = []
    for winner in winners:
        if winner not in losses and winner not in used:
            zero_losses.append(winner)
            used[winner] = True

    single_losses = [loser for loser in losers if losses[loser] == 1]
    return [zero_losses, single_losses]

# LeetCode Daily Jan 22 Set Mismatch Easy
# https://leetcode.com/problems/set-mismatch/description/?envType=daily-question&envId=2024-01-22
# given a set of len n that original contains nums from 1 to n, but has one num replaced with another num between 1 and n
# return the num that has a duplicate and the num that is missing
def findErrorNums(self, nums: List[int]) -> List[int]:
    vals, duplicate, missing = Counter(nums), None, None

    for i in range(1, len(nums) + 1):
        if i not in vals:
            missing = i
        if vals[i] == 2:
            duplicate = i 

    return [duplicate, missing]

# Best Time to Buy and Sell Stock LeetCode Easy
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
# thought this was going to be a O(n^2) solution because of the nature of needing to compare array elements with each other
# but didn't realize this was not the case because of the forward-moving solution, as elements only ever need to compare themselves with ones in front
# I actually had to watch a video for this though because I hadn't done 2 pointer solutions in a bit and was looking for a dynamic programming solution
# TC: O(n), SC: O(1)
def maxProfit(self, prices: List[int]) -> int:
    left, right, maxx = 0, 1, 0
    while right < len(prices):
        maxx = max(maxx, prices[right]-prices[left])
        if prices[right] < prices[left]:
            left = right
        right += 1
    return maxx

# ugly solution ugly question I wouldn't even bother
# Divide Array Into Arrays With Max Difference LeetCode Medium
# https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/description/?envType=daily-question&envId=2024-02-01
def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
    groups, prev, res = [], None, []

    # sort the array (nlogn)
    nums.sort()

    # divide the array into groups with values all within k
    group = [nums[0]]
    for i in range(1, len(nums)):
        if nums[i] - nums[i-1] > k:
            print(nums[i], group[0], k)
            groups.append(group)
            group = [nums[i]]
        else:
            group.append(nums[i])
        
    if group: groups.append(group)

    # break the groups up into arrays of size 3
    for group in groups:
        curr = []
        for num in group:
            if len(curr) < 3:
                curr.append(num)
            else:
                res.append(curr)
                curr = [num]
        
        if len(curr) == 3:
            res.append(curr)

    # validate the group - check that all groups have length 3, all values are within k, and all items have been included in a group
    total = 0
    for group in res:
        total += len(group)
        if len(group) != 3:
            return []
        for num in group:
            if num - group[0] > k: return []
    if total != len(nums):
        return []
    return res

# Find Polygon With the Largest Perimiter LeetCode Medium
# https://leetcode.com/problems/find-polygon-with-the-largest-perimeter/description/?envType=daily-question&envId=2024-02-15
# took 6 mins
# TC: O(n), SC: O(1)
def largestPerimeter(self, nums: List[int]) -> int:
    nums.sort()
    total, curr, best = 0, 0, 0
    while curr < len(nums):
        if nums[curr] < total:
            best = total + nums[curr]
        total += nums[curr]
        curr += 1
    return best or -1

# Rearrange Array Elements by Sign LeetCode Medium
# took like 4 mins because was trying to come up with SC: O(1) solution but this would be pretty merked
# TC: O(n), SC: O(n)
def rearrangeArray(self, nums: List[int]) -> List[int]:
    positives, negatives, res = [], [], []
    for num in nums:
        if num < 0:
            negatives.append(num)
        else:
            positives.append(num)
    pos_p, neg_p, pos = 0,0, True
    for _ in range(len(nums)):
        if pos:
            res.append(positives[pos_p])
            pos_p += 1
        else:
            res.append(negatives[neg_p])
            neg_p += 1
        pos = not pos
    return res