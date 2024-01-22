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