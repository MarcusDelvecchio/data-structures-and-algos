
# Minimum Window Substring LeetCode Hard
# https://leetcode.com/problems/minimum-window-substring/description/
# took over 2 hours because tried DP trbulation at first but kept getting TLE (see invalid-solutions.py)
# after doing window though the solution took about 45 because of debugging issues but probably could have done in like 30
# TC: O(n), SC: O(n)
from collections import Counter
def minWindow(self, s: str, t: str) -> str:
    if s == t: return t
    remaining, p1, p2, minimum = Counter(t), 0, 0, s*2
    if s[0] in remaining: remaining[s[0]] -= 1

    # func to check if dict has any positive values (any remaining chars we need to find)
    def has_remaining(rem):
        for key in remaining: 
            if remaining[key] > 0: return True
        return False

    while p2 < len(s):
        # move right pointer right until we find a valid solution
        if has_remaining(remaining):
            p2 += 1
            if p2 < len(s) and s[p2] in remaining: 
                remaining[s[p2]] -= 1
        else:
            # if the left item cannot be be left behind, move the right forward until we find something to replace the item on the left
            if remaining[s[p1]] > 0:
                while s[p2] != s[p1]:
                    p2 += 1
                    if p2 >= len(s): return minimum
                    if s[p2] in remaining: remaining[s[p2]] -= 1

            # now move the left forward as much as possible
            while not has_remaining(remaining) and p1 <= p2:
                minimum = min(minimum, s[p1:p2+1], key=len)
                if s[p1] in remaining: remaining[s[p1]] += 1
                p1 += 1
    return minimum if len(minimum) <= len(s) else ""

# Longest Substring Without Repeating Characters LeetCode Medium
# https://leetcode.com/problems/longest-substring-without-repeating-characters/
# TC: O(n), SC: O(n)
# took 8:30
# sliding window appraoach
# approach: expand the window while the items inside are unique, then contract the window to allow it to keep expanding forward, keepign track of the maximum width it becomes
def lengthOfLongestSubstring(self, s: str) -> int:
    maxx = 0
    curr = set()
    left, right = 0, 0
    while right < len(s):
        if s[right] in curr:
            # move left forward until the value we are trying to add on the right is found
            while s[left] != s[right]:
                curr.remove(s[left])
                left += 1
            left += 1 
        else:
            curr.add(s[right])
        maxx = max(right - left + 1, maxx)
        right += 1
    return maxx

# todo cleaner approach available by doing 'for r in range(len(s))' rather than 'while right < len(s)'

# Permutation in String LeetCode Medium
# https://leetcode.com/problems/permutation-in-string/description/
# TC: O(n), SC: O(1)? Becuase we reassign s1 to a dict?
# took 18 mins
def checkInclusion(self, s1: str, s2: str) -> bool:
    target = len(s1)
    s1 = Counter(s1)
    left = 0
    for right in range(len(s2)):
        # if the right element is in s1, ensure we haven't used too many of said element
        if s2[right] in s1:

            # move left forward until that element is available again if it isn't
            while s1[s2[right]] == 0:
                if s2[left] in s1:
                    s1[s2[left]] += 1
                left += 1
            
            # check if we have new longest
            if right - left + 1 == target:
                return True

            # decrement availability of this char
            s1[s2[right]] -= 1
        
        # if the right element is not in s1 we cannot expand the window any more, so move left and right to the element after the current right
        else:
            while left <= right:
                if s2[left] in s1:
                    s1[s2[left]] += 1
                left += 1
    return False