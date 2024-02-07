
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