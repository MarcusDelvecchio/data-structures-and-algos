
# TLE for Minimum Window Substring LeetCode Hard
# https://leetcode.com/problems/minimum-window-substring/description/
# need to use sliding window
from collections import Counter
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) == 1 and t in s: return t
        remaining, s_len, res, smallest = Counter(t), len(s), [], float('inf')

        for col in range(s_len, -1, -1):
            dp = [None]*s_len
            dp[col] = {**remaining, "len": 1}
            if s[col] in dp[col]:
                if dp[col][s[col]] == 1: dp[col].pop(s[col])
                else: dp[col][s[col]] -= 1
            for row in range(col-1, -1, -1):
                dp[row] = dp[row+1]
                dp[row]["len"] += 1
                dp[row+1] = None
                if dp[row]["len"] > smallest: break
                if s[row] in dp[row] and dp[row][s[row]] > 0:
                    dp[row][s[row]] -= 1
                    if dp[row][s[row]] == 0: dp[row].pop(s[row])
                    if len(dp[row].keys()) == 1:
                        end = row + dp[row]["len"]
                        res.append(s[row:end])
                        smallest = min(smallest, dp[row]["len"])
                        break
        return min(res, key=len) if res else ""

# TLE for 161. One Edit Distance LeetCode Medium
# thought I would just do calculate edit-distance solution and return if edit distance == 1 but got TLE
def isOneEditDistance(self, s: str, t: str) -> bool:
    if not s or not t: return len(s) == 1 or len(t) == 1
    dp = [[0]*(len(s)+1) for _ in range(len(t)+1)]

    for r in range(len(t)):
        dp[r][-1] = len(t)-r
    for c in range(len(s)):
        dp[-1][c] = len(s)-c

    # populate 2D matrix with True/False values whether the substrings are 1 edit distance away
    for i in range(len(t)-1,-1,-1):
        seen_one_edit_dis = False
        for j in range(len(s)-1,-1,-1):
            if t[i] == s[j]:
                dp[i][j] = dp[i+1][j+1]
            else:
                dp[i][j] = 1 + min(dp[i+1][j], dp[i][j+1], dp[i+1][j+1])
    return dp[0][0] == 1

# and again a TLE by condensing the solution down into O(n) rather than O(N^2) (why would doing so reduce time complexity? idk)
#
#
def isOneEditDistance(self, s: str, t: str) -> bool:
    if not s or not t: return len(s) == 1 or len(t) == 1
    dp = [len(s)-i for i in range(len(s)+1)]

    # populate 2D matrix with True/False values whether the substrings are 1 edit distance away
    for i in range(len(t)-1,-1,-1):
        new_dp = [0]*len(s) + [len(t)-i]
        for j in range(len(s)-1,-1,-1):
            if t[i] == s[j]:
                new_dp[j] = dp[j+1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j+1], dp[j+1])
        dp = new_dp
    return dp[0] == 1

# even tried adding a check that if no new rows in the dp array (going bottom up) had a 1 or 0 edit distance we would short-circut return, but no avail
# still TLE
def isOneEditDistance(self, s: str, t: str) -> bool:
    if not s or not t: return len(s) == 1 or len(t) == 1
    dp = [len(s)-i for i in range(len(s)+1)]

    # populate 2D matrix with True/False values whether the substrings are 1 edit distance away
    for i in range(len(t)-1,-1,-1):
        new_dp = [0]*len(s) + [len(t)-i]
        seen_one_edit_dis = False
        for j in range(len(s)-1,-1,-1):
            if t[i] == s[j]:
                new_dp[j] = dp[j+1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j+1], dp[j+1])
            if new_dp[j] in [0,1]:
                seen_one_edit_dis = True
        if not seen_one_edit_dis:
            return False
        dp = new_dp
    return dp[0] == 1 

# realized cells can only be max 1 away from diagonal for edit distance to be 1
def isOneEditDistance(self, s: str, t: str) -> bool:
    if not s or not t: return len(s) == 1 or len(t) == 1
    dp = [len(s)-i for i in range(len(s)+1)]

    # populate 2D matrix with True/False values whether the substrings are 1 edit distance away
    for i in range(len(t)-1,-1,-1):
        new_dp = [0]*len(s) + [len(t)-i]
        seen_one_edit_dis = False
        for j in range(len(s)-1,-1,-1):
            if abs(i-j) > 1: continue
            if t[i] == s[j]:
                new_dp[j] = dp[j+1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j+1], dp[j+1])
        dp = new_dp
    return dp[0] == 1    

# todo finish this solution
def isOneEditDistance(self, s: str, t: str) -> bool:
    if not s or not t: return len(s) == 1 or len(t) == 1
    dp = [1, 0]

    # populate 2D matrix with True/False values whether the substrings are 1 edit distance away
    for i in range(len(t)-1, -1, -1):
        new_dp = [0]*3
        for j in range(i+1, i-2, -1):
            if j > len(s) - 1 or t[i] == s[j]:
                new_dp[j] = dp[j]
            elif j == i:                   
                new_dp[j] = 1 + dp[j]
        dp = new_dp
    return dp[0] == 1  

# Subsequence With the Minimum Score LeetCode Hard
# thought it was similar to removal/edit distance but you can only remove from one string and you have to maintain a range
# but solution not valid
def minimumScore(self, s: str, t: str) -> int:
    dp = [[(-1, -1)]*(len(s)+1) for _ in range(len(t)+1)]

    for r in range(len(t)):
        dp[r][-1] = (r, len(t)-1)
    # for c in range(len(t)):
    #     dp[-1][c] = (-1,-1)

    for i in range(len(t)-1, -1, -1):
        for j in range(len(s)-1, -1, -1):
            if s[j] == t[i]:
                dp[i][j] = dp[i+1][j+1]
            else:
                if dp[i][j+1][1] == -1:
                    dp[i][j] = (-1, -1)
                elif dp[i][j+1] == (-1, -1):
                    dp[i][j] = (-1, -1)
                elif dp[i+1][j] == (-1, -1):
                    dp[i][j] = (i, i)
                else:
                    right_dif = dp[i][j+1][1] - dp[i][j+1][0]
                    below_dif = dp[i+1][j][1] - i
                    dp[i][j] = (i, dp[i+1][j][1]) if below_dif < right_dif else dp[i][j+1]
    return dp[0][0][1] - dp[0][0][0] + 1