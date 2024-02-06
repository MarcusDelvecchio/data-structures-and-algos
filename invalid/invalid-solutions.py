
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