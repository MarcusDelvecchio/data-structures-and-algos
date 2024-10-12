
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

# Critical Connections in a Network LeetCode Hard
# TLE - input i snodes <= 10,000 so should have noticed we cannot use O(n^2) solution (for DFS it is O((E + V)^2))
def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
    critical = []
    neighbors = collections.defaultdict(list)
    for src, tar in connections:
        neighbors[src].append(tar)
        neighbors[tar].append(src)

    for i in range(len(connections)):
        visited = set()
        nodes = [0]
        while nodes:
            node = nodes.pop()
            if node in visited: continue
            visited.add(node)
            for neighbor in neighbors[node]:
                if [node, neighbor] != connections[i] and [neighbor, node] != connections[i]:\
                    nodes.append(neighbor)
        # check that every node can be visited
        for node in range(n):
            if node not in visited:
                critical.append(connections[i])
                break
    
    return critical

# Find the Running Median HackerRank Hard
# https://www.hackerrank.com/challenges/find-the-running-median/problem
# this is a O(n*nlogn) solution that does binary search to find the place tyo insert each int into the array and then finds the median
# but shifting the array after each insert causes extreme inefficiencies and the solution goes from being O(nlogn) to (n^2)logn
# Complete the 'runningMedian' function below.
# The function is expected to return a DOUBLE_ARRAY.
# The function accepts INTEGER_ARRAY a as parameter.
def addItemToList(items, mid, i):
    if items[mid] <= i:
        items = items[:mid+1] + [i] + items[mid+1:]
    else:
        items = items[:mid] + [i] + items[mid:]
    return items

def runningMedian(a):
    ans = []
    items = []
    for i in a:
        left, right = 0, len(items)-1
        mid_idx = 0
        while left < right:
            mid_idx = (left+right)//2
            if items[mid_idx] < i:
                left = mid_idx + 1
            elif items[mid_idx] > i:
                right = mid_idx - 1
            else:
                # items[mid_idx] == i
                break
        if len(items) >= 1:
            items = addItemToList(items, (left+right)//2, i)
        else:
            items = [i]
        
        # add median to ans
        mid = len(items)//2
        ans.append(float(items[mid] if len(items) % 2 == 1 else (items[mid] + items[mid-1])/2))
    return ans


# approach: calculate all possible substrings (2^n) and if any are in forbidden, mark that portion of the string as unusable
# at the end we just find the largest space between all portions of the substring marked as unusable
# NOTE that we do not need to do recursion for substrings and subarrays, only subsequences
def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
    forbidden = set(forbidden)

    # find all intervals (substring indices) that contain forbidden words
    forbidden_intervals = []
    for start in range(len(word)):
        for end in range(len(word)):
            if word[start:end+1] in forbidden:
                forbidden_intervals.append([start, end])
    
    # traverse our string and find the largest interval we can make without encapsulating an entire interval
    # NOTE that our forbidden intervals are already sorted by start and if they have the start they are also sorted by end (increasing)
    # so a sliding window expanding while adding the start of any interval to our interval
    # but if ever hit the end of any interval that have included, we need to shrink until we remove that interval's start
    # create a map of 'starts' end 'ends' that map index: intervalNumber
    # when we hit an index that is the open index of a interval, we add it to our open intervals
    # if we hit an index that closes an already open interval, we have to close our window
    closes = collections.defaultdict(collections.deque)
    for interval_number, interval in enumerate(forbidden_intervals):
        closes[interval[1]].append(interval[0])


    # do sliding window
    right = 0
    ans = 0
    cur_interval = 0
    left = 0
    while right < len(word):
        if left > right: continue
        
        # move left forward while we encapsulate an entire interval
        while right in closes and closes[right] and closes[right][0] < left:
            closes[right].popleft()

        while right in closes and closes[right]:
            left = closes[right][0] + 1
            closes[right].popleft()
        if right - left + 1 > ans:
            ans = max(ans, right - left + 1)
        right += 1
    return ans     


# https://leetcode.com/problems/snakes-and-ladders/
# invalid solution took over an hour when getting back into things
# fell for the initial tempations to use tabulation, then tried DFS (why) and then realized BFS is best approach
# todo another time do BFS
class Solution:
    # rememeber: why we cannot use tabulation for this question, even though you might think we can
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)
        total_cells = n*n

        cell_coords = dict()
        fwd = True
        cell_num = 1
        for r in range(n):
            start = 0 if fwd else n - 1
            end = n if fwd else - 1
            step = 1 if fwd else - 1
            for c in range(start, end, step): # 0, n, 1
                cell_coords[cell_num] = (n-r-1, c)
                cell_num += 1     
            fwd = not fwd  
        
        memo = dict()

        # flag cells where whether or not a snake / ladder was recently used doesn't matter because there are no further snakes / ladders in range
        usedSnakeOrLadderDoesntMatter = set()
        spaces_since_ladder_or_snake = float('inf')
        for i in range(total_cells, 0, -1):
            x, y = cell_coords[i]
            if board[x][y] != -1:
                spaces_since_ladder_or_snake = 0
            else:
                spaces_since_ladder_or_snake += 1
            if spaces_since_ladder_or_snake > 6:
                usedSnakeOrLadderDoesntMatter.add(i)

        
        def getShortestPath(cell, usedSnakeOrLadder, seen):
            if cell >= total_cells: return 0
            if cell in seen: return float('inf')
            if (cell, usedSnakeOrLadder) in memo:
                return memo[cell, usedSnakeOrLadder]
            if abs(cell - total_cells) <= 6 and cell_coords[cell] == -1:
                return 1
            seen.add(cell)

            # if this cell has a snake or ladder, we cannot stop on this cell
            x, y = cell_coords[cell]
            shortest = 1 if cell_coords[cell] == -1 and abs(cell - total_cells) <= 6 else float('inf')
            if board[x][y] != -1 and not usedSnakeOrLadder:
                shortest = min(shortest, getShortestPath(board[x][y], True, seen))
            else:
                # explore the next 6 cells
                for i in range(1, 7):
                    shortest = min(shortest, getShortestPath(cell + i, False, seen) + 1)
            memo[cell, usedSnakeOrLadder] = shortest
            seen.remove(cell)
            return shortest

        
        ans = getShortestPath(1, False, set())
        return ans if ans != float('inf') else -1