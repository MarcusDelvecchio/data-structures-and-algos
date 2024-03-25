# Number of Islands LeetCode Medium
# https://leetcode.com/problems/number-of-islands/
# took 6 mins
# TC: O(n) SC: (n)
def numIslands(self, grid: List[List[str]]) -> int:
    rows, cols = len(grid), len(grid[0])
    visited = set()
    res = 0

    def dfs(row, col):
        if (row, col) in visited or row < 0 or row > rows - 1 or col < 0 or col > cols - 1 or grid[row][col] != "1": return
        visited.add((row, col))
        dfs(row+1, col)
        dfs(row-1, col)
        dfs(row, col+1)
        dfs(row, col-1)

    for row in range(rows):
        for col in range(cols):
            if (row, col) not in visited and grid[row][col] == "1":
                res += 1
                dfs(row, col)

    return res

# Clone Graph LeetCode Medium
# https://leetcode.com/problems/clone-graph
# TC: O(n), SC: O(n)
# my solution here: https://leetcode.com/problems/clone-graph/solutions/4915896/python-simple-bfs-12-lines/
# Took over an hour because debugging issue with cloning neighbors when creating cloned node. Otherwise, solution was implemented in like 15 mins.
def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
    if not node: return None
    copies = { node.val: Node(node.val, [nei for nei in node.neighbors])} # without [nei for nei] solution doesn't work
    q = collections.deque([copies[node.val]])

    while q:
        removed = q.popleft()
        for idx, n in enumerate(removed.neighbors):
            if n.val not in copies:
                copies[n.val] = Node(n.val, [nei for nei in n.neighbors]) # without [nei for nei] solution doesn't work
                q.append(copies[n.val])
            removed.neighbors[idx] = copies[n.val]
    return copies[node.val]

# Max Area of Island LeetCode Medium
# https://leetcode.com/problems/max-area-of-island
# TC: O(n), SC: O(n)
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    visited, maxArea = set(), 0

    def dfs(row, col):
        if (row, col) in visited or row == rows or row == -1 or col == -1 or col == cols or grid[row][col] == 0: return 0
        visited.add((row, col))
        return dfs(row+1, col) + dfs(row-1, col) + dfs(row, col-1) + dfs(row, col+1) + 1

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1 and (row, col) not in visited:
                maxArea = max(maxArea, dfs(row, col))
    return maxArea

# Pacific Atlantic Water Flow LeetCode Medium
# invalid solution
def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    rows, cols = len(heights), len(heights[0])
    memo = {}

    def dfs(parent, row, col):
        if row == -1 or col == -1 or row == rows or col == cols:
            return row == -1 or col == -1, row == rows or col == cols
        if heights[row][col] > parent: return False, False
        if (row, col) in memo: return memo[(row, col)]
        memo[(row, col)] = False, False
        below_pac, below_atl = dfs(heights[row][col], row+1, col)
        above_pac, above_atl = dfs(heights[row][col], row-1, col)
        right_pac, right_atl = dfs(heights[row][col], row, col+1)
        left_pac, left_atl = dfs(heights[row][col], row, col-1)
        res = above_pac or below_pac or right_pac or left_pac, above_atl or below_atl or right_atl or left_atl
        memo[(row, col)] = res
        return res
    
    for row in range(rows):
        for col in range(cols):
            if (row, col) not in memo:
                dfs(float('inf'), row, col)
    return [[row, col] for col in range(cols) for row in range(rows) if memo[(row, col)][0] and memo[(row, col)][1]]

# Pacific Atlantic Water Flow LeetCode Medium
# https://leetcode.com/problems/pacific-atlantic-water-flow
# valid solution
# TC: O(n), SC: O(n)
# took like 45 mins because had invalid approach (see above)
def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    rows, cols = len(heights), len(heights[0])
    memo = {}
    res = set()

    def dfs(row, col, atl, pac):
        if (row, col, atl, pac) in memo or not atl and not pac: return  # if we have already explored this cell in this state, continue
        memo[(row, col, atl, pac)] = True  # note that we have explored this cell with this state
        atlantic_access = atl or (row, col, True, False) in memo
        pacific_access = pac or (row, col, False, True) in memo
        if atlantic_access and pacific_access:  # add cell to res if access to alt and pac
            res.add((row, col))
        for next_row in [-1, 0, 1]:  # explore adjacent cells
            for next_col in [-1, 0, 1]:
                if row+next_row < 0 or row+next_row == rows or col+next_col < 0 or col+next_col == cols: continue # ensure next cell is valid
                if heights[row+next_row][col+next_col] >= heights[row][col] and (not next_row or not next_col): # verify next cell can be traveled to
                    dfs(row+next_row, col+next_col, atlantic_access, pacific_access)
        

    for row in range(rows):
        for col in range(cols):
            dfs(row, col, row == rows - 1 or col == cols - 1, row == 0 or col == 0)
    return res