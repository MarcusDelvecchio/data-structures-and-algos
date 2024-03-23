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