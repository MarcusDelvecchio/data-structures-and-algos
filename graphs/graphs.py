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