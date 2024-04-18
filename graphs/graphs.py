# Island Perimeter LeetCode Easy
# Given a grid of cells where 1 represents land and zero water, and there is only a single island, return the perimeter of the island
# https://leetcode.com/problems/island-perimeter/description/
# TC: O(n), SC: O(n)
# took 4 mins, submitted first try
def islandPerimeter(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    perimiter = 0
    visited = set()

    def isEdge(row, col):
        return row == rows or col == cols or row < 0 or col < 0 or grid[row][col] == 0

    def explore(row, col):
        if (row, col) in visited: return
        visited.add((row, col))
        nonlocal perimiter

        if isEdge(row+1, col):
            perimiter += 1
        else:
            explore(row+1, col)

        if isEdge(row-1, col):
            perimiter += 1
        else:
            explore(row-1, col)
        
        if isEdge(row, col+1):
            perimiter += 1
        else:
            explore(row, col+1)
        
        if isEdge(row, col-1):
            perimiter += 1
        else:
            explore(row, col-1)

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1:
                explore(row, col)
                return perimiter


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

# Number of Connected Components in an Undirected Graph LeetCode Medium
# https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/
# You have a graph of n nodes. You are given an integer n and an array edges where edges[i] = [ai, bi] indicates that there is an edge between ai and bi in the graph.
# Return the number of connected components in the graph.
# took 3 mins
# TC: O(n + V) where v = vertices, n = nodes
def countComponents(self, n: int, edges: List[List[int]]) -> int:
    seen = set()
    neighbors = defaultdict(list)
    # create dict of node: neighbors list
    for edge_from, edge_to in edges:
        neighbors[edge_from].append(edge_to)
        neighbors[edge_to].append(edge_from)
    
    # explore node and neighbors until all seen
    def dfs(node):
        for neighbor in neighbors[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                dfs(neighbor)

    # try to explore all nodes. If they haven't been explored (seen) yet it must be a new "island"/component, so add 1 to ans
    ans = 0
    for node in range(n):
        if node not in seen:
            dfs(node)
            ans += 1
    return ans

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
# todo review this
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

# Surrounded Regions LeetCode Medium
# https://leetcode.com/problems/surrounded-regions
# took like 30 mins and tried to do ugly dfs approach but then realized simple trick
# TC: O(n), SC: O(n)
def solve(self, board: List[List[str]]) -> None:
    # set all squares to X if not connected to the border
    # therefore, we can find every O connected to the boarder and do dfs on them
    # all other O's connected to them will be saved, the rest will be reset
    rows, cols = len(board), len(board[0])
    saved = set()

    def dfs(row, col):
        if row == rows or row < 0 or col == cols or col < 0 or board[row][col] == "X" or (row, col) in saved: return
        saved.add((row, col))

        for r in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                if (not r or not c) and (r != c):
                    dfs(row+r, col+c)
    
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == "O" and (row == 0 or row == rows - 1 or col == cols - 1 or col == 0):
                dfs(row, col)

    for row in range(rows):
        for col in range(cols):
            if (row, col) not in saved:
                board[row][col] = "X"


# Rotting Oranges LeetCode Medium
# https://leetcode.com/problems/rotting-oranges/
# approach: BFS from initiall rotten fruits, adding neighboring fruits and tracking time. After no more neighbors are added, check if any fruits lasted (not beighboring to fruits that are rotten/going to be rotten) and return time else -1 if any fruits survived
# TC: O(n), SC: O(n)
# took 20
def orangesRotting(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])

    # gather initially rotten cells and add to bfs queue
    rotten = []
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 2:
                rotten.append((row, col))
    q = collections.deque(rotten)
    added = set(rotten)

    def isExplorable(row, col):
        return (row, col) not in added and row > -1 and row < rows and col > -1 and col < cols and grid[row][col] == 1

    # perform dfs unti no more adjacent fruits added to queue
    time = 0
    prev_size = 0
    while q:
        size = len(q)
        for _ in range(size):
            row, col = q.popleft()
            for r in [-1, 0, 1]:
                for c in [-1, 0, 1]:
                    next_row, next_col = row+r, col+c
                    if (not r or not c) and (r != c) and isExplorable(next_row, next_col):
                        q.append((next_row, next_col))
                        added.add((next_row, next_col))
        prev_size = size
        time += 1

    # check if there are any 1s remaining in the grid that have not been explored (are not neighboring rotten fruits)
    for row in range(rows):
        for col in range(cols):
            if (row, col) not in added and grid[row][col] == 1:
                return -1
    return time - 1 if time > 0 else 0

# Walls and Gates LeetCode Medium
# Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.
# https://leetcode.com/problems/walls-and-gates/description/
# approach: BFS on the initial gates. Doing DFS would be O(n^2)
# TC: O(n), SC: O(n)
# took 5 mins to come up w solution, 5 to implement and 5 to debug issues
# takeaways. this row+r, col+c caused issues because IO kept doing row+c etc. I also did "row+c > 0" instead of ">= 0" so be careful of that
def wallsAndGates(self, rooms: List[List[int]]) -> None:
    q = collections.deque([])
    rows, cols = len(rooms), len(rooms[0])

    # all all gates to the queue initially
    for row in range(rows):
        for col in range(cols):
            if rooms[row][col] == 0:
                q.append((row, col))
    
    # performs dfs adding items and updating the values as per the distance to gate
    distance = 1
    while q:
        size = len(q)
        for _ in range(size):
            row, col = q.popleft()

            # insert all neighboring cells into the queue and update their distance values
            for r in [-1, 0, 1]:
                for c in [-1, 0, 1]:
                    if (not r or not c) and (r != c) and (row+r > -1 and row+r < rows and col+c > -1 and col+c < cols) and rooms[row+r][col+c] == 2147483647:
                        rooms[row+r][col+c] = distance
                        q.append((row+r, col+c))
        distance += 1
    return

# Course Schedule LeetCode Medium
# https://leetcode.com/problems/course-schedule/
# took 19 mins. takeaways: careful with confusion with maps/dicts. Naming confused me and had to debug
# approach: BFS. Take initial courses that don't have prereqs, add then to queue, remove them from list of prerequs of other courses. If those courses have no more prerequs, queue them.
# TC: O(n), SC: O(n)
# NC does this with DFS. todo redo this problem doing dfs. seems simpler
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    # create two maps: 1. course: next courses 2. course: prerequisits
    has_prereqs = set()
    next_courses = collections.defaultdict(set)
    prereqs = collections.defaultdict(set) # turns out this can just be a number not a set of all prereqs, but it's fine
    for preReq, course in prerequisites:
        next_courses[preReq].add(course)
        prereqs[course].add(preReq)
        has_prereqs.add(course)
    
    # get initial queue of courses that don't have prereqs
    no_prereqs = [course for course in range(numCourses) if course not in has_prereqs]
    q = collections.deque(no_prereqs)
    visited = set(no_prereqs)

    # do bfs continually add courses that those courses are prerequists to
    while q:
        size = len(q)
        for _ in range(size):
            course = q.popleft()

            for next_course in next_courses[course]: # iterate through courses that the current course is a prereq of
                prereqs[next_course].remove(course) # remove this course as prereq
                if not len(prereqs[next_course]): # if there are no more prereqs, we can take/add this course
                    q.append(next_course)
                    visited.add(next_course)
    return len(visited) == numCourses

# Course Schedule II LeetCode Medium
# https://leetcode.com/problems/course-schedule-ii/description/
# "Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array."
# approach: DFS
# TC: O(n+v) (n=nodes, v=vertices), SC: O(n)
# approach: perform DFS and at every node, explore all nodes that are it's prerequisites, until nodes have no more prerequisites
# if there is a cycle, we return false
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    # get map of { course: prerequisites list }
    # O(v) - worst case though v could be n^2 - each node could be dependant on every other node. So worset case is O(n^2)
    preReq = collections.defaultdict(set)
    for pre, course in prerequisites:
        preReq[pre].add(course)

    visited = set() # dict for determining if nodes are along the current path - used to detect cycles
    added = set() # dict for determining if nodes have already been added to the ouhtput - memoizes courses that can be/are complete
    res = [] # res set (not that this)
    def dfs(course):
        if course in visited: return True
        if course in added: return
        visited.add(course)
        for pre in preReq[course]:
            if dfs(pre): return True
        visited.remove(course)
        added.add(course) # add course to added - memoize solution that we can complete it so we don't explore children again
        res.append(course) # add course to res

    # attempt to complete each node.
    # O(n+v)
    for course in range(numCourses):
        if dfs(course): return []

    return res if len(res) == numCourses else []

# Word Search LeetCode Medium
# https://leetcode.com/problems/word-search/description/
# Given an m x n grid of characters board and a string word, return true if word exists in the grid.
# took 10 mins and slower than 80%?
# TC: O(n^2), SC: O(n)
def exist(self, board: List[List[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])
    def dfs(row, col, idx, path):
        if (row, col) in path or board[row][col] != word[idx]: return False
        path.add((row, col))
        if idx == len(word)-1: return True
        for r in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                if (r or c) and (not r or not c) and row+r > -1 and row+r<rows and col+c>-1 and col+c<cols:
                    if dfs(row+r, col+c, idx+1, path):
                        return True
        path.remove((row, col))
        return False

    for row in range(rows):
        for col in range(cols):
            if dfs(row, col, 0, set()): return True
    return False