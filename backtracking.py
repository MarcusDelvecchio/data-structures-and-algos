# backtracking interview prep notes here https://leetcode.com/problems/letter-combinations-of-a-phone-number/solutions/780232/Backtracking-Python-problems+-solutions-interview-prep/


# Binary Tree Paths
# https://leetcode.com/problems/binary-tree-paths/description/
# took like 4 mins idk how it's backtracking though
def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
    paths = []
    
    def dfs(root, p):
        if not root: return None

        p = p + ("->" if len(p) else "") + str(root.val)
        if not root.left and not root.right:
            paths.append(p)
        else:
            dfs(root.left, p)
            dfs(root.right, p)
    dfs(root, "")
    return paths

# Combination Sum LeetCode Medium 
# https://leetcode.com/problems/combination-sum/solutions/429538/general-backtracking-questions-solutions-in-python-for-reference/
# took like 13 mins
# still not sure where or what makes it specifically backtracking
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    solutions = {}
    
    def get_combos(target, combination, idx):
        for i in range(idx, len(candidates)):
            if target - candidates[i] > 0:
                target -= candidates[i]
                combination.append(candidates[i])
                get_combos(target, combination, i)

                # reset values after returning
                target += candidates[i]
                combination.pop()
            elif target - candidates[i] == 0:
                solutions[tuple(sorted(combination + [candidates[i]]))] = True
        # if have gone through all possible solutions and no more to be explored: backtrack
        return
    get_combos(target, [], 0)
    return [list(combination) for combination in solutions.keys()]

# time complexity is ??

# Combinations LeetCode Medium
# https://leetcode.com/problems/combinations/description/
# took 18 mins bc took a bit to think about the solution
# after looking into other solutions I updated the solution with the following:
# 1. at first I thought I would have to use a hash map for the res and curr values. When I would find a solution to the backtracking problem I would then sort the array of values and insert
# it into the res hash map. But I relaized there is no need to do this as all of the solutions will automatically be unique and sorted, since the indx value prevents duplicate values from being selected anyways
# see that solution here https://leetcode.com/problems/combinations/submissions/1112553976/ - but this one still uses hashmap for curr, which is also unnecessary
def combine(self, n: int, k: int) -> List[List[int]]:
    res = []

    def combos(l, curr, idx):
        # backtrack if we get a solution
        if l == k:
            res.append(curr.copy())
            return
            
        for i in range(idx, n + 1):
            curr.append(i)
            combos(l + 1, curr, i + 1)
            curr.pop()
    combos(0, [], 1)
    return res

# Subsets LeetCode Medium
# https://leetcode.com/problems/subsets/submissions/
# took 13 mins nice. Ran second try. I just forgot a 'def' beside the function definition
# interesting problem. Was just having issues thinking about the most efficient way to handle items being unique etc. and preventing duplicates
# and after reviewing the solutions realized that again, the idx prop prevents duplicates etc so see better and new solution below this one. But left here for reference
def subsets(self, nums: List[int]) -> List[List[int]]:
    res = {(): True}

    def sets(l, idx, curr, curr_hash):
        if l > len(nums):
            return
        
        for i in range(idx, len(nums)):
            if nums[i] not in curr_hash:
                curr.append(nums[i])
                res[tuple(curr)] = True
                curr_hash[nums[i]] = True
                sets(l+1, i+1, curr, curr_hash)
                del curr_hash[nums[i]]
                curr.pop()
    sets(0, 0, [], {})
    return list(res.keys())

# much cleaner, simpler. Don't need has maps because indx prevents duplicates
def subsets(self, nums: List[int]) -> List[List[int]]:
    res = []

    def sets(idx, curr):
        res.append(curr)
        for i in range(idx, len(nums)):
            sets(i+1, curr + [nums[i]])
    sets(0, [])
    return res

# and again I found a better solution - one that is more backtrack like
# is O(2^n+1) it seems 
# https://leetcode.com/problems/subsets/solutions/360873/python-backtracking/
# but unsure of how the creator would come up with that method of solving the problem
# specifically, how they said "At every level, the decision is whether to include the first element from the remaining set into the chosen set"
# which makes perfect sense and results in a perfectly generated tree
# but where is the theory on coming up with such solutions?

# Permutations LeetCode Medium
# took 7:30
# this one was easier than the ones above becuase due to the nature of permuations, you know that the solution will (can) simpy be O(n^2)
# because all the permuations should have the same length as the nums input and any num in nums can appear anywhere, so we have to consider all cases
# (no need to account for same-combo-different-order as we do in combinations)
def permute(self, nums: List[int]) -> List[List[int]]:
    res, used = [], {num: False for num in nums}

    def explore(path):
        if len(path) == len(nums):
            res.append(path)
            return
        
        for num in nums:
            if not used[num]:
                used[num] = True
                explore(path + [num])
                used[num] = False
    explore([])
    return res

# Sudoku Solver LeetCode Hard
# https://leetcode.com/problems/sudoku-solver/description/
# took 37 mins nice nice nice couldn't figure it out for like 3 hours last time
# using backtracking cool
def solveSudoku(self, board: List[List[str]]) -> None:
    def decisions(row, col):
        available = {str(num): True for num in range(10)}
        for num in board[row]:
            if num != ".":
                available[num] = False

        for r in board:
            if r[col] != ".":
                available[r[col]] = False
                
        for r in range(floor(row/3)*3, floor(row/3)*3+3):
            for c in range(floor(col/3)*3, floor(col/3)*3+3):
                if board[r][c] != ".":
                    available[board[r][c]] = False
        return available
        
    def solve(row, col):
        if col == 9:
            if row == 8:
                return True
            return solve(row+1, 0)
        if board[row][col] != ".":
            return solve(row, col + 1)
        available = decisions(row, col)
        for i in range(1, 10):
            if available[str(i)]:
                board[row][col] = str(i)
                if solve(row, col+1):
                    return True
        board[row][col] = "."
        return False
    
    solve(0,0)

# N-Queens LeetCode Hard
# https://leetcode.com/problems/n-queens/solutions/4367712/n-queens-python-recursive-backtracking-o-2-n/
# took 40 mins, just had to think through a method of validating diagonals that took a sec
class Solution:
    placed = 0
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        b = ["." * n]*n
        av_cols = {col: True for col in range(0, n)}
        av_rows = {row: True for row in range(0, n)}
        av_diags_neg = {row-col: True for row in range(0,n) for col in range(0, n)}
        av_diags_pos = {row+col: True for row in range(0,n) for col in range(0, n)}

        def explore(row, col):
            # if at the end of a col go to the next row
            if col == n:
                if row == n - 1:
                    if self.placed == n:
                        res.append(b)
                    return
                return explore(row + 1, 0)

            # if we can add a queen here, add a queen and explore
            if av_cols[col] and av_rows[row] and av_diags_neg[row-col] and av_diags_pos[row+col]:
                av_cols[col] = False
                av_rows[row] = False
                av_diags_neg[row-col] = False
                av_diags_pos[row+col] = False

                # skip the entire row and go to the next col
                self.placed += 1
                b[row] = b[row][:col] + "Q" + b[row][col + 1:]

                if self.placed == n:
                    res.append(b.copy())
                elif row < n - 1:
                    explore(row + 1, 0)

                # remove the queen
                self.placed -= 1
                b[row] = b[row][:col] + "." + b[row][col + 1:]
                av_cols[col] = True
                av_rows[row] = True
                av_diags_neg[row-col] = True
                av_diags_pos[row+col] = True

            # explore without adding a queen
            explore(row, col + 1)
        explore(0,0)
        return res

# Word Ladder II LeetCode Hard
# https://leetcode.com/problems/word-ladder-ii/
# INVALID SOLUTION
# I tried this for 4 hours using backtracking and dfs. I did not realize that BFS if the more correct, and efficient method of solving the problem.
def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        if endWord not in wordList:
            return []
        res, words, letters, shortest = [], {}, defaultdict(set), [500]
        distances = defaultdict(list, { endWord: [[endWord]]})
        
        # put words and letters into dicts so that they can be accessed in O(1) time
        for i in range(len(wordList)):
            words[wordList[i]] = True
            for j in range(len(beginWord)):
                letters[j].add(wordList[i][j])

        def solve(word, visited):
            # if we have already explore the shortest possible distance from this word to endWord simply return those distances
            if word in distances and distances[word]:
                return distances[word]

            for i in range(len(beginWord)):
                for letter in letters[i]:
                    if letter != word[i]:
                        new = word[:i] + letter + word[i+1:]
                        if new in words and new not in visited:
                            # add new word to visited
                            visited[new] = True

                            # solve for the shortest dsiatcnce from this new word to endWord
                            shortest_paths = solve(new, visited)

                            # remove new word from visited
                            del visited[new]

                            # update distance of current word if new word distance is shorter
                            if not shortest_paths:
                                continue
                            elif word not in distances or len(distances[word][0]) > len(shortest_paths[0]) + 1:
                                distances[word] = [[word] + p for p in shortest_paths]
                            elif word in distances and len(distances[word][0]) == len(shortest_paths[0]) + 1:
                                distances[word].extend([[word] + p for p in shortest_paths])
            
            # after trying all possible next words from this word, return the shortest distance we found from this word to the endWord
            if word in distances:
                return distances[word]
            else:
                return []

        solve(beginWord, {beginWord: True})
        return distances[beginWord]

# Word Ladder II LeetCode Hard
# not even backtracking but recursion and DFS
# took about 10 hours new record lol
# at first tried DFS for like 3 hours
# then BFS for hours 4-6 but was trying to keep track of the paths
# but you cannot do this the easier way to do this is simply return the parent before the endWord and then recursively try to 
# find the path to that parent and then to the parent before that
# so recursive DFS
# don't make the same mistake again. Do not underestimate the problem/tree size; it needs to be throroughly thought out and encorporated in the solution
# https://leetcode.com/problems/word-ladder-ii/description/
# took 10 hours
def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    if endWord not in wordList:
        return []
    
    def find_paths(start, end):
        if start == end:
            return [[start]]
            
        layer, words, letters = deque([start]), {word: True for word in wordList}, defaultdict(set)
        visited = {beginWord: True}
        parents = []
        new_found = True

        for w in wordList:
            for i in range(len(w)):
                letters[i].add(w[i])

        while not parents and new_found:
            next = set()
            new_found = False
            for word in layer:
                visited[word] = True
                for i in range(len(word)):
                    for letter in letters[i]:
                        new_word = word[:i] + letter + word[i+1:]
                        if new_word in words and new_word != word and new_word not in visited:
                            next.add(new_word)
                            new_found = True

                            if new_word == end:
                                parents.append(word)
            layer = list(next)
            
        # return the result
        res = []
        for item in parents:
            parent_paths = find_paths(start, item)
            for path in parent_paths:
                res.append(path + [end])
        return res

    return find_paths(beginWord, endWord)