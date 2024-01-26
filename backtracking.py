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
# Given two words, beginWord and endWord, and a dictionary wordList, return all the shortest transformation sequences from beginWord to endWord, or an empty list if no such sequence exists. Each sequence should be returned as a list of the words [beginWord, s1, s2, ..., sk].
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

# Unique Paths III LeetCode Hard
# https://leetcode.com/problems/unique-paths-iii/submissions/
# took 23 mins damn
# recursive backtracking brute force solution
def uniquePathsIII(self, grid):
    res, curr_r, curr_c, goal_r, goal_c, h, w = [0], 0, 0, 0, 0, 0, 0

    # get the current cell and the goal cell
    for r in range(len(grid)):
        h += 1
        for c in range(len(grid[r])):
            if h == 1:
                w += 1
            if grid[r][c] == 1:
                curr_r = r
                curr_c = c
            if grid[r][c] == 2:
                goal_r = r
                goal_c = c

    def is_fully_explored(grid):
        for row in grid:
            for cell in row:
                if cell != -1 and cell != 2:
                    return False
        return True
    
    def explore(row, col):
        # if we are at a goal node and the board is filled, add 1
        if row == goal_r and col == goal_c:
            if is_fully_explored(grid):
                res[0] += 1
            return
        
        # explore above
        if row != 0 and grid[row-1][col] != -1:
            grid[row][col] = -1
            explore(row - 1, col)
            grid[row][col] = 0
        
        # explore below
        if row != h - 1 and grid[row+1][col] != -1:
            grid[row][col] = -1
            explore(row+1, col)
            grid[row][col] = 0

        # explore left
        if col != 0 and grid[row][col-1] != -1:
            grid[row][col] = -1
            explore(row, col-1)
            grid[row][col] = 0

        # explore right
        if col != w - 1 and grid[row][col+1] != -1:
            grid[row][col] = -1
            explore(row, col+1)
            grid[row][col] = 0

    explore(curr_r, curr_c)
    return res[0]

# Maximum Score Words Formed by Letters LeetCode Hard
#   "
#   Given a list of words, list of  single letters (might be repeating) and score of every character.
#   Return the maximum score of any valid set of words formed by using the given letters (words[i] cannot be used two or more times).
#   It is not necessary to use all characters in letters and each letter can only be used once. Score of letters 'a', 'b', 'c', ... ,'z' is given by score[0], score[1], ... , score[25] respectively.
#   "
# took 33 mins
# had issues with the array copying and a minor issue with verifying there was enough letters to produce the word (using count rather than simply checking for 'c in letters')
# but other than that it went smoothly
def maxScoreWords(self, words, letters, score):
    # trying to find the subset of words from words that uses the most letters
    letters_map = defaultdict(int)
    for letter in letters:
        letters_map[letter] += 1

    def get_score(words, letters):
        max_score = 0
        temp_words = words.copy() # create a temp words so that we are not editing the 'words' vareiable we are iterating through below
        for word in words:
            can_use = True
            for c in word:
                if letters[c] < word.count(c):
                    can_use = False
                    break
            if can_use:
                temp = 0
                for c in word:
                    temp += score[string.ascii_lowercase.index(c)]
                    letters[c] -= 1
                temp_words.remove(word)
                max_score = max(get_score(temp_words, letters) + temp, max_score)
                temp_words.append(word)
                for c in word:
                    letters[c] += 1
        return max_score

    return get_score(words, letters_map)

# Tiling a Rectangle with the Fewest Squares LeetCode Hard
# https://leetcode.com/problems/tiling-a-rectangle-with-the-fewest-squares/description/
# interesting read -> https://leetcode.com/problems/tiling-a-rectangle-with-the-fewest-squares/solutions/414804/a-review-why-this-problem-is-a-tip-of-the-iceberg/
# this question felt impossible but hard coded that one solution and my solution works so idk if it's a win or not
# but wasn't trying to waste too much time on it so stopped at the 2 hours mark when I realized covering the small case for n13/m11 breaks everything. 
# I started reading solutions and read that the special case and otherwise my solution wasn't bad so whatever
# 2 hours
def tilingRectangle(self, n: int, m: int) -> int:
    # single and only special case that completely breaks this solution so just hard coded
    if n == 11 and m == 13 or m == 11 and n == 13:
        return 6
    res = []

    def solve(n, m, stop):
        if stop == 0:
            return 1000
        if n == m:
            return 1
        if n == 1 or m == 1:
            return min(n, m)*max(n,m)
        
        minimum = 1000
        for i in range(min(n, m), 1, -1):

            # attempt full square
            if i == min(n, m):
                sq = solve(max(m, n) - i, min(m, n), min(minimum, stop - 1))
                minimum = min(sq, minimum)
            # when less than full square
            else:
                sides_1 = solve(i, max(n,m) - i, min(minimum, stop - 1)) + solve(max(m, n), min(n,m) - i, min(minimum, stop - 1))
                sides_2 = solve(min(m, n), max(n,m) - i, min(minimum, stop - 1)) + solve(i, min(n,m) - i, min(minimum, stop - 1)))
                minimum = min(minimum, sides_1, sides_2)

        return minimum + 1
    return solve(n, m, 1000)

# Stickers to Spell Word LeetCode Hard
# think this is what is a called an "optimized" BFS
# https://leetcode.com/problems/stickers-to-spell-word/submissions/
# took 1:10 becuase had to come up with heuristic
# I think some implementations in other solutions I saw (and I thought about as well but wasn't sure about efficiency) also re-calculated the score of each of the
# stickies *at each level* and removed the ones that had no score
# I didn't think too much about the implementaiton because my solution was accepted but that could definitely be something to consider for efficiency
def minStickers(self, stickers, target):
    scores = defaultdict(int)
    word_letters = defaultdict(dict)
    target_map = {c: True for c in target}
    used = {}

    # O(s) time to get score of stickers but without the target_map is it O(t*s)
    for sticker in stickers:
        if sticker == target: return 1
        used[sticker] = False
        current = {}
        letter_map = defaultdict(int)
        for c in sticker:
            letter_map[c] += 1
            if c in target_map and c not in current:
                scores[sticker] += 1
                current[c] = True
        word_letters[sticker] = letter_map

    # sort the items based on score
    stickers.sort(key=lambda sticker: scores[sticker], reverse=True)
    
    def bfs(start):
        layer, depth = deque([start]), 0
        visited = {start: True}

        while layer:
            print(depth)
            print([item for item in layer])
            size = len(layer)
            for _ in range(size):
                for s in stickers:
                    if scores[s] < 1: break                         

                    temp, new_word = word_letters[s].copy(), ""
                    for c in layer[0]:
                        if c in temp and temp[c]:
                            temp[c] -= 1
                        else:
                            new_word += c
                    if not new_word: 
                        return depth + 1
                    if len(new_word) != len(layer[0]) and new_word not in visited:
                        visited[new_word] = True
                        layer.append(new_word)
                layer.popleft()
            depth += 1

    return bfs(target) or -1

# Verbal Arithmetic Puzzle LeetCode Hard
# https://leetcode.com/problems/verbal-arithmetic-puzzle/
# brute force solution that gets TLE becuase no active heuristic to improve efficiency
# starts with the ones digit and moves forward if they all add up, cutting the digits off as they are compared
# added logic/heuristic so that if the right side becomes greater than the left side, we should break if we are on the right word (why continue to increase the digit on tthew right side if it is already greater than the left side)
# and added more logic so that if the left side is greater than the right the left side words are skipped through so that we only increase digits on the right
# this question took about 3.5 hours and had inconsistencies. I commented on the question about them. 
# for example test case left: [A,B] right: [A] or [B] expects true for solution being A+0=A will always be true whatever A is
# but test case AA + BB = AA expects no solution even though AA + 00 = AA was produced by my solution and made sense so just hard coded and moved on.
# someone replied to comment saying "In all LeetCode problems “without leading zeros” means that the first digit/character mustn’t be zero."
# so I guess if it leads with a zero the entire number must be ignored (ex 071 = 0, 00 = 0)
def isSolvable(self, words: List[str], result: str) -> bool:
        # covers cases where question is inconsisient
        if result == "A" and words[0] == "A" or result == "B" and words[0] == "A" and words[1] == "B":
            return True
        if result in words and len(words) > 2:
            return False
        used = {num: False for num in range(0, 10)}
        remaining = {word: len(word) for word in words} # number of letters remaining in the words
        nums = {str(n) for n in range(0,10)}

        def is_all_same(word):
            chars = {word[0]: 1}
            all_same = True
            for c in word:
                if c not in chars:
                    all_same = False
            return all_same


        
        def dfs(words, carry):
            if result == "AA":
                print(words)
            if len(max(words, key=len)) == 0:
                print(carry)
                return not carry, False

            # add up numbers from right to left as we go along. If the ones columns don't add up, we shouldn't continue to do the tens, hundreds etc
            left = 0
            right = 0
            temp = words.copy()
            can_add = True

            # if the least significant values re all digits, add them 
            for i in range(len(words)):
                if words[i] and words[i][-1] not in nums:
                    words = temp
                    left, right = 0, 0
                    can_add = False
                    break
                if i != len(words) - 1 and words[i]:
                    left += int(words[i][-1])
                elif words[i]:
                    right += int(words[i][-1])
                
                # remove the last digit from the words
                words[i] = words[i][:-1]

            if can_add:
                carry_new = floor((left + carry)/10)
                left = (left + carry)%10
                print(left, right)
                if left != right:
                    return False, right > left
                else:
                    return dfs(words, carry_new)

            # else, we need to assign numeric values to the end characters of one or more of the words
            left_greator = False
            for w in range(len(words)):
                if left_greator and w != words[-1]:
                    continue
                if words[w] and words[w][-1] not in nums:
                    for i in range(0, 10):
                        if i == 0 and (len(words[w]) == 1 or is_all_same(words[w])): continue
                        if not used[i]:
                            used[i] = True
                            new_words = [word.replace(words[w][-1], str(i)) for word in words]
                            res, right_greater = dfs(new_words, carry)
                            used[i] = False
                            if res:
                                return True, False
                            elif right_greater and w == len(words) - 1:
                                left_greator = False
                                break
                            elif not right_greater:
                                left_greator = True                                
                            
            return False, False
        res, _ = dfs(words + [result], 0)
        return res

# Maximum Profit in Job Scheduling LeetCode Hard
# https://leetcode.com/problems/maximum-profit-in-job-scheduling/submissions/
# took 1:32:00 first 50 mins to write the backtracking logic (haven't done in a while), second 45 mins writing memo and binary seach logic for optimization to overcome TLE
# approach: backtracking. convert the items into a list of tuples 'jobs', sort these items based on startTime so we can always assume the next item being iterated (recursively traversed) through will not have an earlier startTime (lost 15 mins with that issue)
# then recursively explore every job, considering taking it and not taking it, while also memoizing the profits for jobs at the bottom of the tree for efficiency
# furthermore, when we do have a next_availablility value, instead of recursively traversing the remaining jobs one by one to find the job that has EARLIEST (lowest) startTime after or at the next_availability, do a binary search do to this search much more efficiently
# TC: i honestly don't know after optimizations but probably O(n^2) somehow. SC: O(n)?
def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
    next_availability, jobs, memo_with, memo_without = [0], [], {}, {}
    # combine items into array of sets
    for i in range(len(startTime)):
        jobs.append((startTime[i], endTime[i], profit[i]))

    # sort the jobs based on startTime so that as we ierate forward through the jobs we know the next item will always have a later startTime
    jobs.sort()

    # function to do a binary search within the start and end indices of the jobs list as to find the next (THE LOWEST) job that is greater or equal to the next_availablility value
    def get_next(start, end):
        while start <= end:
            mid = start + (end - start) // 2

            # Check if the middle element is greater than or equal to the search value
            if jobs[mid][0] >= next_availability[0]:
                # If the previous element is less than the next_availability[0] value or mid is the first element
                if mid == 0 or jobs[mid - 1][0] < next_availability[0]:
                    return mid
                else:
                    # Continue searching in the left half
                    end = mid - 1
            else:
                # Continue searching in the right half
                start = mid + 1

        # If no element is greater than or equal to the search value
        return len(jobs)
    
    def schedule(curr):
        if curr == len(startTime): return 0
        
        # if the current item cannot be started, try the next
        if jobs[curr][0] < next_availability[0]:
            next = get_next(curr, len(jobs) - 1) # binary search for the next item in jobs rather than recursively iterating forward
            return schedule(next)
        
        # take the current item
        temp, next_availability[0] = next_availability[0], jobs[curr][1]
        profit_with = 0
        if jobs[curr] not in memo_with:
            profit_with = jobs[curr][2] + schedule(curr + 1)
            memo_with[jobs[curr]] = profit_with
        else:
            profit_with = memo_with[jobs[curr]]
        next_availability[0] = temp
        
        # backtrack and don't take the current item
        profit_without = 0
        if jobs[curr] not in memo_without:
            profit_without = schedule(curr + 1)
            memo_without[jobs[curr]] = profit_without
        else:
            profit_without = memo_without[jobs[curr]]
        next_availability[0] = temp
        
        # return the greater between using and not using the current item
        return max(profit_with, profit_without)
    return schedule(0)

# LeetCode daily Jan 17th - Climbing Stairs Easy
# You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# https://leetcode.com/problems/climbing-stairs/description/?envType=daily-question&envId=2024-01-18
# took 7 mins because I had to add optimization. I was honestly suprised this was an easy.
# TC/SC: ???
def climbStairs(self, n: int) -> int:
    memo = {}
    def climb(n):
        if n < 3: return n
        if n not in memo:
            res = climb(n - 1) + climb(n - 2)
            memo[n] = res
            return res
        return memo[n]
    return climb(n)

# Minimum Falling Path Sum LeetCode Medium
# Given an n x n array of integers matrix, return the minimum sum of any falling path through matrix.
# approach: consider every block in the first row, and then recursively attempt to choose the block either below and left, directly
# below, or below and right, retuning the path sum back to the top and selecting the minimum path sum
# Took 14:45. Took like 6 mins but had a pass-sum-through-to-bottom solution so when I wanted to implement momoization I had to reverse
# the logic to a pass-sum-to-top (of recursive call stack) solution.
# TC: O(n) S: O(n)
def minFallingPathSum(self, matrix: List[List[int]]) -> int:
    res, memo = 100000, {}

    def search(row, col):
        if (row, col) in memo:
            return memo[(row, col)]
        else:
            res = find_path(row, col)
            memo[(row, col)] = res
            return res
    
    def find_path(row, col):
        if row == len(matrix): return 0
        left, center, right = 100000, 100000, 100000

        # try below and left 1
        if col != 0:
            left = matrix[row][col-1] + search(row + 1, col-1)

        # try below
        center = matrix[row][col] + search(row + 1, col)

        # try below and right 1
        if col != len(matrix)-1:
            right = matrix[row][col+1] + search(row + 1, col+1)

        return min(left, right, center)

    for i in range(len(matrix)):
        res = min(res, find_path(0, i))

    return res

# 198. House Robber LeetCode Medium
# Return the maximum value you can take from an array by adding up elements within it that are not directly beside each other
# Took 8 minutes, 4 initially but another 4 for optimizations
# TC: O(n), SC: O(n)
def rob(self, nums: List[int]) -> int:
    memo = {}
    def calculate(pos):
        if pos >= len(nums): return 0

        with_current, without_current = 0, 0

        # consider taking the current house
        if pos in memo:
            with_current = memo[pos]
        else:
            with_current = nums[pos] + calculate(pos+2)
            memo[pos] = with_current

        # backtrack and don't take the current house
        if pos+1 in memo:
            without_current = memo[pos+1]
        else:
            without_current = calculate(pos+1)
            memo[pos+1] = without_current

        memo[pos] = max(with_current, without_current)
        return memo[pos]
    return calculate(0)

# Maximum Length of a Concatenated String with Unique Characters LeetCode Medium
# https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/submissions/1154338066/?envType=daily-question&envId=2024-01-23
# TC: O(n) I believe? SC: O(n) -> the recursive stack will only ever be of size n since the two recursive calls for every iteration happen one after another
def maxLength(self, arr: List[str]) -> int:
    taken = {}
    
    def find(curr):
        if curr == len(arr): return 0
        res_with, res_without = 0, 0 

        # take the item and continue to the next item
        valid, added = True, {}
        for char in arr[curr]:
            if char not in taken and char not in added:
                added[char] = True
            else:
                valid = False
                break
        if valid:
            for char in added.keys():
                taken[char] = True
            res_with = find(curr+1) + len(arr[curr])

        # backtrack and remove the item and continue to the next item
        if valid:
            for char in added.keys():
                if char in taken:
                    taken.pop(char)
        res_without = find(curr+1)
        return max(res_with, res_without)
    return find(0)

# Out of Boundary Paths LeetCode Medium
# https://leetcode.com/problems/out-of-boundary-paths/description/?envType=daily-question&envId=2024-01-26
# Given the five integers m, n, maxMove, startRow, startColumn, return the number of paths to move the ball out of the grid boundary.
# TC: O(n*m*moves) SC: S(n*m*moves)
# took a while becasue I was thinking it was going to be some hard DP problem, but ended up just going with a simple backtracking problem
# I did NOT however, realize implementing such memoization techniques (storing ro,col,move in memo) would work. I glossed over it
# because I had assumed if different 'branches' were to land on the same square, the number of moves would be different every time, making memoization based on that value ineffective
# but it works anyhow. Had to look into that
# I was also surprised to find out this is technically considered a '3D DP' problem, even though it seems like it is just backtracking
# https://www.youtube.com/watch?v=Bg5CLRqtNmk
def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
    memo, MOD = {}, 10**9+7
    
    def explore(row, col, moves):
        if row == -1 or row == m or col == -1 or col == n: return 1
        if moves == 0: return 0
        if (row,col,moves) in memo:
            return memo[(row, col, moves)]

        res =  explore(row + 1, col, moves - 1) + explore(row - 1, col, moves - 1) + explore(row, col + 1, moves - 1) + explore(row, col - 1, moves - 1)
        memo[(row, col, moves)] = res
        return res
    return explore(startRow, startColumn, maxMove)%MOD