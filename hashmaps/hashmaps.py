from collections import defaultdict
import random

# Two Sum Easy
# https://leetcode.com/problems/two-sum/
# TC: O(n), SC: O(n)
def twoSum(self, nums: List[int], target: int) -> List[int]:
    find = {}
    for i in range(len(nums)):
        if nums[i] in find:
            return [find[nums[i]], i]
        find[target - nums[i]] = i

# Replace Words LeetCode Medium (Easy)
# https://leetcode.com/problems/replace-words/description/
# TC: O(n), SC: O(n)
def replaceWords(self, dictionary: List[str], sentence: str) -> str:
    dictionary = set(dictionary)
    sentence = sentence.split()
    ans = []
    for word in sentence:
        ans.append(word)
        for i in range(len(word)):
            if word[:i+1] in dictionary:
                ans[-1] = word[:i+1]
                break
    return " ".join(ans)

# Find Common Characters LeetCode Easy
# https://leetcode.com/problems/find-common-characters/description/
# : Given a string array words, return an array of all characters that show up in all strings within the words (including duplicates). You may return the answer in any order.
# TC: O(n), SC: O(n)
def commonChars(self, words: List[str]) -> List[str]:
    # create counter from first word characters
    counts = collections.Counter(words[0])

    # iterate through the rest of the words and if they are missing characters from the counter or have less, update the counter
    for word in words:
        chars = collections.Counter(word)
        for c in counts:
            if c not in chars:
                counts[c] = -1
            elif counts[c] > chars[c]:
                counts[c] = chars[c]
    ans = []
    # create array from the counter (rather than string to avoid O(n^2) to constantly append to the end of the string
    for c in counts:
        if counts[c] == -1: continue
        ans.extend([c]*counts[c])

    # convetr the array to a string and return it
    return "".join(ans)

# Leetcode Roman to Integer Easy
# https://leetcode.com/problems/roman-to-integer/description/
# Given a roman numeral, convert it to an integer.
def romanToInt(self, s):

        def convert(c):
            if c == "I":
                return 1
            elif c == "V":
                return 5
            elif c == "X":
                return 10
            elif c == "L":
                return 50
            elif c == "C":
                return 100
            elif c == "D":
                return 500
            elif c == "M":
                return 1000

        res = 0
        i = 0
        while i < len(s):
            if i+1 < len(s) and s[i] == "I" and s[i+1] == "V":
                res += 4
                i += 2
            elif i+1 < len(s) and s[i] == "I" and s[i+1] == "X":
                res += 9
                i += 2
            elif i+1 < len(s) and s[i] == "X" and s[i+1] == "L":
                res += 40
                i += 2
            elif i+1 < len(s) and s[i] == "X" and s[i+1] == "C":
                res += 90
                i += 2
            elif i+1 < len(s) and s[i] == "C" and s[i+1] == "D":
                res += 400
                i += 2
            elif i+1 < len(s) and s[i] == "C" and s[i+1] == "M":
                res += 900
                i += 2
            else:
                res += convert(s[i])
                i += 1

        return res

# Leetcode Integer to Roman Medium
# https://leetcode.com/problems/integer-to-roman/submissions/
# Given an integer convert it to a roman numeral.
# took 19 mins
# but there is a way simpler and hash map way to do it see https://leetcode.com/problems/integer-to-roman/solutions/6274/simple-solution/
def intToRoman(self, num):
        res = []

        # divide by 1000s to get number of Ms
        M = floor(num/1000)
        num -= M*1000
        res.extend(["M"]*M)

        CM = floor(num/900)
        num -= CM*900
        res.extend(["CM"]*CM)
        

        # divide by 500s to get number of Ds
        D = floor(num/500)
        num -= D*500
        res.extend(["D"]*D)

        CD = floor(num/400)
        num -= CD*400
        res.extend(["CD"]*CD)

        # divide by 100s to get number of Cs
        C = floor(num/100)
        num -= C*100
        res.extend(["C"]*C)

        XC = floor(num/90)
        num -= XC*90
        res.extend(["XC"]*XC)

        # divide by 50 to get number of Ls
        L = floor(num/50)
        num -= L*50
        res.extend(["L"]*L)

        XL = floor(num/40)
        num -= XL*40
        res.extend(["XL"]*XL)

        # divide by 10 to get number of Xs
        X = floor(num/10)
        num -= X*10
        res.extend(["X"]*X)

        IX = floor(num/9)
        num -= IX*9
        res.extend(["IX"]*IX)

        # divide by 5 to get number of Vs
        V = floor(num/5)
        num -= V*5
        res.extend(["V"]*V)

        IV = floor(num/4)
        num -= IV*4
        res.extend(["IV"]*IV)

        # divide by 1 to get number of Is
        I = floor(num/1)
        num -= I*1
        res.extend(["I"]*I)
    
        return ''.join(res)
# notes on the problem:
# since the max sudoku length is 9, we can focus more on readability than performance, as we know our inputs
# will become large enough to cause issues
# this isn't to say performance and good code convention isn't important though
# see this very simple, readable, and modula solution
# rather than doing what I did and looping through the rows, verifying the rows at the end of each row, 
# while collecting the columns and verifying the columns at the end of every column
# while collecting the boxes and verifying them at the end of every box

# this person simply passes the board into 3 separate functions, each that verify the 3 rules separately
# https://leetcode.com/problems/valid-sudoku/solutions/15451/a-readable-python-solution/
# and to explain the 



# Leetcode #36 Valid Sudoku
# https://leetcode.com/problems/valid-sudoku/description/
# Determine if a 9 x 9 Sudoku board is valid (see question)
# this took me like 35 mins I was stuck losing my mind over the simplest issue with verifying the boxes
# and idk why i used a defaultdict in the first place i feel like I could have just used lists
from collections import defaultdict
class Solution:
    def isValidSudoku(self, board):
        cols = defaultdict(list)
        row_boxes = [[], [], []] 

        for i in range(9):
            row = []
            
            for j in range(9):
                if board[i][j] != ".":
                    cols[j].append(board[i][j])
                    row.append(board[i][j])
                    row_boxes[floor(j/3)].append(board[i][j])

                # if at the bottom rows we can validate the columns
                if i == 8 and len(cols[j]) > len(set(cols[j])):
                    return False

                # if at the last column in a row we can validate the row
                if j == 8 and len(row) > len(set(row)):
                    return False
                
                # if at last column 
                if j == 8 and i in [2,5,8]:
                    for box in row_boxes:
                        if len(box) > len(set(box)):
                            return False
                    row_boxes = [[], [], []] 
            
        return True

# same as above but with single pass
# there is also a 7 line solution here https://leetcode.com/problems/valid-sudoku/ that uses sets to compare uniqueness of values in their
# row, column and set
def isValidSudoku(self, board: List[List[str]]) -> bool:
    seen_cols = {col: {} for col in range(len(board))}
    seen_box = [{}, {}, {}]
    for r in range(len(board)):
        row = board[r]
        seen_row = {}

        # reset the boxes
        if r > 1 and r%3 == 0:
            seen_box = [{}, {}, {}]

        for i in range(len(row)):
            box = i//3
            if row[i] != "." and (row[i] in seen_row or row[i] in seen_cols[i] or row[i] in seen_box[box]):
                return False
            else:
                seen_row[row[i]] = True  # add value to seen in this row
                seen_cols[i][row[i]] = True  # add value to seen in this column
                seen_box[box][row[i]] = True
    return True

# Sudoku Solver
# too lazy to resolve rescursive issue (took too long)
class Solution:
    def solveSudoku(self, board):

        def get_box_availbilities(board, row, col):
            box_row, box_col = 3*(floor(row/3)), 3*(floor(col/3))
            
            # get the items in the current box
            box_vals = [board[x][y] for x in range(box_row, box_row + 3) for y in range(box_col, box_col + 3)]

            # return list of available item
            return [item for item in [str(x) for x in range(1,10)] if item not in box_vals]

        def get_row_availabilities(board, row):
            row_vals = [board[row][x] for x in range(9)]

            # return list of available items
            return [item for item in [str(x) for x in range(1,10)] if item not in row_vals]

        def get_col_availabilities(board, col):
            col_vals = [board[x][col] for x in range(9)]

            # return list of available items
            return [item for item in [str(x) for x in range(1,10)] if item not in col_vals]

        def updateCellAvailabilities(i, j, board, solutionDict):
            cell_box_availabilities = get_box_availbilities(board, i, j)
            cell_row_availabilities = get_row_availabilities(board, i)
            cell_col_availabilities = get_col_availabilities(board, j)
            cell_availability = [item for item in cell_col_availabilities if item in cell_row_availabilities and item in cell_box_availabilities]

            # update solution dict with items
            solutionDict[str(i) + str(j)] = cell_availability

            # if the cell only has 1 possible value apply it
            if len(cell_availability) == 1:
                board[i][j] = cell_availability[0]

                # clear the cell from the dict of potential cell values
                if str(i) + str(j) in solutionDict:
                    solutionDict[str(i) + str(j)] = []

                # update/recalculate related columns and rows 
                for ij in solutionDict.keys():
                    if int(ij[0]) == i and len(solutionDict[ij]) == 2:
                        i = int(ij[0])
                        j = int(ij[1])
                        board, solutionDict = updateCellAvailabilities(i, j, board, solutionDict)
                        break
                    elif int(ij[1]) == j and len(solutionDict[ij]) == 2:
                        i = int(ij[0])
                        j = int(ij[1])
                        board, solutionDict = updateCellAvailabilities(i, j, board, solutionDict)
                        break
                    # could recalculate box as well
            return board, solutionDict

        cells = [str(i) + str(j) for i in range(1,9) for j in range(1,9)]
        solutionDict = { key: [] for key in cells }

        restart = False
        first = True
        while True:
            continues = 0
            for i in range(9):
                for j in range(9):
                    if board[i][j] != ".":
                        continues += 1
                        if continues > 70:
                        if continues == 81:
                            return
                        continue
                    else:
                        board, solutionDict = updateCellAvailabilities(i, j, board, solutionDict)
        return
        

# Leetcode First Missing Positive
# https://leetcode.com/problems/first-missing-positive
# solved in 21:00 wasn't bad just has issues with edge cases and neg numbers  
def firstMissingPositive(self, nums):
    found = {}
    for num in nums:
        found[num] = True
    
    i = 0
    for i in range(0, len(nums) + 1):
        if i not in found and i != 0:
            return i
        
    return i + 1

                   
# Leetcode Group Anagrams Leetcode Medium
# https://leetcode.com/problems/group-anagrams/submissions/
# completed in 4.5 minutes but saw video explaining it yesterday
def groupAnagrams(self, strs):
    res = []
    groups = defaultdict(list)
    
    for word in strs:
        key = "".join(sorted(word))
        groups[key].append(word)
        
    for group in groups.keys():
        res.append(groups[group])
        
    return res

# Set Matrix Zeros Leetcode Medium
# https://leetcode.com/problems/set-matrix-zeroes/
# complete in 15 mins just had a minor issue with inconsistint length/width of the matrix
# be careful with that
def setZeroes(self, matrix):
    width = len(matrix)
    height = len(matrix[0])
    reset_rows = []
    reset_cols = []
    
    # storing all of the rows and columns that need to be reset
    for i in range(width):
        for j in  range(height):
            if matrix[i][j] == 0:
                reset_rows.append(i)
                reset_cols.append(j)
                
    # now reset all of the rows and columns that are to be reset    
    reset_rows = set(reset_rows)
    reset_cols = set(reset_cols)
    for i in range(width):
        for j in range(height):
            if i in reset_rows or j in reset_cols:
                matrix[i][j] = 0


# Minimum Window Substring LeetCode Hard
# https://leetcode.com/problems/minimum-window-substring/
# Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window
# took 2:20 holy hell. Thought it would be easy at the beginning too. Used all hints but got through it
from collections import defaultdict
def minWindow(self, s, t):
    back = 0
    front = 0
    
    best_start = None
    best_end = None
    best = 0
    
    # create dict of values in t and their occurances
    finds = {x: 0 for x in t}
    for char in t:
        finds[char] += 1
    
    # move front pointer forward until all items are inside the window
    success = False
    for front in range(len(s)):
        char = s[front]
        if char in t:
            finds[char] -= 1
            
            # check if all found
            all_found = True
            for key in finds.keys():
                if finds[key] > 0:
                    all_found = False
                    break
            if all_found:
                success = True
                break
                
    # if no max distance then the string is not contained so return empty
    if not success:
        return ""
    else:
        best_start = back
        best_end = front
        best = front-back
    
    # if only one char just return it (won't get any better)
    if front == back:
        return s[front]
    
    # now move left pointer forward as much as possible while all items are in the window
    while back + 1 < len(s) and back < front:
        char = s[back]
        
        # if we can't move the back forward just break
        if char in t and finds[char] + 1 > 0:
            break
            
        # else move back forward
        else:
            if char in t:
                finds[char] += 1
            back += 1
            best_start += 1
            best -= 1
            
    # now move right pointer forward again
    while front + 1 < len(s):
        front += 1
        char = s[front]
        
        # if we found the char that the back is stuck on then we can move back forward again
        if char == s[back]:
            
            while back + 1 < len(s) and back < front:
                back += 1
                back_char = s[back]
                
                if front - back < best:
                    best_start = back
                    best_end = front
                    best = front - back
                    
                if back_char == 'c':

                if back_char in t and finds[back_char] + 1 > 0:
                    break
                else:
                    if back_char in t:
                        finds[back_char] += 1
        elif char in t:
            finds[char] -= 1
    return s[best_start:best_end+1]

# LeetCode Longest Consecutive Sequence Medium
# https://leetcode.com/problems/longest-consecutive-sequence/
# took 20 mins there is a minor gotcha to imporve efficiency
def longestConsecutive(self, nums):
    if len(nums) == 0:
        return 0

    # sort the items from greatest to least so that we can implement logic to improve efficiency
    nums.sort()

    items = {}
    for num in nums:
        items[num] = ""
    
    bestStart = nums[0]
    bestLen = 1
    # loop through and find largest sequence
    for num in nums:
        current = 1
        next = num+1
        
        # if the number is within bestLen of bestStart, then we can continue (because it will have alrerady been included in the sequence of bestStart)
        if abs(bestStart - num) + 1 < bestLen:
            continue
        
        while(next in items):
            current += 1
            if current > bestLen:
                bestStart = num
                bestLen = current
            next += 1
                    
    return bestLen


# LeetCode Max Points on a Line Hard
# https://leetcode.com/problems/max-points-on-a-line/
# Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, return the maximum number of points that lie on the same straight line.
# took me an hour an 50 mins
def maxPoints(self, points):
    lines = defaultdict(set[list])
    
    # draw lines from every point to every point
    for point in points:
        for secondPoint in points:
            
            # get the slope between two points
            rise = point[1] - secondPoint[1]
            run = point[0] - secondPoint[0]
            
            slope = 0
            if not rise:
                slope = 0
            elif run and rise:
                slope = rise/run
            else: # rise and no run
                slope = None
            
            # get the y-intercept
            if slope == 0:
                b = point[1]
            elif not slope:
                b = point[0]                  
            elif not point[1]:
                b = secondPoint[1] - (slope*secondPoint[0])
            else:
                b = point[1] - (slope*point[0])
            
            # add the points and z intercept to the lines dictionary
            lines[tuple([slope, b])].add(tuple(point))
            lines[tuple([slope, b])].add(tuple(secondPoint))
    
    # return the slope/b key (line) with the greatest number of items
    greatest = 0
    for key, value in lines.items():
        if len(value) > greatest:
            greatest = len(value)
    return greatest

# LeetCode 815. Bus Routes Hard
# https://leetcode.com/problems/bus-routes/
# took 55 mins hour. Wasn't bad, interesting question and solution. Had to add simple logic for performance by implementing hashmap for lookup rather than array
def numBusesToDestination(self, routes, source, target):
    if source == target:
        return 0
    
    # convert routes to hash maps so that traversing is much faster
    hashedRoutes = []
    for i in range(len(routes)):
        routeAsHash = {}
        for stop in routes[i]:
            routeAsHash[stop] = True
        hashedRoutes.append(routeAsHash)
        
    
    busesAway = 0
    nodesAway = defaultdict(set)
    nodesAway[0] = set([target])
    
    addedNodes = {}
    # find the nodes that are 1 away from the current set of nodes and add them to a new key being 1 more away
    while busesAway in nodesAway:
        for node in nodesAway[busesAway]:
            for route in hashedRoutes:
                # if the node in the current set appears in another set, add all of those items to nodesAway+1 (bc they are one more away)
                if node in route:
                    # add nodes that have not yet been added to other sets (if they are in other sets they are closer)
                    # and if we are adding them, mark them as added
                    for stop in route.keys():
                        if stop not in addedNodes:
                            addedNodes[stop] = True
                            nodesAway[busesAway + 1].add(stop)
        busesAway += 1
        
    # find the fastest route from the starting point
    nodesConnectedToStart = set()
    for route in routes:
        if source in route:
            nodesConnectedToStart.update(route)
            
    # now that we have all of the nodes connected to the start, we can determine which node is the least nodesAway
    closest = None
    for node in nodesConnectedToStart:
        for key in nodesAway.keys():
            if node in nodesAway[key]:
                if closest == None or key < closest:
                    closest = key
                break
            
    if closest == None:
            return -1
    return closest + 1

# LeetCode Recover Array Hard
# https://leetcode.com/problems/recover-the-original-array/submissions/
# took about 1:30:00. Thought I could do it in like 45 but then got stuck on issues and gotcha with finding a k but it being invalid
def recoverArray(self, nums):
    nums_sorted = sorted(nums)
    
    isAbove = {}
    positives = 0
    # create hasmap of distance from every item to every other item
    map = defaultdict(set)
    for num in nums:
        for otherNum in nums:
            if num != otherNum:
                map[num].add(num - otherNum)
    
    # loop through all of the sets of distances between num and other nums and find the k value that exists in every set
    k = None
    for distance in map[list(map.keys())[0]]:
        success = True
        for num in map.keys():
            if not +distance in map[num] and not -distance in map[num]:
                success = False
                break
        if success and distance%2 == 0:
            k = abs(int(distance/2))
    
            # with this potential k try to compose the original array by continuously using the lowest value
            res = []
            newNums = nums_sorted.copy()
            index = 0
            is_valid = True
            while index < len(newNums):
                res.append(newNums[index] + k)
                if newNums[index] + 2*k in newNums:
                    newNums.remove(newNums[index] + 2*k)
                    
                # if the corresponding higher[i] cannot be found for a value then the k is invalid
                else:
                    is_valid = False
                    break  
                index += 1
            
            # if we make it all the way through the array then the solution is valid and we can return res
            if is_valid:
                return res

# LeetCode Daily Jan 16th - Insert Delete GetRandom O(1) Medium
# Implement the RandomizedSet class where RandomizedSet() Initializes the RandomizedSet object with remove(), insert() and getRandom() functions to interact with the set.
# All of the class methods should perform their functions in O(1) time
# https://leetcode.com/problems/insert-delete-getrandom-o1/description/?envType=daily-question&envId=2024-01-16
# this actually took me a long time because AGAIN i had an issue where i had a 'not val' check at the top of the remove and add methods that were breaking the functionality
# when they tried to insert 0. SO took me like 45 mins. But initial implementation was correct other than that, taking 15 mins
# TC: O(1) insert, delete, getRandom
class RandomizedSet:

    def __init__(self):
        self.random_set = {}
        self.idx_key_mapping = {}
        self.key_idx_mapping = {}

    def insert(self, val: int) -> bool:

        if val not in self.random_set:
            self.random_set[val] = True

            # update key mapping
            index = len(self.random_set.keys()) - 1
            self.idx_key_mapping[index] = val
            self.key_idx_mapping[val] = index
            return True
        return False

    def remove(self, val: int) -> bool:

        if val not in self.random_set:
            return False

        # remove the item from the set
        self.random_set.pop(val)

        # get indices of item being removed and the (last) item in the map that will replace it
        index_of_key = self.key_idx_mapping[val]
        index_of_last_key = len(self.idx_key_mapping.keys()) - 1

        # if there is only 1 item remove it from the mapping and simply return
        if index_of_key == index_of_last_key:
            self.key_idx_mapping.pop(val)
            self.idx_key_mapping.pop(index_of_key)
            return True

        # replace the current index of the item being removed with the item at the last index
        last_item = self.idx_key_mapping[index_of_last_key]
        self.idx_key_mapping[index_of_key] = last_item
        self.idx_key_mapping.pop(index_of_last_key)

        # replace the value at the index-value map 
        self.key_idx_mapping[self.idx_key_mapping[index_of_key]] = index_of_key
        self.key_idx_mapping.pop(val)
        return True

    def getRandom(self) -> int:
        if len(self.random_set.keys()) == 1: 
            return self.idx_key_mapping[0]

        idx = random.randint(0, len(self.random_set.keys()) - 1)
        return self.idx_key_mapping[idx]


# Group Anagrams LeetCode Medium
# https://leetcode.com/problems/group-anagrams/solutions/
# took like 7 mins because I couldn't figure out how to use a Counter
# as a dictionary key
# TC: O(n), SC: O(n)
from collections import Counter
from collections import defaultdict
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams, res = defaultdict(list), []
        for s in strs:
            key = tuple(sorted(Counter(s).items()))
            anagrams[key].append(s)
        return [anagrams[group] for group in anagrams]

# Contiguous Array LeetCode Medium
# this is really a hashmap problem
# https://leetcode.com/problems/contiguous-array/submissions/1205754849/?envType=daily-question&envId=2024-03-16
# TC: O(n), SC: O(n)
# took a while, had to watch video. Considered DP, sliding window etc, didn't think of the solution with the hashmap
def findMaxLength(self, nums: List[int]) -> int:
    least_dif = {} # map of { difference: earliest idx that dif occurs } -> we will use this map so that at every idx we check if there exists an index with that difference that we can just 'cut off', but we also want it to be as early as possible
    longest = 0
    prev_dif = 0 # keeping track of the total difference in 1s and 0s as we go along. We only ever need the difference from the index before
    for i in range(len(nums)):
        diff = prev_dif + (1 if nums[i] == 1 else -1)
        if diff not in least_dif: least_dif[diff] = i # only add the current difference to the least_diff map if there is no earlier element with the same difference. Otherwise we would want to cut off the earlier element
        if diff == 0:
            longest = i+1
        elif diff in least_dif: # given the difference in 1s and 0s at our current element, if there exists a subarray that we can cut off that has the same difference, check if it is the new longest if we were to do so
            longest = max(longest, i - least_dif[diff])
        prev_dif = diff
    return longest