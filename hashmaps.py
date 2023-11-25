# Leetcode Roman to Integer Easy
# https://leetcode.com/problems/roman-to-integer/description/
# Given a roman numeral, convert it to an integer.
def romanToInt(self, s: str) -> int:

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
def intToRoman(self, num: int) -> str:
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
    def isValidSudoku(self, board: List[List[str]]) -> bool:
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
                    print("column invalid returning false")
                    return False

                # if at the last column in a row we can validate the row
                if j == 8 and len(row) > len(set(row)):
                    print("row invalid returning false")
                    return False
                
                # if at last column 
                if j == 8 and i in [2,5,8]:
                    for box in row_boxes:
                        if len(box) > len(set(box)):
                            print("box invalid returning false")
                            return False
                    row_boxes = [[], [], []] 
            
        return True

# Sudoku Solver
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:

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
                        print("here")
                        i = int(ij[0])
                        j = int(ij[1])
                        board, solutionDict = updateCellAvailabilities(i, j, board, solutionDict)
                        break
                    elif int(ij[1]) == j and len(solutionDict[ij]) == 2:
                        print("here")
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
                    # print(i,j)
                    if board[i][j] != ".":
                        continues += 1
                        if continues > 70:
                            # print
                            print(board)
                        if continues == 81:
                            print(board)
                            print("RETURNING")
                            return
                        continue
                    else:
                        print('here')
                        board, solutionDict = updateCellAvailabilities(i, j, board, solutionDict)
        return
        

# Leetcode First Missing Positive
# https://leetcode.com/problems/first-missing-positive
# solved in 21:00 wasn't bad just has issues with edge cases and neg numbers  
def firstMissingPositive(self, nums: List[int]) -> int:
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
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
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
def setZeroes(self, matrix: List[List[int]]) -> None:
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
                    print(finds)

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
def longestConsecutive(self, nums: List[int]) -> int:
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