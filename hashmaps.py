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
from collections import defaultdict
class Solution:
    available = ['1','2', '3', '4', '5', '6', '7', '8', '9']
    def get_box_availbilities(board, row, col):
        box_row, box_col = 3*(floor(row/3)), 3*(floor(col/3))
        
        # get the items in the current box
        box_vals = [board[x][y] for x in range(box_row, box_row + 3) for y in range(box_col, box_col + 3)]

        # return list of available item
        return [item for item in Solution.available if item not in box_vals]

    def get_row_availabilities(board, row):
        row_vals = [board[row][x] for x in range(9)]

        # return list of available items
        return [item for item in Solution.available if item not in row_vals]

    def get_col_availabilities(board, col):
        col_vals = [board[x][col] for x in range(9)]

        # return list of available items
        return [item for item in Solution.available if item not in col_vals]

    def solveSudoku(self, board: List[List[str]]) -> None:
        # solutionDict = defaultdict(list)

        restart = False
        first = True
        while True:
            continues = 0
            for i in range(9):
                for j in range(9):
                    if board[i][j] != ".":
                        continues += 1
                        if continues == 81:
                            return
                        continue

                    cell_box_availabilities = Solution.get_box_availbilities(board, i, j)
                    cell_row_availabilities = Solution.get_row_availabilities(board, i)
                    cell_col_availabilities = Solution.get_col_availabilities(board, j)
                    cell_availability = [item for item in cell_col_availabilities if item in cell_row_availabilities and item in cell_box_availabilities]

                    # print(cell_availability)
                    if len(cell_availability) == 1:
                        board[i][j] = cell_availability[0]
                        print("CERTAIN FOUND RESTARTING")
                        # reset 
                        restart = True
                        break

                    # solutionDict[str(i).join(str(j))] = cell_availability
                    # if i == 9 and j == 9:
                    #     print("HERE")  


                if restart:
                    restart = False
                    break

        

        

                   
