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

    def solveSudoku(self, board: List[List[str]]) -> None:
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
                        if continues == 81:
                            print("RETURNING")
                            return
                    else:
                        board, solutionDict = Solution.updateCellAvailabilities(i, j, board, solutionDict)

    def updateCellAvailabilities(i, j, board, solutionDict):
        cell_box_availabilities = Solution.get_box_availbilities(board, i, j)
        cell_row_availabilities = Solution.get_row_availabilities(board, i)
        cell_col_availabilities = Solution.get_col_availabilities(board, j)
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
                    board, solutionDict = Solution.updateCellAvailabilities(i, j, board, solutionDict)
                elif int(ij[1]) == j and len(solutionDict[ij]) == 2:
                    i = int(ij[0])
                    j = int(ij[1])
                    board, solutionDict = Solution.updateCellAvailabilities(i, j, board, solutionDict)
                # could recalculate box as well
        return board, solutionDict