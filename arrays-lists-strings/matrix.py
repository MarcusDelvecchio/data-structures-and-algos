
# Search a 2D Matrix LeetCode Medium
# https://leetcode.com/problems/search-a-2d-matrix/
# TC: O(logn), SC: O(1) (zero auxillary)
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    T, B = 0, len(matrix) - 1

    # find the target row using BS within the rows
    target_row = 0
    while T <= B:
        mid = (T + B) // 2
        if target > matrix[mid][-1]:
            T = mid + 1
        elif target < matrix[mid][0]:
            B = mid - 1
        else:
            target_row = mid
            break
    
    # find the target in the target row
    L, R = 0, len(matrix[0]) - 1
    while L <= R:
        mid = (L + R) // 2
        if matrix[target_row][mid] > target:
            R = mid - 1
        elif matrix[target_row][mid] < target:
            L = mid + 1
        else:
            return True
    return False


# Game of Life LeetCode Medium
# https://leetcode.com/problems/game-of-life/description/
# TC: O(n), SC: O(1)
def gameOfLife(self, board: List[List[int]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    
    to_die = set()
    to_birth = set()
    rows = len(board); cols = len(board[0])

    for r in range(rows):
        for c in range(cols):
            neighbors = 0

            # count the neighbors
            for next_r in [-1, 0, 1]:
                for next_c in [-1, 0, 1]:
                    if next_r == 0 and next_c == 0: continue
                    new_r = r + next_r; new_c = c + next_c
                    if new_r >= 0 and new_r < rows and new_c >= 0 and new_c < cols and board[new_r][new_c]:
                        neighbors += 1
            
            # apply rules based on neighbor count
            print(r, c, neighbors)
            if board[r][c] and (neighbors < 2 or neighbors > 3):
                to_die.add((r,c))
            if not board[r][c] and neighbors == 3:
                to_birth.add((r,c))                
    
    def new_val(r, c):
        if board[r][c]:
            return (r, c) not in to_die
        else:
            return (r, c) in to_birth

    # update the values based on to_die and to_birth
    for r in range(rows):
        for c in range(cols):
            board[r][c] = int(new_val(r, c))