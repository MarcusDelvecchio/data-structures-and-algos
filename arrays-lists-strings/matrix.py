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