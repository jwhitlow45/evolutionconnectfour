from typing import List

BOARD_WIDTH = 7
BOARD_HEIGHT = 6
EMPTY = 0

def isValidMove(state: List[List[int]], col: int):
    
    for col in range(BOARD_WIDTH):
        if state[0][col] != EMPTY:
            return False
    return True