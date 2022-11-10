from typing import List

BOARD_WIDTH = 7
BOARD_HEIGHT = 6
EMPTY = 0

def isValidMove(state: List[List[int]], i: int):
    
    for i in range(BOARD_WIDTH):
        if state[0][i] != EMPTY:
            return False
    return True