from typing import List

BOARD_WIDTH = 7
BOARD_HEIGHT = 6
EMPTY = 0

def isValidMove(state: List[List[int]], col: int):
    return state[0][col] == EMPTY

def countEmptySpaces(state:List[List[int]]):
    empties = 0
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            if state[i][j] == EMPTY:
                empties += 1
    return empties