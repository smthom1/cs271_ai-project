import numpy as np
from constraints import generate_constraints
from minesweeper import CLOSED, FLAG

#  3x3 board w/o flags
print("=== 3x3 w/o flags===")
test_board = np.array([
    [1, 1, CLOSED],
    [0, 1, CLOSED],
    [0, 0, CLOSED]
])
constraints = generate_constraints(test_board, 3)                       # get all constraints from the board
print(f"Constraints found: {len(constraints)}")                         # how many constraints found

for i in range(len(constraints)):                                       # i is constraint number 
    neighbors = constraints[i][0]                                       # list of hidden cells in this constraint
    bombs = constraints[i][1]                                           # number of bombs among those cells
    print(f"  Constraint {i+1}: {len(neighbors)} cells, {bombs} bomb(s)")
    print(f"    Cells: {neighbors}")

#  With flags
print("\n=== With Flags ===")
test_board2 = np.array([
    [2, FLAG, CLOSED],
    [1, 2, CLOSED],
    [0, 0, CLOSED]
])
constraints2 = generate_constraints(test_board2, 3)                     # get constraints from board with flags
print(f"Constraints found: {len(constraints2)}")


for i in range(len(constraints2)):                                      # loop through each constraint
    neighbors = constraints2[i][0]                                      # hidden cells
    bombs = constraints2[i][1]                                          # bombs remaining
    print(f"  Constraint {i+1}: {len(neighbors)} cells, {bombs} bomb(s)")
    print(f"    Cells: {neighbors}")

