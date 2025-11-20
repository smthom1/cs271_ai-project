import numpy as np
from constants import CLOSED, FLAG

def generate_constraints(my_board, board_size):
    constraints = []
    
    def is_valid_local(x, y):
        return (x >= 0) and (x < board_size) and (y >= 0) and (y < board_size)
    
    for r in range(board_size):                         # r is row num
        for c in range(board_size):                     # c is col num
            cell_value = my_board[r, c]                 # val of cell visible to the player
            
            # only revealed number cells (>=0) provide constraints
            if cell_value >= 0:
                hidden_neighbors = []
                flagged_count = 0
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        nr = r + dx
                        nc = c + dy
                        
                        if is_valid_local(nr, nc):
                            neighbor_value = my_board[nr, nc]
                            
                            if neighbor_value == CLOSED:
                                hidden_neighbors.append((nr, nc))
                            elif neighbor_value == FLAG:
                                flagged_count += 1
                
                bombs_remaining = cell_value - flagged_count
                
                # only create constraint if hidden neighbors involved
                if len(hidden_neighbors) > 0:
                    # if bombs_remaining < 0, player made a mistake (flagged too many), 
                    # (pass it anyway or clamp)
                    constraint = (hidden_neighbors, bombs_remaining) 
                    constraints.append(constraint)
    
    return constraints