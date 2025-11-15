import numpy as np
# from minesweeper import CLOSED, FLAG

def generate_constraints(my_board, board_size):
    
    #ADDED
    from minesweeper import CLOSED, FLAG

    constraints = []
    
    def is_valid_local(x, y):
        return (x >= 0) and (x < board_size) and (y >= 0) and (y < board_size)
    
    for r in range(board_size):                         #r is row num
        for c in range(board_size):                     #c is col num
            cell_value = my_board[r, c]                 # val of cell visible to the player atp
            
            if cell_value >= 0:                         # >=0 i.e cells that are revealed, only revealed cells can give constraint
                hidden_neighbors = []
                flagged_count = 0
                
                for dx in [-1, 0, 1]:                   # dx is change in x: -1, left etc..
                    for dy in [-1, 0, 1]:               #dy is chnage in y: -1, down etc...
                        if dx == 0 and dy == 0:         # if no movement in x and y then continue 
                            continue
                        
                        nr = r + dx                     # neighbor row
                        nc = c + dy                     # neighbor col
                        
                        if is_valid_local(nr, nc):
                            neighbor_value = my_board[nr, nc]           #what board currently sees at neighbor
                            
                            if neighbor_value == CLOSED:                #neighbor val hidden
                                hidden_neighbors.append((nr, nc))       #add these vars to constraint var list
                            elif neighbor_value == FLAG:                #neighbor val already flaged
                                flagged_count += 1                      #increment count of flaged
                
                bombs_remaining = cell_value - flagged_count            #how many bombs remaining amongst hidden bombs 
                
                if len(hidden_neighbors) > 0 and bombs_remaining >= 0:  #only create constraint when unkown neighbors to constrain
                    constraint = (hidden_neighbors, bombs_remaining)    #(list of coords, int)
                    constraints.append(constraint)
    
    return constraints