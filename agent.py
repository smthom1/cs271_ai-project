# AUTOMATED VERSION WITH SIMPLE LOGIC AGENT

import gymnasium as gym
import numpy as np
import time
import random

from minesweeper import MinesweeperDiscreetEnv, CLOSED, FLAG, is_valid

def get_neighbors(board_state, r, c):

    neighbors = []
    board_size = board_state.shape[0]
    
    for _r in range(r - 1, r + 2):
        for _c in range(c - 1, c + 2):
            # skip cell itself
            if (_r == r and _c == c):
                continue
            # check if neighbor is within board bounds
            if is_valid(_r, _c): # using is_valid from minesweeper.py
                neighbors.append( ((_r, _c), board_state[_r, _c]) )
                
    return neighbors

def find_safe_moves(board_state):
    safe_moves = set()
    board_size = board_state.shape[0]

    # every cell in board
    for r in range(board_size):
        for c in range(board_size):
            cell_value = board_state[r, c]
            
            if cell_value > 0 and cell_value <= 8:
                
                num_flagged_neighbors = 0
                closed_neighbors = [] # store coords of closed neighbors
                
                # check all neighbors
                for (nr, nc), n_val in get_neighbors(board_state, r, c):
                    if n_val == FLAG:
                        num_flagged_neighbors += 1
                    elif n_val == CLOSED:
                        closed_neighbors.append((nr, nc))
                
                # **Core safe logic:
                # if number of flags = cell's number, all other closed cells are safe
                if num_flagged_neighbors == cell_value:
                    for (nr, nc) in closed_neighbors:
                        safe_moves.add((nr, nc))
                        
    return safe_moves

def find_flag_moves(board_state):
    flag_moves = set()
    board_size = board_state.shape[0]
    
    # look over every cell on board
    for r in range(board_size):
        for c in range(board_size):
            cell_value = board_state[r, c]
            
            # revealed number cells (1-8)
            if cell_value > 0 and cell_value <= 8:
                
                num_flagged_neighbors = 0
                closed_neighbors = [] # store coords of closed neighbors
                
                # check all neighbors
                for (nr, nc), n_val in get_neighbors(board_state, r, c):
                    if n_val == FLAG:
                        num_flagged_neighbors += 1
                    elif n_val == CLOSED:
                        closed_neighbors.append((nr, nc))
                
                # **Core flag logic:
                # if (num of closed + num of flags) = cell's number,
                # then all those closed cells MUST be mines
                if (num_flagged_neighbors + len(closed_neighbors)) == cell_value:
                    for (nr, nc) in closed_neighbors:
                        flag_moves.add((nr, nc))
                        
    return flag_moves


def run_agent_game():
    
    # create environment with "human" render_mode to watch the agent play
    env = MinesweeperDiscreetEnv(render_mode="human")
    
    # get initial (empty) observation
    observation, info = env.reset()
    board_state = observation
    done = False
    
    print("--- AGENT GAME START ---")
    
    # Make first move
    
    # choose rand (row, col)
    first_r = random.randint(0, env.board_size - 1)
    first_c = random.randint(0, env.board_size - 1)
    
    # convert (r, c) to discrete action integer
    first_action = first_r * env.board_size + first_c
    
    print(f"Making random first move at: ({first_r}, {first_c})")
    
    # take first step
    observation, reward, done, truncated, info = env.step(first_action)
    
    # Main agent loop
    while not done:    
        made_a_move_in_loop = True # flag tracks if we're getting anywhere
        
        while made_a_move_in_loop and not done:
            
            made_a_move_in_loop = False # Assume STUCK until move found
            board_state = observation   # get current board state
            
            # Apply "safe" strategy
            safe_moves_to_make = find_safe_moves(board_state)
            
            if safe_moves_to_make:
                print(f"Found {len(safe_moves_to_make)} safe moves: {safe_moves_to_make}")
                made_a_move_in_loop = True # found move!
                
                # reveal ALL safe cells found in this pass
                for (r, c) in safe_moves_to_make:
                    # check if cell is still closed (should be, but just to make sure)
                    if observation[r, c] == CLOSED and not done:
                        action = r * env.board_size + c
                        observation, reward, done, truncated, info = env.step(action)
                
                continue 

            # Apply "Flag" Strategy, only run if NO safe moves found
            flag_moves_to_make = find_flag_moves(board_state)
            
            if flag_moves_to_make:
                print(f"Found {len(flag_moves_to_make)} mines to flag: {flag_moves_to_make}")
                made_a_move_in_loop = True # --- found a move
                
                # flag ALL cells found in this pass
                for (r, c) in flag_moves_to_make:
                    # check if cell is still closed (not flagged by logic)
                    if observation[r, c] == CLOSED:
                        env.toggle_flag(r, c)
                
                # manually update 'observation' variable after flagging
                # bc toggle_flag() doesn't return it.
                observation = env.my_board
                
                # rereun strategies
                continue
        
        # What to do if STUCK
        #   If: game is NOT 'done' (win/lose).
        #   AND agent is "stuck" (no more simple moves found).
        
        if not done:
            # stuck!
            print("\n--- AGENT STUCK ---")
            print("No more simple 'safe' or 'flag' moves found. Making a random guess...")
            
            # find all available (CLOSED) cells
            closed_r, closed_c = np.where(board_state == CLOSED)
            
            # are any cells left to guess?
            if len(closed_r) == 0:
                print("No closed cells left to guess. Agent is truly stuck.")
                break # No choice but to stop
            
            # pick rand cell from list of closed cells
            random_index = random.randint(0, len(closed_r) - 1)
            guess_r = closed_r[random_index]
            guess_c = closed_c[random_index]
            
            print(f"Guessing at: ({guess_r}, {guess_c})")
            
            # conv(r, c) to  discrete action
            guess_action = guess_r * env.board_size + guess_c
            
            observation, reward, done, truncated, info = env.step(guess_action)
            
            # No more break!
    
    # --- Game Over ---
    if done:
        print("\n--- GAME OVER ---")
        if env.game_over_status == "win":
            print("Result: Agent won")
        else:
            print("Result: Agent lost")
        print(f"Final Score: {env.total_reward}")
    
    # keeps window open for a few seconds so you can see the board in full
    print("closing in 5 seconds...")
    time.sleep(5)
    env.close()


# run the agent!
if __name__ == "__main__":
    run_agent_game()