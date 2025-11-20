import gymnasium as gym
import numpy as np
import time
import random

from minesweeper import MinesweeperDiscreetEnv, is_valid
from constants import CLOSED, FLAG

def get_neighbors(board_state, r, c):
    neighbors = []
    board_size = board_state.shape[0]
    
    for _r in range(r - 1, r + 2):
        for _c in range(c - 1, c + 2):
            if (_r == r and _c == c): continue
            if is_valid(_r, _c, board_size):
                neighbors.append( ((_r, _c), board_state[_r, _c]) )
    return neighbors

def find_safe_moves(board_state):
    safe_moves = set()
    board_size = board_state.shape[0]

    for r in range(board_size):
        for c in range(board_size):
            cell_value = board_state[r, c]
            
            if cell_value > 0 and cell_value <= 8:
                num_flagged_neighbors = 0
                closed_neighbors = [] 
                
                for (nr, nc), n_val in get_neighbors(board_state, r, c):
                    if n_val == FLAG: num_flagged_neighbors += 1
                    elif n_val == CLOSED: closed_neighbors.append((nr, nc))
                
                if num_flagged_neighbors == cell_value:
                    for (nr, nc) in closed_neighbors:
                        safe_moves.add((nr, nc))
    return safe_moves

def find_flag_moves(board_state):
    flag_moves = set()
    board_size = board_state.shape[0]
    
    for r in range(board_size):
        for c in range(board_size):
            cell_value = board_state[r, c]
            if cell_value > 0 and cell_value <= 8:
                num_flagged_neighbors = 0
                closed_neighbors = []
                
                for (nr, nc), n_val in get_neighbors(board_state, r, c):
                    if n_val == FLAG: num_flagged_neighbors += 1
                    elif n_val == CLOSED: closed_neighbors.append((nr, nc))
                
                if (num_flagged_neighbors + len(closed_neighbors)) == cell_value:
                    for (nr, nc) in closed_neighbors:
                        flag_moves.add((nr, nc))
    return flag_moves

def run_agent_game():
    env = MinesweeperDiscreetEnv(render_mode="human")
    observation, info = env.reset()
    board_state = observation
    done = False
    
    print("--- AGENT GAME START ---")
    first_r = random.randint(0, env.board_size - 1)
    first_c = random.randint(0, env.board_size - 1)
    first_action = first_r * env.board_size + first_c
    print(f"Making random first move at: ({first_r}, {first_c})")
    
    observation, reward, done, truncated, info = env.step(first_action)
    
    while not done:    
        made_a_move_in_loop = True
        while made_a_move_in_loop and not done:
            made_a_move_in_loop = False 
            board_state = observation   
            
            safe_moves_to_make = find_safe_moves(board_state)
            if safe_moves_to_make:
                print(f"Found {len(safe_moves_to_make)} safe moves")
                made_a_move_in_loop = True
                for (r, c) in safe_moves_to_make:
                    if observation[r, c] == CLOSED and not done:
                        action = r * env.board_size + c
                        observation, reward, done, truncated, info = env.step(action)
                continue 

            flag_moves_to_make = find_flag_moves(board_state)
            if flag_moves_to_make:
                print(f"Found {len(flag_moves_to_make)} mines to flag")
                made_a_move_in_loop = True
                for (r, c) in flag_moves_to_make:
                    if observation[r, c] == CLOSED:
                        env.toggle_flag(r, c)
                observation = env.my_board
                continue
        
        if not done:
            print("\n--- AGENT STUCK ---")
            closed_r, closed_c = np.where(board_state == CLOSED)
            if len(closed_r) == 0: break 
            
            random_index = random.randint(0, len(closed_r) - 1)
            guess_r = closed_r[random_index]
            guess_c = closed_c[random_index]
            guess_action = guess_r * env.board_size + guess_c
            
            print(f"Guessing at: ({guess_r}, {guess_c})")
            observation, reward, done, truncated, info = env.step(guess_action)
    
    if done:
        print("\n--- GAME OVER ---")
        print(f"Result: {env.game_over_status}")
        print(f"Final Score: {env.total_reward}")
    
    time.sleep(5)
    env.close()

if __name__ == "__main__":
    run_agent_game()