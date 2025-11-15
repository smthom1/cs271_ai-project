## HUMAN VERSION, NO AUTOMATION

import gymnasium as gym
from minesweeper import MinesweeperDiscreetEnv, CLOSED
import numpy as np
import pygame  # using PYGAME (human plays w mouse clicks)
import sys

# create env
env = MinesweeperDiscreetEnv(render_mode="human")
print("--- GAME START ---")
print("Human player: Left-click to reveal, Right-click to flag.")
print("Close the window to quit.")

# reset env to get first board
observation, info = env.reset()     # env.reset() calls render() automatically

# Main game loop ===
running = True
done = False

# get constants from vis for coordinate conversion
if env.visualizer:
    CELL_SIZE = env.visualizer.cell_size
    HEADER_HEIGHT = env.visualizer.HEADER_HEIGHT
else:
    # fallback if not in human mode
    CELL_SIZE = 40                  # doesn't really work well, but shouldn't be used regardless (just a fallback)
    HEADER_HEIGHT = 60


while running:
    
    # wait for user event (mouse click, quit)
    event = pygame.event.wait() 
    
    # window close ===
    if event.type == pygame.QUIT:
        running = False
            
    # mouse click ===
    # >> only process clicks if the game is NOT done
    if not done and event.type == pygame.MOUSEBUTTONDOWN:
        
        # convert pixel coords to *grid action*
        pixel_x, pixel_y = event.pos
        
        # is click in header area?
        if pixel_y < HEADER_HEIGHT:
            print("Clicked on header, no action taken.")
            continue 

        # convert pixel coords to grid row x column
        grid_row = (pixel_y - HEADER_HEIGHT) // CELL_SIZE
        grid_col = pixel_x // CELL_SIZE

        # make sure click is within board bounds
        if not (0 <= grid_row < env.board_size and 0 <= grid_col < env.board_size):
            continue
        
        # LEFT CLICK (reveal) ===
        if event.button == 1: 
            # conv (row, col) to single action int
            action = grid_row * env.board_size + grid_col
            
            # check if action valid (square is not already opened)
            current_cell_state = env.my_board[grid_row, grid_col]
            if current_cell_state != CLOSED:
                print(f"Invalid move: ({grid_row}, {grid_col}) is already open or flagged. Try again.")
                continue
            
            print(f"Left-Clicked: ({grid_row}, {grid_col})")

            # take action in env
            observation, reward, done, truncated, info = env.step(action)
            
            if done:
                print("\n--- GAME OVER ---")
                if reward > 0: # win!
                    print(f"You Win! Final Score: {env.total_reward}")
                else: # lose!
                    print(f"You hit a Mine! Final Score: {env.total_reward}")
                print("Close the window to exit.")
            
            if truncated:
                running = False
        
        # RIGHT click (flag) ===
        elif event.button == 3:
            print(f"Right-Clicked: ({grid_row}, {grid_col})")
            env.toggle_flag(grid_row, grid_col)


# clean up, close everything
print("Game finished. Closing...")
env.close()
pygame.quit()
sys.exit()