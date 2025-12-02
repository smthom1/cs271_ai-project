# +1 Point: For every safe cell you successfully reveal.
# +1000 Points: Bonus for winning the game (clearing the board)
# -100 Points: Penalty for hitting a mine (losing)
# -1 Point: Penalty for wasting a turn on a cell that is already open

import pygame
import sys
import tkinter as tk
from tkinter import messagebox
from minesweeper import MinesweeperDiscreetEnv, MinesweeperInfiniteEnv
from constants import CLOSED # *switch to shared constants*

def get_user_config():
    config = {"size": 20, "mode": "Infinite"}
    
    root = tk.Tk()
    root.title("Setup")
    
    width = 300
    height = 250
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))

    tk.Label(root, text="Minesweeper Settings", font=("Arial", 14)).pack(pady=10)
    
    # MODE SELECTION
    mode_var = tk.StringVar(value="Standard") # default to std for testing
    frame_mode = tk.Frame(root)
    frame_mode.pack(pady=5)
    tk.Radiobutton(frame_mode, text="Standard (Fixed)", variable=mode_var, value="Standard").pack(anchor="w")
    tk.Radiobutton(frame_mode, text="Infinite (Expanding)", variable=mode_var, value="Infinite").pack(anchor="w")

    # Size input
    frame_size = tk.Frame(root)
    frame_size.pack(pady=5)
    tk.Label(frame_size, text="Grid/View Size:").pack(side=tk.LEFT)
    entry_size = tk.Entry(frame_size, width=5)
    entry_size.insert(0, "10")
    entry_size.pack(side=tk.LEFT)

    def on_start():
        try:
            s = int(entry_size.get())
            if s < 5 or s > 50:
                messagebox.showerror("Error", "Size must be 5-50")
                return
            config["size"] = s
            config["mode"] = mode_var.get()
            root.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid Size")

    tk.Button(root, text="PLAY", command=on_start, bg="#4CAF50", fg="white").pack(pady=20)
    root.mainloop()
    return config

# --- SETUP ---
config = get_user_config()
if not config: sys.exit()

camera_x = 0
camera_y = 0
is_infinite = False

if config["mode"] == "Standard":
    env = MinesweeperDiscreetEnv(board_size=config["size"], num_mines=int(config["size"]**2 * 0.15), render_mode="human")
    env.reset()
    is_infinite = False
else:
    env = MinesweeperInfiniteEnv(view_w=config["size"], view_h=config["size"], render_mode="human")
    is_infinite = True

print("--- CONTROLS ---")
print("Left Click: Reveal")
print("Right Click: Flag")
if is_infinite:
    print("WASD / ARROWS: Move Camera")

# --- LOOP ---
running = True

# safe to have one for the event loop --
clock = pygame.time.Clock()

while running:
    # rendering
    if is_infinite:
        env.render(camera_x, camera_y)
    else:
        env.render()

    # event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # cam movment (for infinite!)
        if is_infinite and event.type == pygame.KEYDOWN:
            step = 1
            if pygame.key.get_mods() & pygame.KMOD_SHIFT: step = 5 
            
            if event.key in [pygame.K_LEFT, pygame.K_a]: camera_x -= step
            if event.key in [pygame.K_RIGHT, pygame.K_d]: camera_x += step
            if event.key in [pygame.K_UP, pygame.K_w]: camera_y -= step
            if event.key in [pygame.K_DOWN, pygame.K_s]: camera_y += step

        # clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            
            if env.visualizer:
                header_h = env.visualizer.HEADER_HEIGHT
                cell_s = env.visualizer.cell_size
                
                if my > header_h:
                    screen_c = mx // cell_s
                    screen_r = (my - header_h) // cell_s
                    
                    # bounds check regarding viewport
                    if 0 <= screen_c < env.visualizer.view_width_cells and \
                       0 <= screen_r < env.visualizer.view_height_cells:
                        
                        if is_infinite:
                            world_c = camera_x + screen_c
                            world_r = camera_y + screen_r
                            if event.button == 1: env.step(world_r, world_c)
                            elif event.button == 3: env.toggle_flag(world_r, world_c)
                        else:
                            # std mode
                            if 0 <= screen_r < env.board_size and 0 <= screen_c < env.board_size:
                                action = screen_r * env.board_size + screen_c
                                if event.button == 1: 
                                    # capture reward to update score
                                    obs, reward, done, truncated, info = env.step(action)
                                    env.total_reward += reward # <--- SCORE FIX
                                elif event.button == 3: 
                                    env.toggle_flag(screen_r, screen_c)

    clock.tick(30) # keep loop from spinning too fast

env.close()
pygame.quit()
sys.exit()
