import sys
from six import StringIO
from random import randint
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except ImportError:
    print("Pygame not found. Please install it with: pip install pygame")
    sys.exit(1)

# --- Constants ---
BOARD_SIZE = 10
NUM_MINES = 9
CLOSED = -2
MINE = -1
FLAG = -3

# --- Helper Functions (Unchanged) ---

def board2str(board, end='\n'):
    s = ''
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            s += str(board[x][y]) + '\t'
        s += end
    return s[:-len(end)]

def is_new_move(my_board, x, y):
    return my_board[x, y] == CLOSED

def is_valid(x, y):
    return (x >= 0) & (x < BOARD_SIZE) & (y >= 0) & (y < BOARD_SIZE)

def is_win(my_board):
    """ return if the game is won """
    # num of cells hidden (closed or flagged)
    unopened_cells = np.count_nonzero(my_board == CLOSED) + np.count_nonzero(my_board == FLAG)

    # game = won given num of unoopened_cells == num of mines
    return unopened_cells == NUM_MINES

def is_mine(board, x, y):
    return board[x, y] == MINE

def place_mines(board_size, num_mines):
    # ** standard mine placing **
    mines_placed = 0
    board = np.zeros((board_size, board_size), dtype=int)
    while mines_placed < num_mines:
        rnd = randint(0, board_size * board_size - 1)
        x = int(rnd / board_size)
        y = int(rnd % board_size)
        if is_valid(x, y):
            if not is_mine(board, x, y):
                board[x, y] = MINE
                mines_placed += 1
    return board

# Visualizer Class!! ========================

# ++ bad luck protection
def place_mines_safely(board_size, num_mines, first_x, first_y):
    # places mines on board but confirms that first cell (first_x, first_y) is NOT a mine
    # is *not* a mine); no losing on the first move
    mines_placed = 0
    board = np.zeros((board_size, board_size), dtype=int)
    
    while mines_placed < num_mines:
        rnd = randint(0, board_size * board_size - 1)
        x = int(rnd / board_size)
        y = int(rnd % board_size)
        
        # Check if the random (x, y) is valid AND not the first-move cell
        if is_valid(x, y) and not is_mine(board, x, y) and (x != first_x or y != first_y):
            board[x, y] = MINE
            mines_placed += 1
            
    return board

class MinesweeperVisualizer:
    """
    Handles all Pygame rendering for the Minesweeper environment.
    """
    def __init__(self, board_size, cell_size=40):
        self.board_size = board_size
        self.cell_size = cell_size
        self.HEADER_HEIGHT = 60                 # shove some space in here to allow info vis
        self.window_size_x = self.board_size * self.cell_size
        self.window_size_y = self.board_size * self.cell_size + self.HEADER_HEIGHT
        
        self.window = None
        self.clock = None
        self.cell_font = None
        self.header_font = None
        self.game_over_font = None              # game over font

        # colorss
        self.colors = {
            "header_bg": (50, 50, 50),
            "header_text": (255, 255, 255),
            "closed": (180, 180, 180),
            "flag": (255, 180, 0),              # flags are orange hex
            "open": (210, 210, 210),
            "grid": (120, 120, 120),
            "mine": (255, 0, 0),
            "text": (0, 0, 0),
            1: (0, 0, 255),     # 1: Blue
            2: (0, 128, 0),     # 2: Green
            3: (255, 0, 0),     # 3: Red
            4: (0, 0, 128),     # 4: Dark Blue
            5: (128, 0, 0),     # 5: Maroon
            6: (0, 128, 128),   # 6: Teal
            7: (0, 0, 0),       # 7: Black
            8: (128, 128, 128)  # 8: Gray
        }

    def _init_pygame(self):
        """Initialize Pygame, create window, font, and clock."""
        pygame.init()
        pygame.display.set_caption("Minesweeper")
        self.window = pygame.display.set_mode((self.window_size_x, self.window_size_y))
        self.clock = pygame.time.Clock()
        try:
            self.cell_font = pygame.font.Font(None, int(self.cell_size * 0.8))
            self.header_font = pygame.font.Font(None, int(self.HEADER_HEIGHT * 0.6))
            self.game_over_font = pygame.font.Font(None, int(self.cell_size * 1.5))
        except:
            self.cell_font = pygame.font.SysFont('Arial', int(self.cell_size * 0.8))
            self.header_font = pygame.font.SysFont('Arial', int(self.HEADER_HEIGHT * 0.6))
            self.game_over_font = pygame.font.SysFont('Arial', int(self.cell_size * 1.5))

    def render_frame(self, board, score, bombs_left, game_over_status=None):
        """Render the current board state to the Pygame window."""
        if self.window is None:
            self._init_pygame()
        
        # Header ===
        self.window.fill(self.colors["header_bg"])
        header_rect = pygame.Rect(0, 0, self.window_size_x, self.HEADER_HEIGHT)
        pygame.draw.rect(self.window, self.colors["header_bg"], header_rect)
        
        score_text = self.header_font.render(f"Score: {score}", True, self.colors["header_text"])
        score_rect = score_text.get_rect(midleft=(20, self.HEADER_HEIGHT // 2))
        self.window.blit(score_text, score_rect)
        
        bomb_text = self.header_font.render(f"Bombs: {bombs_left}", True, self.colors["header_text"])
        bomb_rect = bomb_text.get_rect(midright=(self.window_size_x - 20, self.HEADER_HEIGHT // 2))
        self.window.blit(bomb_text, bomb_rect)

        # Draw board cells ===
        for r in range(self.board_size):
            for c in range(self.board_size):
                val = board[r, c]
                rect = pygame.Rect(c * self.cell_size, 
                                   r * self.cell_size + self.HEADER_HEIGHT, 
                                   self.cell_size, self.cell_size)
                
                if val == CLOSED:
                    pygame.draw.rect(self.window, self.colors["closed"], rect)
                    pygame.draw.line(self.window, (255,255,255), rect.topleft, rect.topright, 2)
                    pygame.draw.line(self.window, (255,255,255), rect.topleft, rect.bottomleft, 2)
                    pygame.draw.line(self.window, (100,100,100), rect.bottomright, rect.topright, 2)
                    pygame.draw.line(self.window, (100,100,100), rect.bottomright, rect.bottomleft, 2)
                
                elif val == FLAG:
                    pygame.draw.rect(self.window, self.colors["closed"], rect)
                    pygame.draw.rect(self.window, self.colors["flag"], rect.inflate(-10, -10))
                    text = self.cell_font.render("F", True, self.colors["text"])
                    text_rect = text.get_rect(center=rect.center)
                    self.window.blit(text, text_rect)
                
                elif val == MINE:
                    pygame.draw.rect(self.window, self.colors["mine"], rect)
                    pygame.draw.circle(self.window, self.colors["text"], rect.center, int(self.cell_size * 0.3))

                elif val >= 0:
                    pygame.draw.rect(self.window, self.colors["open"], rect)
                    if val > 0:
                        text_color = self.colors.get(val, self.colors["text"])
                        text = self.cell_font.render(str(val), True, text_color)
                        text_rect = text.get_rect(center=rect.center)
                        self.window.blit(text, text_rect)
                
                pygame.draw.rect(self.window, self.colors["grid"], rect, 1)

        # Game overlay (game over) ===
        if game_over_status:
            # semi-transparent overlay from pygame documentation: https://www.pygame.org/docs/ref/surface.html
            overlay_rect = pygame.Rect(0, self.HEADER_HEIGHT, self.window_size_x, self.window_size_y - self.HEADER_HEIGHT)
            overlay_surface = pygame.Surface((overlay_rect.width, overlay_rect.height), pygame.SRCALPHA)
            
            if game_over_status == "win":
                overlay_surface.fill((0, 255, 0, 128)) # green tint
                text = "YOU WIN!"
                text_color = (255, 255, 255)
            else: # "loss"
                overlay_surface.fill((255, 0, 0, 128)) # red tint
                text = "GAME OVER"
                text_color = (255, 255, 255)
            
            # draw tint
            self.window.blit(overlay_surface, overlay_rect.topleft)
            
            # draw text
            game_over_text = self.game_over_font.render(text, True, text_color)
            text_rect = game_over_text.get_rect(center=overlay_rect.center)
            self.window.blit(game_over_text, text_rect)
            
        # display updates
        pygame.display.flip()
        self.clock.tick(10) # max 10 FPS
        return

    def close(self):
        """Close the Pygame window."""
        if self.window:
            pygame.quit()
            self.window = None
            self.clock = None
            self.cell_font = None
            self.header_font = None
            self.game_over_font = None


# OG Minesweeper Env ===
class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]} 
    def __init__(self, board_size=BOARD_SIZE, num_mines=NUM_MINES):
        self.board_size = board_size
        self.num_mines = num_mines
        self.board = place_mines(board_size, num_mines)
        self.my_board = np.ones((board_size, board_size), dtype=int) * CLOSED
        self.valid_actions = np.ones((self.board_size, self.board_size), dtype=bool)

        self.observation_space = spaces.Box(low=-3, high=8, 
                                            shape=(self.board_size, self.board_size), dtype=int)
        self.action_space = spaces.MultiDiscrete([self.board_size, self.board_size])

    def count_neighbour_mines(self, x, y):
        neighbour_mines = 0
        for _x in range(x - 1, x + 2):
            for _y in range(y - 1, y + 2):
                if is_valid(_x, _y):
                    if is_mine(self.board, _x, _y):
                        neighbour_mines += 1
        return neighbour_mines

    def open_neighbour_cells(self, my_board, x, y):
        for _x in range(x-1, x+2):
            for _y in range(y-1, y+2):
                if is_valid(_x, _y):
                    if is_new_move(my_board, _x, _y):
                        my_board[_x, _y] = self.count_neighbour_mines(_x, _y)
                        if my_board[_x, _y] == 0:
                            my_board = self.open_neighbour_cells(my_board, _x, _y)
        return my_board

    def get_next_state(self, state, x, y):
        my_board = state
        game_over = False
        
        if self.board is None:
            raise Exception("gen_next_state called before self.board is generated")
        
        if is_mine(self.board, x, y):
            my_board[x, y] = MINE
            game_over = True
        else:
            my_board[x,y] = self.count_neighbor_mines(x,y)
            #[x, y] = self.count_neighbour_mines(x, y)
            if my_board[x, y] == 0:
                my_board = self.open_neighbour_cells(my_board, x, y)
        self.my_board = my_board
        return my_board, game_over

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles seeding
        self.board = place_mines(self.board_size, self.num_mines)
        self.my_board = np.ones((self.board_size, self.board_size), dtype=int) * CLOSED
        self.valid_actions = np.ones((self.board_size, self.board_size), dtype=bool)
        
        info = {'valid_actions': self.valid_actions}
        return self.my_board, info

    def step(self, action):
        state = self.my_board
        x = int(round(action[0]))
        y = int(round(action[1]))

        if bool(self.valid_actions[x, y]) is False:
            raise Exception("Invalid action was selected!")

        next_state, reward, done, info = self.next_step(state, x, y)
        self.my_board = next_state
        self.valid_actions = (next_state == CLOSED)
        info['valid_actions'] = (next_state == CLOSED)
        
        truncated = False # This env doesn't have a time limit
        return next_state, reward, done, truncated, info

    def next_step(self, state, x, y):
        my_board = state
        info = {}
        if not is_new_move(my_board, x, y):
            return my_board, -1, False, info
        
        state, game_over = self.get_next_state(my_board, x, y)
        if not game_over:
            if is_win(state):
                return state, 1000, True, info
            else:
                return state, 5, False, info
        else:
            return state, -100, True, info

    def render(self, mode='ansi'):
        if mode == 'ansi':
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            s = board2str(self.board)
            outfile.write(s)
            return outfile
        else:
            super().render(mode=mode) # Not supported
    
    def close(self):
        pass

class MinesweeperDiscreetEnv(gym.Env):
    """
    This version uses a discrete action space (0-99) and supports
    Pygame rendering, flagging, score, and bomb count display.
        → WITH the bad luck protection
    """
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 10}

    def __init__(self, board_size=BOARD_SIZE, num_mines=NUM_MINES, render_mode=None):
        
        self.board_size = board_size
        self.num_mines = num_mines
        # self.board = place_mines(board_size, num_mines)
        self.board = None # will be set on first move
        self.my_board = np.ones((self.board_size, self.board_size), dtype=int) * CLOSED
        self.num_actions = 0
        
        self.total_reward = 0
        self.flags_placed = 0
        self.game_over_status = None

        # ++ NEW part: flag to track if first move has been made
        self.first_move_made = False
        
        self.observation_space = spaces.Box(low=-3, high=8, 
                                            shape=(self.board_size, self.board_size), dtype=int)
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.valid_actions = np.ones((self.board_size * self.board_size), dtype=bool)

        self.render_mode = render_mode
        self.visualizer = None
        if self.render_mode == "human":
            self.visualizer = MinesweeperVisualizer(self.board_size)

    def count_neighbour_mines(self, x, y):
        neighbour_mines = 0
        for _x in range(x - 1, x + 2):
            for _y in range(y - 1, y + 2):
                if is_valid(_x, _y):
                    if is_mine(self.board, _x, _y):
                        neighbour_mines += 1
        return neighbour_mines

    def open_neighbour_cells(self, my_board, x, y):
        # safety check
        if self.board is None:
            # should NOT happen if logic is correct, just a safeguard
            print("Error: Tried to open cells before board was generated.")
            return my_board
        
        for _x in range(x-1, x+2):
            for _y in range(y-1, y+2):
                if is_valid(_x, _y):
                    if is_new_move(my_board, _x, _y):
                        my_board[_x, _y] = self.count_neighbour_mines(_x, _y)
                        if my_board[_x, _y] == 0:
                            my_board = self.open_neighbour_cells(my_board, _x, _y)
        return my_board

    def get_next_state(self, state, x, y):
        my_board = state
        game_over = False
        
        # safety check, make sure board is generated BEFORE checking for mines
        if self.board is None:
            #would be a logic error if occurred
            raise Exception("gen_next_state called before self.board is generated")
        
        if is_mine(self.board, x, y):
            my_board[x, y] = MINE
            game_over = True
        else:
            my_board[x, y] = self.count_neighbour_mines(x, y)
            if my_board[x, y] == 0:
                my_board = self.open_neighbour_cells(my_board, x, y)
        self.my_board = my_board
        return my_board, game_over

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        #reset 'my_board' (player view)
        self.my_board = np.ones((self.board_size, self.board_size), dtype=int) * CLOSED
        
        #self.board = place_mines(self.board_size, self.num_mines)
        self.board = None # will be set on first move
        self.num_actions = 0
        self.valid_actions = np.ones((self.board_size * self.board_size), dtype=bool)
        
        self.total_reward = 0
        self.flags_placed = 0
        self.game_over_status = None # RESET status
        
        self.first_move_made = False
        
        info = {'valid_actions': self.valid_actions}
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self.my_board, info

    def step(self, action):
        state = self.my_board
        x = int(action / self.board_size)
        y = int(action % self.board_size)
        
        # BAD LUCK PROTECTION (first move)
        if not self.first_move_made:
            self.first_move_made = True
            self.board = place_mines_safely(self.board_size, self.num_mines, x, y)
            # *debug print*
            print(f"First move at ({x}, {y}). Board generated.")

        next_state, reward, done, info = self.next_step(state, x, y)
        
        self.total_reward += reward
        
        self.my_board = next_state
        self.num_actions += 1
        self.valid_actions = (next_state.flatten() == CLOSED)
        info['valid_actions'] = self.valid_actions
        info['num_actions'] = self.num_actions

        truncated = False 

        # --- NEW: Update game over status ---
        if done:
            if reward > 0:
                self.game_over_status = "win"
            else:
                self.game_over_status = "loss"

        if self.render_mode == "human":
            self._render_frame()

        return next_state, reward, done, truncated, info

    def next_step(self, state, x, y):
        my_board = state
        info = {}
        
        if my_board[x, y] != CLOSED:
            return my_board, -1, False, info
        
        state, game_over = self.get_next_state(my_board, x, y)
        if not game_over:
            if is_win(state):
                return state, 1000, True, info
            else:
                return state, 5, False, info
        else:
            return state, -100, True, info

    def toggle_flag(self, x, y):
        """place/remove a flag on a cell"""
        if not is_valid(x, y):
            return 
        
        if self.my_board[x, y] == CLOSED:
            self.my_board[x, y] = FLAG
            self.flags_placed += 1
        elif self.my_board[x, y] == FLAG:
            self.my_board[x, y] = CLOSED
            self.flags_placed -= 1
        
        if self.render_mode == "human":
            self._render_frame()

    def render(self):
        if self.render_mode == 'ansi':
            return self._render_ansi()
        elif self.render_mode == 'human':
            return self._render_frame()

    def _render_ansi(self):
        outfile = StringIO()
        s = board2str(self.my_board)
        outfile.write(s)
        sys.stdout.write(outfile.getvalue())
        return outfile.getvalue()

    def _render_frame(self):
        if self.visualizer:
            bombs_left = self.num_mines - self.flags_placed
            self.visualizer.render_frame(self.my_board, self.total_reward, bombs_left, self.game_over_status)

    def close(self):
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = None