import sys
from random import randint, seed, random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- IMPORTS ---
from constants import DEFAULT_BOARD_SIZE, CLOSED, MINE, FLAG
from constraints import generate_constraints

try:
    import pygame
except ImportError:
    print("Pygame not found. Please install it with: pip install pygame")
    # We don't exit here to allow headless agents to run without pygame if needed
    pass

# --- Helper Functions ---

def board2str(board, end='\n'):
    s = ''
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            s += str(board[x][y]) + '\t'
        s += end
    return s[:-len(end)]

def is_new_move(my_board, x, y):
    return my_board[x, y] == CLOSED

def is_valid(x, y, board_size):
    return (x >= 0) & (x < board_size) & (y >= 0) & (y < board_size)

def is_win(my_board, num_mines):
    unopened_cells = np.count_nonzero(my_board == CLOSED) + np.count_nonzero(my_board == FLAG)
    return unopened_cells == num_mines

def is_mine(board, x, y):
    return board[x, y] == MINE

def place_mines_safely(board_size, num_mines, first_x, first_y):
    forbidden = set()
    for x in range(first_x - 1, first_x + 2):
        for y in range(first_y - 1, first_y + 2):
            if 0 <= x < board_size and 0 <= y < board_size:
                forbidden.add((x, y))  

    board = np.zeros((board_size, board_size), dtype=int)
    mines_placed = 0

    while mines_placed < num_mines:
        rnd = randint(0, board_size * board_size - 1)
        x = rnd // board_size
        y = rnd % board_size

        if (x, y) in forbidden:
            continue

        if is_valid(x, y, board_size) and not is_mine(board, x, y):
            board[x, y] = MINE
            mines_placed += 1

    return board

# --- Visualizer ---

class MinesweeperVisualizer:
    def __init__(self, view_width, view_height, cell_size=30):
        self.view_width_cells = view_width
        self.view_height_cells = view_height
        self.cell_size = cell_size
        self.HEADER_HEIGHT = 60
        
        self.CONSTRAINT_WIDTH = 250 
        self.window_size_x = self.view_width_cells * self.cell_size + self.CONSTRAINT_WIDTH
        self.window_size_y = self.view_height_cells * self.cell_size + self.HEADER_HEIGHT
        
        self.window = None
        self.clock = None
        self.cell_font = None
        self.header_font = None
        self.constraint_font = None
        self.flag_icon = None

        self.colors = {
            "header_bg": (40, 40, 40),
            "header_text": (255, 255, 255),
            "closed": (180, 180, 180),
            "flag": (255, 165, 0),
            "open": (220, 220, 220),
            "grid": (100, 100, 100),
            "mine": (200, 0, 0),
            "text": (0, 0, 0),
            1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0), 4: (0, 0, 128),
            5: (128, 0, 0), 6: (0, 128, 128), 7: (0, 0, 0), 8: (128, 128, 128)
        }

    def _init_pygame(self):
        if 'pygame' not in sys.modules: return
        pygame.init()
        pygame.display.set_caption("Minesweeper AI Environment")
        self.window = pygame.display.set_mode((self.window_size_x, self.window_size_y))
        self.clock = pygame.time.Clock()
        
        font_size = int(self.cell_size * 0.8)
        self.cell_font = pygame.font.SysFont('Arial', font_size, bold=True)
        self.header_font = pygame.font.SysFont('Arial', 20)
        self.constraint_font = pygame.font.SysFont('Consolas', 14)

        try:
            self.flag_icon = pygame.image.load("flag_icon.png")
            icon_size = int(self.cell_size * 0.6)
            self.flag_icon = pygame.transform.scale(self.flag_icon, (icon_size, icon_size))
        except:
            self.flag_icon = None

    def render_frame(self, board_getter, camera_x, camera_y, score, status, constraints=None):
        if 'pygame' not in sys.modules: return
        if self.window is None:
            self._init_pygame()
        
        self.window.fill(self.colors["header_bg"])
        
        # Header
        header_text = f"Score: {score}"
        if status == "win": header_text += " | WIN!"
        elif status == "loss": header_text += " | GAME OVER"
        
        score_surf = self.header_font.render(header_text, True, self.colors["header_text"])
        self.window.blit(score_surf, (20, self.HEADER_HEIGHT // 2 - 10))
        
        # Grid
        for r in range(self.view_height_cells):
            for c in range(self.view_width_cells):
                world_x = camera_x + c
                world_y = camera_y + r
                
                val = board_getter(world_y, world_x)
                
                rect = pygame.Rect(c * self.cell_size, 
                                   r * self.cell_size + self.HEADER_HEIGHT, 
                                   self.cell_size, self.cell_size)
                
                if val == CLOSED or val == FLAG:
                    pygame.draw.rect(self.window, self.colors["closed"], rect)
                elif val == MINE:
                    pygame.draw.rect(self.window, self.colors["mine"], rect)
                else:
                    pygame.draw.rect(self.window, self.colors["open"], rect)
                
                pygame.draw.rect(self.window, self.colors["grid"], rect, 1)

                if val == FLAG:
                    if self.flag_icon:
                        icon_rect = self.flag_icon.get_rect(center=rect.center)
                        self.window.blit(self.flag_icon, icon_rect)
                    else:
                        pygame.draw.circle(self.window, self.colors["flag"], rect.center, self.cell_size//4)
                elif val == MINE:
                    pygame.draw.circle(self.window, (0,0,0), rect.center, self.cell_size//3)
                elif val > 0:
                    text_color = self.colors.get(val, self.colors["text"])
                    text = self.cell_font.render(str(val), True, text_color)
                    text_rect = text.get_rect(center=rect.center)
                    self.window.blit(text, text_rect)

        # Constraints
        sidebar_rect = pygame.Rect(
            self.view_width_cells * self.cell_size, 
            self.HEADER_HEIGHT, 
            self.CONSTRAINT_WIDTH, 
            self.window_size_y - self.HEADER_HEIGHT
        )
        pygame.draw.rect(self.window, (30, 30, 30), sidebar_rect)
        
        title = self.header_font.render("Active Constraints", True, (255, 255, 255))
        self.window.blit(title, (sidebar_rect.x + 10, sidebar_rect.y + 10))

        if constraints:
            y_offset = sidebar_rect.y + 40
            for i, (cells, bombs) in enumerate(constraints[:20]):
                cell_str = str(cells).replace("),", "), ")
                if len(cell_str) > 20: cell_str = cell_str[:18] + "..."
                
                txt = f"{bombs} mine{'s' if bombs!=1 else ''} in {len(cells)} cells"
                tsurf = self.constraint_font.render(txt, True, (200, 200, 200))
                self.window.blit(tsurf, (sidebar_rect.x + 10, y_offset))
                
                csurf = self.constraint_font.render(cell_str, True, (150, 150, 150))
                self.window.blit(csurf, (sidebar_rect.x + 10, y_offset + 15))
                y_offset += 40

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None

# --- Standard Fixed Env ---
class MinesweeperDiscreetEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 10}

    def __init__(self, board_size=10, num_mines=10, render_mode=None):
        self.board_size = board_size
        self.num_mines = num_mines
        self.board = None
        self.my_board = np.ones((board_size, board_size), dtype=int) * CLOSED
        self.total_reward = 0
        self.flags_placed = 0
        self.game_over_status = None
        self.first_move_made = False
        self.current_constraints = []
        
        self.render_mode = render_mode
        self.visualizer = None
        if self.render_mode == "human":
            self.visualizer = MinesweeperVisualizer(board_size, board_size)
        
        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Box(low=-3, high=8, shape=(board_size, board_size), dtype=int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.my_board = np.ones((self.board_size, self.board_size), dtype=int) * CLOSED
        self.board = None
        self.total_reward = 0
        self.flags_placed = 0
        self.game_over_status = None
        self.first_move_made = False
        self.current_constraints = []
        return self.my_board, {}

    def step(self, action):
        x = int(action / self.board_size)
        y = int(action % self.board_size)
        
        if not self.first_move_made:
            self.first_move_made = True
            self.board = place_mines_safely(self.board_size, self.num_mines, x, y)

        if self.my_board[x,y] != CLOSED:
            self.current_constraints = generate_constraints(self.my_board, self.board_size)
            return self.my_board, -1, False, False, {}

        if is_mine(self.board, x, y):
            self.my_board[x,y] = MINE
            self.game_over_status = "loss"
            self.current_constraints = generate_constraints(self.my_board, self.board_size)
            return self.my_board, -100, True, False, {}
        
        self._reveal(x, y)
        self.current_constraints = generate_constraints(self.my_board, self.board_size)

        if is_win(self.my_board, self.num_mines):
            self.game_over_status = "win"
            return self.my_board, 1000, True, False, {}
            
        return self.my_board, 1, False, False, {}

    def _reveal(self, x, y):
        if not is_valid(x, y, self.board_size): return
        if self.my_board[x, y] != CLOSED: return
        
        mines = 0
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0: continue
                nx, ny = x+dx, y+dy
                if is_valid(nx, ny, self.board_size) and is_mine(self.board, nx, ny):
                    mines += 1
        
        self.my_board[x, y] = mines
        if mines == 0:
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if dx!=0 or dy!=0: self._reveal(x+dx, y+dy)

    def toggle_flag(self, x, y):
        if not is_valid(x, y, self.board_size): return
        if self.my_board[x, y] == CLOSED:
            self.my_board[x, y] = FLAG
            self.flags_placed += 1
        elif self.my_board[x, y] == FLAG:
            self.my_board[x, y] = CLOSED
            self.flags_placed -= 1
        self.current_constraints = generate_constraints(self.my_board, self.board_size)
        if self.render_mode == "human": self.render()

    def render(self):
        if self.render_mode == "human" and self.visualizer:
            def getter(r, c):
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    return self.my_board[r, c]
                return CLOSED
            self.visualizer.render_frame(getter, 0, 0, self.total_reward, self.game_over_status, self.current_constraints)

    def close(self):
        if self.visualizer: self.visualizer.close()

# --- INFINITE CHUNK-BASED ENV ---
class MinesweeperInfiniteEnv:
    CHUNK_SIZE = 16
    DENSITY = 0.15

    def __init__(self, render_mode="human", view_w=20, view_h=15):
        self.render_mode = render_mode
        self.mines = set()
        self.flags = set()
        self.revealed = {} 
        self.generated_chunks = set()
        self.game_over_status = None
        self.score = 0
        self.view_w = view_w
        self.view_h = view_h
        self.visualizer = None
        if self.render_mode == "human":
            self.visualizer = MinesweeperVisualizer(view_w, view_h)

    def _get_chunk_coords(self, r, c):
        return r // self.CHUNK_SIZE, c // self.CHUNK_SIZE

    def _generate_chunk(self, cr, cc, safe_r=None, safe_c=None):
        if (cr, cc) in self.generated_chunks: return
        for r in range(cr * self.CHUNK_SIZE, (cr + 1) * self.CHUNK_SIZE):
            for c in range(cc * self.CHUNK_SIZE, (cc + 1) * self.CHUNK_SIZE):
                if safe_r is not None and abs(r - safe_r) <= 1 and abs(c - safe_c) <= 1:
                    continue
                if random() < self.DENSITY:
                    self.mines.add((r, c))
        self.generated_chunks.add((cr, cc))

    def _ensure_area_generated(self, r, c, safe_mode=False):
        cr, cc = self._get_chunk_coords(r, c)
        for dcr in [-1, 0, 1]:
            for dcc in [-1, 0, 1]:
                if safe_mode and dcr == 0 and dcc == 0:
                    self._generate_chunk(cr + dcr, cc + dcc, r, c)
                else:
                    self._generate_chunk(cr + dcr, cc + dcc)

    def get_cell_value(self, r, c):
        if (r, c) in self.revealed: return self.revealed[(r, c)]
        if (r, c) in self.flags: return FLAG
        return CLOSED

    def reset(self):
        self.mines.clear()
        self.flags.clear()
        self.revealed.clear()
        self.generated_chunks.clear()
        self.game_over_status = None
        self.score = 0
        return {}

    def step(self, r, c):
        if self.game_over_status: return []
        
        is_first_move = (len(self.revealed) == 0)
        self._ensure_area_generated(r, c, safe_mode=is_first_move)

        if (r, c) in self.flags: return []
        
        if (r, c) in self.mines:
            self.revealed[(r, c)] = MINE
            self.game_over_status = "loss"
            self.score = len(self.revealed)
            return []
        
        newly_revealed = []
        if (r, c) not in self.revealed:
            self._reveal_recursive(r, c, newly_revealed)
            
        self.score = len(self.revealed)
        return newly_revealed

    def _reveal_recursive(self, r, c, newly_revealed):
        if (r, c) in self.revealed: return
        
        mine_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr==0 and dc==0: continue
                if (r+dr, c+dc) in self.mines: mine_count += 1
        
        self.revealed[(r, c)] = mine_count
        newly_revealed.append((r, c))
        
        if mine_count == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr==0 and dc==0: continue
                    nr, nc = r+dr, c+dc
                    self._ensure_area_generated(nr, nc)
                    self._reveal_recursive(nr, nc, newly_revealed)

    def toggle_flag(self, r, c):
        if (r, c) in self.revealed: return
        if (r, c) in self.flags: self.flags.remove((r, c))
        else: self.flags.add((r, c))

    def render(self, camera_x, camera_y):
        if self.visualizer:
            self.visualizer.render_frame(self.get_cell_value, camera_x, camera_y, self.score, self.game_over_status, [])

    def close(self):
        if self.visualizer: self.visualizer.close()