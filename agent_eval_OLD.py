# agent_eval.py
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time
from minesweeper import MinesweeperDiscreetEnv, is_valid
from constants import CLOSED, FLAG

# config
RENDER_DELAY = 0.1 # delay between moves

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
                    for (nr, nc) in closed_neighbors: safe_moves.add((nr, nc))
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
                    for (nr, nc) in closed_neighbors: flag_moves.add((nr, nc))
    return flag_moves

def run_single_eval_game():
    env = MinesweeperDiscreetEnv(render_mode="human") 
    observation, info = env.reset()
    board_state = observation
    done = False
    good_moves = 0
    
    # 0. start with random move
    first_r = random.randint(0, env.board_size - 1)
    first_c = random.randint(0, env.board_size - 1)
    first_action = first_r * env.board_size + first_c
    observation, reward, done, truncated, info = env.step(first_action)

    if env.render_mode == "human":
        env.render()
        time.sleep(RENDER_DELAY)

    if not done: good_moves += 1
    
    # main loop
    while not done:    
        made_a_move_in_loop = True
        
        # logic loop: keep applying logic until stuck
        while made_a_move_in_loop and not done:
            made_a_move_in_loop = False
            board_state = observation
            
            # 1. try guaranteed safe moves (priority)
            safe_moves = find_safe_moves(board_state)
            
            if safe_moves:
                made_a_move_in_loop = True
                for (r, c) in safe_moves:
                    if observation[r, c] == CLOSED and not done:
                        action = r * env.board_size + c
                        observation, reward, done, truncated, info = env.step(action)
                        if not done or env.game_over_status == "win": good_moves += 1
                        
                        if env.render_mode == "human":
                            env.render()
                            time.sleep(RENDER_DELAY)
                continue # if we moved, re-scan board immediately

            # 2. if no safe moves, try guaranteed flags
            flag_moves = find_flag_moves(board_state)
            if flag_moves:
                made_a_move_in_loop = True
                for (r, c) in flag_moves:
                    if observation[r, c] == CLOSED: 
                        env.toggle_flag(r, c)
                        
                        if env.render_mode == "human":
                            env.render()
                            time.sleep(RENDER_DELAY)

                observation = env.my_board
                continue # flags might reveal new safe moves, so re-scan
        
        # 3. logic failed (stuck)? forced to guess randomly
        if not done:
            closed_r, closed_c = np.where(board_state == CLOSED)
            if len(closed_r) == 0: break
            
            random_index = random.randint(0, len(closed_r) - 1)
            guess_action = closed_r[random_index] * env.board_size + closed_c[random_index]
            observation, reward, done, truncated, info = env.step(guess_action)

            if env.render_mode == "human":
                env.render()
                time.sleep(RENDER_DELAY)

            if not done or env.game_over_status == "win": good_moves += 1
        
    won = (env.game_over_status == "win")
    env.close()
    return {'won': won, 'good_moves': good_moves}

def run_evaluation(num_games=100):
    print(f"\nRunning {num_games} games...")
    results = []
    for i in range(num_games):
        print(f"Starting game {i+1}/{num_games}...")
        results.append(run_single_eval_game())

    wins = sum(1 for r in results if r['won'])
    losses = num_games - wins
    win_rate = (wins / num_games) * 100
    
    loss_results = [r for r in results if not r['won']]
    good_moves_in_losses = [r['good_moves'] for r in loss_results]
    
    return {
        'num_games': num_games,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'good_moves_in_losses': good_moves_in_losses,
        'avg_good_moves_when_lost': np.mean(good_moves_in_losses) if good_moves_in_losses else 0
    }

def print_results(stats):
    print(f"\nTotal Games:  {stats['num_games']}")
    print(f"Wins:         {stats['wins']}")
    print(f"Losses:       {stats['losses']}")
    print(f"Win Rate:     {stats['win_rate']}%")
    if stats['losses'] > 0:
        print(f"Avg Good Moves Before Dying: {stats['avg_good_moves_when_lost']:.2f}")

def make_graphs(stats):
    try:
        fig, axes = plt.subplots(1,2, figsize =(12,5))
        axes[0].pie([stats['wins'],stats['losses']], labels =['Wins','Losses'], autopct='%1.1f%%')
        axes[0].set_title(f"Win Rate: {stats['win_rate']}%")

        if stats['losses']>0:
            axes[1].hist(stats['good_moves_in_losses'], bins=10, edgecolor='black')
            axes[1].set_title('Good Moves Before Death (Losses Only)')
        else:
            axes[1].text(0.5, 0.5, 'No Losses!', ha='center')

        plt.savefig('evaluation_results.png')
        print("Graphs saved as 'evaluation_results.png'")
    except Exception as e:
        print(f"Could not generate graphs: {e}")

if __name__ == "__main__":
    stats = run_evaluation(num_games=5) 
    print_results(stats)
    make_graphs(stats)
