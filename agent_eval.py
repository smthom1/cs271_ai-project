import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from minesweeper import MinesweeperDiscreetEnv, CLOSED, FLAG, is_valid

# Copy of agent logic from agent.py so we can run games automatically

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

#Modified run_agent_game from agent.py that tracks stats for single game 

def run_single_eval_game():
    
    # changed the render_mode to none for the eval to make the eval faster, i.e no window popup for each eval
    env = MinesweeperDiscreetEnv(render_mode="None")
    
    # get initial (empty) observation
    observation, info = env.reset()
    board_state = observation
    done = False
    
    #Initializing stats
    total_moves = 0
    good_moves = 0
    bad_moves = 0
    
    # Make first move
    
    # choose rand (row, col)
    first_r = random.randint(0, env.board_size - 1)
    first_c = random.randint(0, env.board_size - 1)
    
    # convert (r, c) to discrete action integer
    first_action = first_r * env.board_size + first_c
    
    print(f"Making random first move at: ({first_r}, {first_c})")
    
    # take first step
    observation, reward, done, truncated, info = env.step(first_action)

    # check if first move was good (first move wasnt mine)
    if not done:
        good_moves += 1
    
    # Main agent loop
    while not done:    
        made_a_move_in_loop = True # flag tracks if we're getting anywhere
        
        while made_a_move_in_loop and not done:
            
            made_a_move_in_loop = False # Assume STUCK until move found
            board_state = observation   # get current board state
            
            # Apply "safe" strategy
            safe_moves_to_make = find_safe_moves(board_state)
            
            if safe_moves_to_make:
                made_a_move_in_loop = True # found move!
                
                # reveal ALL safe cells found in this pass
                for (r, c) in safe_moves_to_make:
                    # check if cell is still closed (should be, but just to make sure)
                    if observation[r, c] == CLOSED and not done:
                        action = r * env.board_size + c
                        observation, reward, done, truncated, info = env.step(action)

                        if not done or env.game_over_status == "win":
                            good_moves += 1
                
                continue 

            # Apply "Flag" Strategy, only run if NO safe moves found
            flag_moves_to_make = find_flag_moves(board_state)
            
            if flag_moves_to_make:
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
            # find all available (CLOSED) cells
            closed_r, closed_c = np.where(board_state == CLOSED)
            
            # are any cells left to guess?
            if len(closed_r) == 0:
                break # No choice but to stop
            
            # pick rand cell from list of closed cells
            random_index = random.randint(0, len(closed_r) - 1)
            guess_r = closed_r[random_index]
            guess_c = closed_c[random_index]
                        
            # conv(r, c) to  discrete action
            guess_action = guess_r * env.board_size + guess_c
            
            observation, reward, done, truncated, info = env.step(guess_action)

            if not done or env.game_over_status == "win":
                good_moves += 1
        
    won = (env.game_over_status == "win")
    
    env.close()
    
    return {
        'won': won,
        'good_moves': good_moves
    }

# Now simulate multiple games

def run_evaluation(num_games=100):
    print(f"\nRunning {num_games} games...")

    results = []

# play all the games
    for i in range(num_games):
        if (i + 1) % 10 == 0:  # show progress every 10 games
            print(f"Completed {i+1}/{num_games} games...")
        
        game_result = run_single_eval_game()
        results.append(game_result)

    # win rate calc
    wins = sum(1 for r in results if r['won'])
    losses = num_games - wins
    win_rate = (wins / num_games) * 100
    
    # get num good moves for games where agent lost 
    loss_results = [r for r in results if not r['won']]
    good_moves_in_losses = [r['good_moves'] for r in loss_results] if loss_results else []
    
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
    
    print(f"\n--- PRIMARY METRIC ---")
    print(f"Win Rate:     {stats['win_rate']}%")
    
    if stats['losses'] > 0:
        print(f"\n--- SECONDARY METRIC (only for losses) ---")
        print(f"Avg Good Moves Before Dying: {stats['avg_good_moves_when_lost']:.2f}")

def make_graphs(stats):

    fig, axes = plt.subplots(1,2, figsize =(12,5))

    #Win/Loss pie chart
    axes[0].pie([stats['wins'],stats['losses']],
                 labels =['Wins','Losses'])
    axes[0].set_title(f"Win Rate: {stats['win_rate']}%",fontsize=14, fontweight='bold')

    #Histo of good moves for lost games
    if stats['losses']>0:
        good_moves = stats['good_moves_in_losses']

        axes[1].hist(good_moves, bins=15, edgecolor='black')
        axes[1].set_title('Good Moves Before Death (Losses Only)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Number of Good Moves')
        axes[1].set_ylabel('Number of Games')
    else:
        # case that agent wins every game 
        axes[1].text(0.5, 0.5, 'No Losses!\nAgent won every game!', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
        axes[1].set_title('Good Moves Before Death (Losses Only)', fontsize=12, fontweight='bold')

    plt.savefig('evaluation_results.png', dpi=300)
    print("Graphs saved as 'evaluation_results.png'")
    plt.show()

# Saving eval to CSV 
def save_results_to_csv(stats, filename='evaluation_results.csv'):
    eval_data={
        'Metric': ['Total Games', 'Wins', 'Losses', 'Win Rate (%)', 'Avg Good Moves (Losses Only)'],
        'Value': [
            stats['num_games'],
            stats['wins'],
            stats['losses'],
            f"{stats['win_rate']}",
            f"{stats['avg_good_moves_when_lost']:.2f}"
        ]
    }
    eval_df = pd.DataFrame(eval_data)
    eval_df.to_csv(filename, index=False)
    print(f"Summary saved to '{filename}'")

    #Breakdown for lost games
    if stats['good_moves_in_losses']:
        loss_data = {
                'Game Number': range(1, len(stats['good_moves_in_losses']) + 1),
                'Result': ['Loss'] * len(stats['good_moves_in_losses']),
                'Good Moves': stats['good_moves_in_losses']
            }
        loss_df = pd.DataFrame(loss_data)
        loss_df.to_csv('loss_game_results.csv', index=False)
        print(f"Loss results saved to 'loss_game_results.csv'")
    
# run everything
if __name__ == "__main__":
    NUM_GAMES = 100 
    stats = run_evaluation(num_games=NUM_GAMES)
    print_results(stats)
    make_graphs(stats)
    save_results_to_csv(stats)