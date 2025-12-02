import numpy as np
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from minesweeper import MinesweeperDiscreetEnv, is_valid
from constants import CLOSED, FLAG
from constraints import generate_constraints

# config
RENDER_DELAY = 0.1 # delay between moves

def solve_csp(board_state):
    board_size = board_state.shape[0]
    
    # 1. generate constraints based on revealed numbers
    all_constraints = generate_constraints(board_state, board_size)
    
    if not all_constraints:
        return set(), set()

    # 2. map variables (hidden cells) to the constraints they belong to
    var_to_constraints = defaultdict(list)
    all_vars = set()
    
    for i, (vars_in_constraint, needed) in enumerate(all_constraints):
        for v in vars_in_constraint:
            var_to_constraints[v].append(i)
            all_vars.add(v)
            
    # 3. find connected components (groups of variables that interact)
    # variables are connected if they share a constraint
    components = []
    visited = set()
    
    for v in all_vars:
        if v in visited: continue
            
        # bfs to gather all connected variables for this component
        component_vars = set([v])
        q = [v]
        visited.add(v)
        
        while q:
            curr = q.pop(0)
            for c_idx in var_to_constraints[curr]:
                c_vars, _ = all_constraints[c_idx]
                for neighbor in c_vars:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component_vars.add(neighbor)
                        q.append(neighbor)
        components.append(list(component_vars))

    # 4. solve each component independently
    safe_moves = set()
    flag_moves = set()
    
    for comp_vars in components:
        # gather only the constraints relevant to this specific component
        comp_constraints = []
        seen_constraint_indices = set()
        for v in comp_vars:
            for c_idx in var_to_constraints[v]:
                if c_idx not in seen_constraint_indices:
                    seen_constraint_indices.add(c_idx)
                    comp_constraints.append(all_constraints[c_idx])
        
        # find all valid mine arrangements for this component
        solutions = backtracking_solve(comp_vars, comp_constraints)
        if not solutions: continue

        # check for unanimous agreement across all solutions
        for i, v in enumerate(comp_vars):
            can_be_safe = False
            can_be_mine = False
            
            for sol in solutions:
                if sol[v] == 0: can_be_safe = True # it's SAFE in at least one solution
                if sol[v] == 1: can_be_mine = True # it's a MINE in at least one solution
            
            # if it CAN be safe but NEVER mine -> guaranteed safe
            if can_be_safe and not can_be_mine: safe_moves.add(v)
            # if it CAN be mine but NEVER safe -> guaranteed mine
            if can_be_mine and not can_be_safe: flag_moves.add(v)

    return safe_moves, flag_moves

def backtracking_solve(variables, constraints):
    solutions = []
    
    # heuristic: sort variables by how many constraints they appear in
    # this causes conflicts to happen earlier, pruning the tree faster
    var_counts = defaultdict(int)
    for v_list, _ in constraints:
        for v in v_list: var_counts[v] += 1
    variables.sort(key=lambda v: -var_counts[v])
    
    assignment = {} 

    def is_valid(assignment):
        # check if current assignment violates any constraint
        for v_list, limit in constraints:
            current_sum = 0
            unassigned = 0
            for v in v_list:
                if v in assignment: current_sum += assignment[v]
                else: unassigned += 1
            
            # violation 1: placed mines exceed the number on the board
            if current_sum > limit: return False
            # violation 2: remaining empty spots aren't enough to satisfy the number
            if current_sum + unassigned < limit: return False
        return True

    def backtrack(idx):
        # limit solutions to prevent hanging on large open areas
        if len(solutions) > 1000: return
        
        # base case: all variables assigned successfully
        if idx == len(variables):
            solutions.append(assignment.copy())
            return

        curr_var = variables[idx]
        # try assigning 0 (safe) then 1 (mine)
        for val in [0, 1]:
            assignment[curr_var] = val
            if is_valid(assignment): backtrack(idx + 1)
        del assignment[curr_var]

    backtrack(0)
    return solutions

# --- EVALUATION ---

def run_single_eval_game():
    env = MinesweeperDiscreetEnv(render_mode="human")
    observation, info = env.reset()
    board_state = observation
    done = False
    good_moves = 0
    total_clicks = 0 

    # track start time
    start_time = time.time()
    
    # 0. start with a random move to open the board
    first_r = random.randint(0, env.board_size - 1)
    first_c = random.randint(0, env.board_size - 1)
    first_action = first_r * env.board_size + first_c
    observation, reward, done, truncated, info = env.step(first_action)
    env.total_reward += reward # update score manually

    if env.render_mode == "human":
        env.render()
        time.sleep(RENDER_DELAY)

    if not done: good_moves += 1

    done = False
    while not done:    
        made_logic_move = True
        
        # logic loop: keep applying logic until stuck
        while made_logic_move and not done:
            made_logic_move = False
            safe, flags = solve_csp(observation)
            
            if safe or flags:
                made_logic_move = True
                
                # 1. apply guaranteed flags (csp said 'always mine')
                for (r, c) in flags:
                    if observation[r, c] == CLOSED: 
                        env.toggle_flag(r, c)
                        if env.render_mode == "human": # pause after flag
                            env.render()
                            time.sleep(RENDER_DELAY)
                
                # 2. apply guaranteed safe moves (csp said 'always safe')
                for (r, c) in safe:
                    if observation[r, c] == CLOSED and not done:
                        action = r * env.board_size + c
                        observation, reward, done, truncated, info = env.step(action)
                        env.total_reward += reward # update score manually
                        
                        if not done or env.game_over_status == "win": good_moves += 1
                        
                        if env.render_mode == "human": # pause after safe move
                            env.render()
                            time.sleep(RENDER_DELAY)

        # 3. logic failed? forced to guess
        if not done:
            closed_r, closed_c = np.where(board_state == CLOSED)
            if len(closed_r) == 0: break
            
            # pick a random closed cell
            random_index = random.randint(0, len(closed_r) - 1)
            guess_action = closed_r[random_index] * env.board_size + closed_c[random_index]
            observation, reward, done, truncated, info = env.step(guess_action)
            env.total_reward += reward # update score manually
            
            if env.render_mode == "human": # pause after guess
                env.render()
                time.sleep(RENDER_DELAY)

            if not done or env.game_over_status == "win": good_moves += 1
        
    # Track end time
    end_time = time.time()
    elapsed_time = end_time - start_time

    won = (env.game_over_status == "win")
    env.close()
    return {
        'won': won,
        'good_moves': good_moves,
        'elapsed_time': elapsed_time,
    }

def run_evaluation(num_games=5): # reduced count for human viewing
    print(f"\nRunning {num_games} games...")
    results = []
    for i in range(num_games):
        if (i + 1) % 1 == 0: print(f"Starting game {i+1}/{num_games}...")
        results.append(run_single_eval_game())

    wins = sum(1 for r in results if r['won'])
    losses = num_games - wins
    win_rate = (wins / num_games) * 100
    
    loss_results = [r for r in results if not r['won']]
    good_moves_in_losses = [r['good_moves'] for r in loss_results]
    
    # avg time for games that were won
    won_results = [r for r in results if r['won']]
    avg_time_to_win = np.mean([r['elapsed_time'] for r in won_results]) if won_results else 0

    return {
        'num_games': num_games,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_time_to_win': avg_time_to_win,
        'good_moves_in_losses': good_moves_in_losses,
        'avg_good_moves_when_lost': np.mean(good_moves_in_losses) if good_moves_in_losses else 0
    }

def print_results(stats):
    print(f"\nTotal Games:  {stats['num_games']}")
    print(f"Wins:         {stats['wins']}")
    print(f"Losses:       {stats['losses']}")
    print(f"Win Rate:     {stats['win_rate']}%")
    if stats['wins'] > 0:
        print(f"Avg Time to Win: {stats['avg_time_to_win']:.2f} seconds")
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
    stats = run_evaluation(num_games=5) # set to 5 for demo
    print_results(stats)
    make_graphs(stats)
