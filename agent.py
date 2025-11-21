import sys
import numpy as np
import time
import random
from collections import defaultdict
from minesweeper import MinesweeperDiscreetEnv
from constraints import generate_constraints
from constants import CLOSED, FLAG, MINE

# csp solver logic

def solve_csp(board_state):
    board_size = board_state.shape[0]
    
    # get constraints from helper
    all_constraints = generate_constraints(board_state, board_size)
    
    if not all_constraints:
        return set(), set()

    # map variables to constraints
    var_to_constraints = defaultdict(list)
    all_vars = set()
    
    for i, (vars_in_constraint, needed) in enumerate(all_constraints):
        for v in vars_in_constraint:
            var_to_constraints[v].append(i)
            all_vars.add(v)
            
    # find connected components
    components = []
    visited = set()
    
    for v in all_vars:
        if v in visited:
            continue
            
        # bfs to group interacting vars
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

    # solve independent components
    safe_moves = set()
    flag_moves = set()
    
    for comp_vars in components:
        # filter relevant constraints
        comp_constraints = []
        seen_constraint_indices = set()
        for v in comp_vars:
            for c_idx in var_to_constraints[v]:
                if c_idx not in seen_constraint_indices:
                    seen_constraint_indices.add(c_idx)
                    comp_constraints.append(all_constraints[c_idx])
        
        # run backtracking solver
        solutions = backtracking_solve(comp_vars, comp_constraints)
        
        if not solutions:
            continue

        # check for certainties across all solutions
        for i, v in enumerate(comp_vars):
            can_be_safe = False
            can_be_mine = False
            
            for sol in solutions:
                if sol[v] == 0: can_be_safe = True
                if sol[v] == 1: can_be_mine = True
            
            if can_be_safe and not can_be_mine:
                safe_moves.add(v)
            if can_be_mine and not can_be_safe:
                flag_moves.add(v)

    return safe_moves, flag_moves

def backtracking_solve(variables, constraints):
    solutions = []
    
    # sort variables by most constrained
    var_counts = defaultdict(int)
    for v_list, _ in constraints:
        for v in v_list: var_counts[v] += 1
    variables.sort(key=lambda v: -var_counts[v])
    
    assignment = {} 

    def is_valid(assignment):
        for v_list, limit in constraints:
            current_sum = 0
            unassigned = 0
            for v in v_list:
                if v in assignment:
                    current_sum += assignment[v]
                else:
                    unassigned += 1
            
            # too many mines
            if current_sum > limit: 
                return False
            # not enough space for required mines
            if current_sum + unassigned < limit:
                return False
        return True

    def backtrack(idx):
        # cap max solutions
        if len(solutions) > 1000: 
            return

        if idx == len(variables):
            solutions.append(assignment.copy())
            return

        curr_var = variables[idx]
        
        # try 0 (safe) and 1 (mine)
        for val in [0, 1]:
            assignment[curr_var] = val
            if is_valid(assignment):
                backtrack(idx + 1)
        
        del assignment[curr_var]

    backtrack(0)
    return solutions

# main game loop

def run_agent_game():
    env = MinesweeperDiscreetEnv(render_mode="human")
    observation, info = env.reset()
    done = False
    
    print("--- CSP AGENT GAME START ---")
    
    # random start
    first_r = random.randint(0, env.board_size - 1)
    first_c = random.randint(0, env.board_size - 1)
    first_action = first_r * env.board_size + first_c
    print(f"Random Start: ({first_r}, {first_c})")
    observation, reward, done, truncated, info = env.step(first_action)
    
    while not done:
        print(f"Thinking... (Board State Analysis)")
        
        # call csp solver
        safe, flags = solve_csp(observation)
        
        made_move = False
        
        # execute flags
        for (r, c) in flags:
            if observation[r, c] == CLOSED:
                print(f"CSP says FLAG: ({r},{c})")
                env.toggle_flag(r, c)
                made_move = True
        
        # execute safe moves
        for (r, c) in safe:
            if observation[r, c] == CLOSED:
                print(f"CSP says SAFE: ({r},{c})")
                action = r * env.board_size + c
                observation, _, done, _, _ = env.step(action)
                made_move = True
                if done: break
        
        # guess if stuck
        if not made_move and not done:
            print("CSP found no guaranteed moves. Guessing...")
            closed_r, closed_c = np.where(observation == CLOSED)
            if len(closed_r) == 0: break
            
            idx = random.randint(0, len(closed_r) - 1)
            r, c = closed_r[idx], closed_c[idx]
            
            print(f"Guessing: ({r},{c})")
            observation, _, done, _, _ = env.step(r * env.board_size + c)

    print("\n--- GAME OVER ---")
    print(f"Result: {env.game_over_status}")
    print(f"Final Score: {env.total_reward}")
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    run_agent_game()
