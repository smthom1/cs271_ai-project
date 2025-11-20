import time
import random
import sys
import tkinter as tk
from tkinter import simpledialog
from statistics import mean
import matplotlib.pyplot as plt
from minesweeper import MinesweeperInfiniteEnv

# config
SOLVER_MAX_SOLUTIONS = 50 
CHUNK_SIZE = 16            
MAX_LOCAL_SEARCHES = 3     

# setup
NEIGHBOR_DELTAS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
sys.setrecursionlimit(5000)

# cache
frontier_cells = set()

def get_neighbors(r, c):
    for dr, dc in NEIGHBOR_DELTAS:
        yield r + dr, c + dc

def update_frontier(env, changed):
    candidates = set(changed)
    for r, c in changed:
        for nr, nc in get_neighbors(r, c):
            candidates.add((nr, nc))

    for r, c in candidates:
        if (r, c) not in env.revealed:
            is_frontier = False
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) in env.revealed and env.revealed[(nr, nc)] != 0:
                    is_frontier = True
                    break
            if is_frontier: frontier_cells.add((r, c))
            elif (r, c) in frontier_cells: frontier_cells.remove((r, c))
            continue
        
        val = env.revealed[(r, c)]
        if val == 0:
            if (r, c) in frontier_cells: frontier_cells.remove((r, c))
            continue
        
        has_unknown = False
        for nr, nc in get_neighbors(r, c):
            if (nr, nc) not in env.revealed and (nr, nc) not in env.flags:
                has_unknown = True
                break
        if has_unknown: frontier_cells.add((r, c))
        elif (r, c) in frontier_cells: frontier_cells.remove((r, c))

# fast pass

def solve_trivial(env):
    safe = set()
    flag = set()
    
    if len(frontier_cells) > 50:
        check_list = random.sample(list(frontier_cells), 50)
    else:
        check_list = list(frontier_cells)

    revealed_boundary = [x for x in check_list if x in env.revealed]
    
    for r, c in revealed_boundary:
        val = env.revealed[(r, c)]
        unknowns = []
        flag_count = 0
        
        for nr, nc in get_neighbors(r, c):
            if (nr, nc) in env.flags: flag_count += 1
            elif (nr, nc) not in env.revealed: unknowns.append((nr, nc))
        
        if not unknowns: continue

        if val == flag_count + len(unknowns):
            for u in unknowns: flag.add(u)
        elif val == flag_count:
            for u in unknowns: safe.add(u)
            
    return safe, flag

# local window builder

def get_local_chunk(env):
    if not frontier_cells: return None, []

    hidden_frontier = [x for x in frontier_cells if x not in env.revealed]
    if not hidden_frontier: return None, []
    
    seed = random.choice(hidden_frontier)
    
    chunk_vars = {seed}
    q = [seed]
    
    while q and len(chunk_vars) < CHUNK_SIZE:
        curr = q.pop(0)
        for n in get_neighbors(*curr):
            if n in frontier_cells and n not in env.revealed and n not in chunk_vars:
                chunk_vars.add(n)
                q.append(n)
    
    chunk_list = list(chunk_vars)
    constraints = []
    chunk_set = set(chunk_list)
    
    relevant_numbers = set()
    for cv in chunk_list:
        for n in get_neighbors(*cv):
            if n in env.revealed: relevant_numbers.add(n)
            
    for r, c in relevant_numbers:
        val = env.revealed[(r, c)]
        unknowns = []
        flag_count = 0
        fully_contained = True
        
        for nr, nc in get_neighbors(r, c):
            if (nr, nc) in env.flags: 
                flag_count += 1
            elif (nr, nc) not in env.revealed:
                if (nr, nc) in chunk_set:
                    unknowns.append(chunk_list.index((nr, nc)))
                else:
                    fully_contained = False
        
        if unknowns and fully_contained:
            constraints.append((val - flag_count, unknowns))
            
    return chunk_list, constraints

def solve_component_smart(coords, constraints):
    if not constraints: return [] 
    
    n = len(coords)
    var_counts = [0] * n
    for needed, vars in constraints:
        for v in vars: var_counts[v] += 1
        
    sorted_indices = sorted(range(n), key=lambda i: -var_counts[i])
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}
    sorted_constraints = []
    for needed, vars in constraints:
        new_vars = tuple(sorted(old_to_new[v] for v in vars))
        sorted_constraints.append((needed, new_vars))
        
    var_to_cons = [[] for _ in range(n)]
    for i, (needed, vars) in enumerate(sorted_constraints):
        for v in vars: var_to_cons[v].append((needed, vars))

    solutions = []
    assignment = [-1] * n

    def solve(idx):
        if len(solutions) >= SOLVER_MAX_SOLUTIONS: return
        if idx == n:
            original_order_sol = [0] * n
            for new_i, val in enumerate(assignment):
                original_order_sol[sorted_indices[new_i]] = val
            solutions.append(tuple(original_order_sol))
            return

        for val in [0, 1]:
            assignment[idx] = val
            valid = True
            for needed, vars in var_to_cons[idx]:
                curr_sum = 0
                unassigned_count = 0
                for v in vars:
                    v_val = assignment[v]
                    if v_val == 1: curr_sum += 1
                    elif v_val == -1: unassigned_count += 1
                
                if curr_sum > needed: 
                    valid = False; break
                if curr_sum + unassigned_count < needed: 
                    valid = False; break
            
            if valid:
                solve(idx + 1)
                if len(solutions) >= SOLVER_MAX_SOLUTIONS: return
        assignment[idx] = -1
    solve(0)
    return solutions

def solve_local(env):
    for _ in range(MAX_LOCAL_SEARCHES):
        coords, constraints = get_local_chunk(env)
        if not coords or not constraints: continue
        
        sols = solve_component_smart(coords, constraints)
        if not sols: continue
        
        total = len(sols)
        counts = [0] * len(coords)
        for s in sols:
            for i, val in enumerate(s): counts[i] += val
        
        safe = set()
        flag = set()
        found_action = False
        
        for i, c in enumerate(counts):
            real_coord = coords[i]
            if c == 0: 
                safe.add(real_coord)
                found_action = True
            elif c == total: 
                flag.add(real_coord)
                found_action = True
        
        if found_action: return safe, flag
            
    return set(), set()

# visualization & stats

def show_results(all_stats):
    if not all_stats: return
    
    scores = [s['Score'] for s in all_stats]
    avg_score = mean(scores)
    max_score = max(scores)
    n = len(scores)
    
    print("\n" + "="*40)
    print("       AGGREGATE STATISTICS       ")
    print("="*40)
    print(f" Total Games:   {n}")
    print(f" Average Score: {avg_score:.2f}")
    print(f" Max Score:     {max_score}")
    print("="*40 + "\n")
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- Plot 1: survival probability (score > X) ---
        sorted_scores = sorted(scores)
        thresholds = [0] + sorted_scores
        survival_rates = [100.0] 
        
        for t in sorted_scores:
            count = sum(1 for s in scores if s >= t)
            pct = (count / n) * 100
            survival_rates.append(pct)
        
        ax1.step(thresholds, survival_rates, where='post', color='green', linewidth=2)
        ax1.fill_between(thresholds, survival_rates, step='post', alpha=0.3, color='green')
        
        ax1.set_title('Survival Probability (Score > X)', fontsize=14)
        ax1.set_xlabel('Score Threshold', fontsize=12)
        ax1.set_ylabel('% of Games Reaching Score', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        milestones = [100, 500, 1000, 2500, 5000, 10000]
        for m in milestones:
            if m < max_score:
                rate = sum(1 for s in scores if s >= m) / n * 100
                if rate > 1.0: 
                    ax1.annotate(f'{m}: {rate:.1f}%', xy=(m, rate), xytext=(m, rate+10),
                                 arrowprops=dict(facecolor='black', arrowstyle='->'))

        # --- Plot 2: score distribution histogram ---
        ax2.hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(avg_score, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_score:.1f}')
        
        ax2.set_title('Score Distribution', fontsize=14)
        ax2.set_xlabel('Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle(f'Speed/50-50 Agent Performance ({n} Runs)', fontsize=16)
        plt.tight_layout()
        print(" Displaying updated charts...")
        plt.show()
    except Exception as e:
        print(f" Could not display chart: {e}")

def get_num_runs():
    root = tk.Tk()
    root.withdraw()
    num = simpledialog.askinteger("setup", "how many games to run?", minvalue=1, maxvalue=10000)
    root.destroy()
    return num if num else 1

# runner

def run():
    num_runs = get_num_runs()
    all_stats = []
    
    for i in range(num_runs):
        print(f"\n--- game {i+1}/{num_runs} ---")
        
        env = MinesweeperInfiniteEnv(render_mode="None")
        frontier_cells.clear()
        
        initial_revealed = env.step(0,0)
        update_frontier(env, initial_revealed)
        
        steps = 0
        running = True
        print("starting...", flush=True)

        try:
            while running and not env.game_over_status:
                changed = []
                did_something = False
                
                # 1. trivial pass
                safe, flags = solve_trivial(env)
                if safe or flags:
                    for r,c in flags:
                        if (r,c) not in env.flags:
                            env.toggle_flag(r,c)
                            changed.append((r,c))
                            did_something = True
                    for r,c in safe:
                        if (r,c) not in env.revealed:
                            new_rev = env.step(r,c)
                            if new_rev: changed.extend(new_rev)
                            did_something = True
                
                # 2. local pass
                if not did_something:
                    safe, flags = solve_local(env)
                    if safe or flags:
                        for r,c in flags:
                            if (r,c) not in env.flags:
                                env.toggle_flag(r,c)
                                changed.append((r,c))
                                did_something = True
                        for r,c in safe:
                            if (r,c) not in env.revealed:
                                new_rev = env.step(r,c)
                                if new_rev: changed.extend(new_rev)
                                did_something = True
                    
                    # 3. fast guess
                    else:
                        frontier_list = [x for x in frontier_cells if x not in env.revealed]
                        if frontier_list:
                            sample = random.sample(frontier_list, min(len(frontier_list), 20))
                            sample.sort(key=lambda x: sum(1 for n in get_neighbors(*x) if n not in env.revealed))
                            f = sample[0]
                            ns = [n for n in get_neighbors(*f) if n not in env.revealed and n not in env.flags]
                            if ns:
                                g = random.choice(ns)
                                new_rev = env.step(*g)
                                if new_rev: changed.extend(new_rev)

                update_frontier(env, changed)
                steps += 1

                if steps % 50 == 0:
                    sys.stdout.write(f"\rrunning... steps: {steps} | score: {env.score}")
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("\nstopped by user")
            break

        sys.stdout.write("\r" + " " * 40 + "\r") 
        print(f"final score: {env.score}")
        print(f"total steps: {steps}")
        
        all_stats.append({
            'Game': i + 1,
            'Score': env.score,
            'Steps': steps
        })
        
        env.close()

    # end of all runs
    show_results(all_stats)

if __name__ == "__main__":
    run()