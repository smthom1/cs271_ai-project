# COMPSCI 271P: Minesweeper AI
### **Team 9:** Eric Yao Huang, Samantha Juteram, Midhuna Mohanraj, Sophia Thompson

## How to Run
Ensure that all `.py` files (including `minesweeper.py`, `constants.py`, and `constraints.py`) are in the same directory.

### 1. Manual Play (Human vs. Game)
To play the game yourself using the visual interface:
```
python run_game.py
````

*A popup will appear asking you to choose between **Standard Mode** (Fixed Grid) or **Infinite Mode** (Expanding).*

-----

### 2\. Infinite AI Agents (Expanding Map)

These agents play on a map that generates forever. Choose your version:

  * **Balanced Agent (Recommended):** Smart, consistent, and visualizes survival stats.
    ```
    python agent_inf_balanced.py
    ```
  * **Speed Agent (50/50):** Extremely fast but risky. Good for stress-testing speed.
    ```
    python agent_inf_50-50.py
    ```

-----

### 3\. Standard AI Agents (Fixed 10x10 Grid)

These agents play on the classic Minesweeper board to test win/loss rates.

  * **Single Game Demo:** Watch the agent play one game in real-time.
    ```
    python agent.py
    ```
  * **Performance Evaluation:** Simulates 100 fast games and generates success metrics/graphs.
    ```
    python agent_eval.py
    ```

-----

## AI Strategy (How steps are chosen)

The agent uses a **Constraint Satisfaction Problem (CSP)** solver to navigate the grid. The decision-making process follows a strict hierarchy:

1.  **Frontier Detection:** The agent identifies "boundary" cells—revealed numbers that have unrevealed neighbors.
2.  **Constraint Solving:** It groups these cells into independent components and mathematically solves for all valid mine arrangements.
3.  **Action Selection:**
      * **Guaranteed Moves:** If a cell is safe in *every* valid solution, it is revealed immediately. If a cell is a mine in *every* valid solution, it is flagged.
      * **Probabilistic Guessing:** If no guaranteed moves exist, the agent calculates the exact probability of a mine for every boundary cell. It then picks the safest option.
      * **Blind Guessing:** As a last resort, it picks a random boundary cell to expand the map.

### Infinite Agent Versions

  * **Speed Version (`agent_inf_50-50.py`):** Prioritizes raw speed. It limits the math solver to 50 attempts and checks only 3 local areas before guessing. High volatility.
  * **Balanced Version (`agent_inf_balanced.py`):** Trades speed for intelligence. It searches 5x as many possibilities (250 attempts) and scans 15 local areas per step. Drastically reduces unforced errors.

## Scoring System

  * **Infinite Mode:** Score = **Total safe cells revealed**. The goal is to expand as far as possible before dying.
  * **Standard Mode:** You win by flagging all mines and revealing all safe cells on the fixed board.

## Requirements

Install the necessary dependencies:

```
pip install numpy gymnasium pygame six pandas matplotlib
```

## Acknowledgements

Based on A. Aylin Tokuc's gym-minesweeper environment.
  - Repository: http://github.com/aylint/gym-minesweeper