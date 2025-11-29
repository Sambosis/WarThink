# WarThink

WarThink is a turn-based strategy game on a 10x10 grid, designed for training an AI agent using reinforcement learning. Two players, each controlling a small army of 5 units, compete to be the last one standing. The project uses `stable-baselines3` with a PPO (Proximal Policy Optimization) model to learn the game, featuring asymmetric self-play and a policy pool for diverse opponents.

![WarThink Gameplay](https://i.imgur.com/example.png)  <!-- Replace with actual gameplay image -->

## Gameplay

The objective of the game is to eliminate all of the opponent's units.

- **Game Board:** A 10x10 grid.
- **Players:** Two players, Player 1 (Blue) and Player 2 (Red).
- **Turns:** The game is turn-based. It ends if one player has no units left.
- **Turn Limit & Tie-Breaker:** If the game reaches 200 turns, a tie-breaker is triggered. The player with more units remaining on the board is declared the winner. If both players have the same number of units, the game is a true draw.
- **Armies:** Each player starts with an army of 5 units in opposite corners of the board.

### Units

There are three types of units, each with unique stats:

| Unit Type | Health (HP) | Attack Damage | Attack Range | Move Range |
| :--- | :--- | :--- | :--- | :--- |
| **Warrior** | 3 | 1 | 1 | 1 |
| **Archer** | 3 | 1 | 3 | 1 |
| **Commander**| 5 | 2 | 1 | 2 |

Each army consists of:
- 2x Warriors
- 2x Archers
- 1x Commander

## Actions

On each turn, the active player issues one command to each of their 5 units simultaneously.

### Action Types

1.  **Stay (Action 0):** The unit holds its position.
2.  **Move (Actions 1-4):** Move North, South, East, or West.
    - Warriors and Archers move 1 square.
    - Commanders move 2 squares.
    - **Collision Rule:** If multiple friendly units attempt to move to the exact same square, none of them will move.
3.  **Attack (Actions 5-8):** Attack in one of the four cardinal directions (North, South, East, West).
    - The attack hits the **first** enemy unit encountered in a straight line in that direction, up to the unit's maximum range.

## Visuals & UI

The game is rendered in an isometric view.

- **Player Colors:** Player 1 is **Blue**, and Player 2 is **Red**.
- **Unit Appearance:**
    - Units are represented by diamond shapes with a surrounding "glow" of their player's color.
    - A health bar is displayed above each unit.
    - A small circle in the center of the unit indicates its type:
        - **Gold Circle:** Commander
        - **White Circle:** Warrior or Archer

- **UI Overlays:**
    - The top-left corner displays the current turn count and active player.
    - During evaluation phases, training statistics like the current episode, average reward, and P1's win rate are shown.
    - At the end of a match, the winner (`Player 1 WINS!`, `Player 2 WINS!`, or `DRAW!`) is displayed prominently in the center of the screen.

## Reinforcement Learning Environment

The AI agent is trained using the Gymnasium API.

### Observation Space

The agent perceives the game state as a `10x10x5` tensor:
- **Channel 0:** Turn Indicator (1.0 for the current player).
- **Channel 1:** Player 1's unit types (normalized).
- **Channel 2:** Player 1's unit HP (normalized).
- **Channel 3:** Player 2's unit types (normalized).
- **Channel 4:** A combination of Player 2's unit HP and a mask indicating valid move/attack locations for the current player.

### Action Space

The agent outputs an action for each of its 5 units, represented by a `MultiDiscrete([9] * 5)` space. Each of the 5 actions is an integer from 0 to 8, mapping to either a "Stay/Move" or "Attack" command.

### Network Architecture

The agent uses a custom Convolutional Neural Network (CNN) to process the grid-based observations:
- **Input:** 10x10x5 tensor.
- **Convolutional Layers:**
    - Conv2d (32 filters, 3x3 kernel, stride 1, padding 1) + ReLU
    - Conv2d (64 filters, 3x3 kernel, stride 1, padding 1) + ReLU
- **Fully Connected Layer:** Flattened output -> Linear layer (256 features) + ReLU.
- **Policy Head:** Maps features to the multi-discrete action space.

### Reward System

The agent is incentivized to learn through a system of rewards and penalties designed to be symmetric for both players.

- **Dense Rewards (during gameplay):**
    - `+1.0` for each point of damage dealt to an enemy unit.
    - `-1.0` for each point of damage taken from an enemy unit on their previous turn.
    - `+20.0` for killing an enemy unit.
    - `-0.05` penalty for each turn taken (to encourage efficiency).

- **Terminal Rewards (at the end of the game):**
    - **Annihilation Win:** `+200.0` (Eliminating all enemy units).
    - **Attrition Win:** `+5.0` (Having more units alive at turn 200).
    - **Annihilation Loss:** `-200.0`.
    - **Attrition Loss:** `-5.0`.
    - **Draw:** `0.0`.

## Self-Play & Pool Evolution

To ensure robust learning and prevent cycling or overfitting to a static opponent, the project implements an asymmetric self-play mechanism with a diverse pool of policies.

### Policy Pool
The agent maintains a pool of policies (models):
- **Pool[0] (Main Agent):** The current best policy that is actively being trained.
- **Pool[1:] (Opponents):** A collection of historical or mutated policies used as opponents.

### Evolution Mechanism
The pool evolves periodically (e.g., every 100 episodes) based on performance:
1.  **Performance Tracking:** The system tracks the win/loss record of every policy in the pool.
2.  **Normalization:** Performance scores are normalized by the number of games played to ensure fairness.
3.  **Selection (Elitism):** The top 2 performing policies are identified as "elites".
4.  **Mutation:** Policies that are *not* elites are mutated by adding Gaussian noise to their parameters. This introduces diversity and new strategies into the pool.
5.  **Promotion:** The best-performing policy is promoted to `Pool[0]` (if it wasn't already) to become the new main training agent. This ensures the agent always learns from the strongest available strategy.

### Training Process
- **Player 1 (Blue):** Always controlled by the Main Agent (`Pool[0]`).
- **Player 2 (Red):** Controlled by a policy randomly selected from the opponent pool (`Pool[1:]`).
- **Noise Injection:** To further encourage robustness, Gaussian noise is occasionally added to Player 2's actions during training.
- **Phases:**
    - **Quick Learning:** Every 10 episodes, the agent trains for 2,048 timesteps.
    - **Intensive Learning:** Every 100 episodes, the agent trains for 20,000 timesteps.

## How to Run

1.  **Installation:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Training:**
    To start the training process, run:
    ```bash
    python main.py
    ```
    - The agent will train and periodically save model checkpoints to the `models/` directory every 500 episodes.
    - Every 100 episodes, an evaluation game will be rendered on-screen. You can close the window or press `ESC` to resume training.

3.  **Watching a Pre-trained Model:**
    (Instructions to be added on how to load and watch a specific model)
