import gymnasium as gym
import numpy as np
from game import GameState, generate_obs, resolve_actions
from typing import Tuple, Dict, Any
from config import cfg


class WarGameEnv(gym.Env):
    """
    Gymnasium-compatible environment for the WarGame turn-based strategy game.
    Wraps GameState logic, provides 5x10x10 observation tensor, processes MultiDiscrete actions
    for 5 units (stay + 4 moves + 4 attacks), computes dense rewards for the acting player,
    alternates turns, handles termination. Headless; rendering in separate Renderer.
    """
    metadata = {'render_modes': ['human'], 'render_fps': cfg.env.render_fps}

    def __init__(self):
        super().__init__()
        # Updated to (5, 10, 10) for CNN
        self.observation_space = gym.spaces.Box(
            low=0.0, high=3.0, shape=(cfg.env.n_units, cfg.env.grid_size, cfg.env.grid_size), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([9] * cfg.env.n_units)
        self.state = GameState()
        self.current_player = 1
        self.damage_taken_by_p1 = 0
        self.damage_taken_by_p2 = 0

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to the initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options (unused).

        Returns:
            Initial observation and empty info dict.
        """
        super().reset(seed=seed)
        self.state.reset()
        self.current_player = 1
        self.damage_taken_by_p1 = 0
        self.damage_taken_by_p2 = 0
        obs = generate_obs(self.state)
        return obs, {}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes simultaneous actions for the current player's 5 units.

        Args:
            actions: np.ndarray of shape (5,) with Discrete(9) actions per unit.

        Returns:
            Tuple of (next_obs, reward, terminated, truncated, info).
        """
        if actions.shape != (cfg.env.n_units,):
            raise ValueError(f"Expected actions shape ({cfg.env.n_units},), got {actions.shape}")

        acting_player = self.state.current_player
        own_units = self.state.get_current_units()
        enemy_units = self.state.get_enemy_units()

        # Get damage taken by acting player from opponent's last turn
        damage_to_own = self.damage_taken_by_p1 if acting_player == 1 else self.damage_taken_by_p2
        if acting_player == 1:
            self.damage_taken_by_p1 = 0
        else:
            self.damage_taken_by_p2 = 0

        # Resolve actions and calculate damage dealt to enemy
        prev_enemy_hp = sum(u.hp for u in enemy_units)
        prev_enemy_count = sum(1 for u in enemy_units if u.alive)
        resolve_actions(self.state, actions)
        post_enemy_hp = sum(u.hp for u in enemy_units)
        post_enemy_count = sum(1 for u in enemy_units if u.alive)
        damage_to_enemy = prev_enemy_hp - post_enemy_hp
        killed_units = prev_enemy_count - post_enemy_count

        # Store damage dealt for the other player's next turn
        if acting_player == 1:
            self.damage_taken_by_p2 = damage_to_enemy
        else:
            self.damage_taken_by_p1 = damage_to_enemy
        ######################################################
        # Dense reward for acting player (symmetric damage reward/penalty)
        reward = (damage_to_enemy * cfg.env.damage_scale) - (damage_to_own * cfg.env.damage_scale) + (killed_units * cfg.env.kill_bonus) + cfg.env.step_penalty
        ######################################################

        # Increment turn count
        self.state.turn_count += 1

        # Switch current player
        self.state.current_player = 2 if self.state.current_player == 1 else 1

        # Next observation
        obs = generate_obs(self.state)

        # Check game over
        terminated = self.state.is_done()
        truncated = False

        info = {
            'acting_player': acting_player,
            'winner': self.state.winner if terminated else None,
            'turn_count': self.state.turn_count,
            'reward_components': {
                'damage_enemy': damage_to_enemy,
                'damage_own': damage_to_own,
                'step_penalty': cfg.env.step_penalty
            }
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """No-op; headless for training. Use Trainer for visualization."""
        pass


class SelfPlayWrapper(gym.Wrapper):
    """
    Wraps WarGameEnv to handle opponent moves internally using a policy pool.
    This makes the environment appear as a single-agent environment to the training agent.
    """
    def __init__(self, env, policy_pool=None):
        super().__init__(env)
        self.policy_pool = policy_pool
        self.opponent_policy = None
        self.opponent_idx = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Select opponent policy if pool is available
        if self.policy_pool and len(self.policy_pool) > 1:
            self.opponent_idx = np.random.randint(1, len(self.policy_pool))
            self.opponent_policy = self.policy_pool[self.opponent_idx]
        else:
            self.opponent_policy = None # Random play or simple heuristic could be fallback
            
        # If P2 starts (unlikely in current reset logic, but for robustness), we need to act
        if self.env.unwrapped.current_player == 2:
             obs, _, done, _, _ = self.step(None) # Trigger opponent move
             if done:
                 return self.reset(**kwargs)
                 
        return obs, info

    def step(self, action):
        # 1. Agent (P1) moves
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            # Calculate terminal rewards for P1
            winner = self.env.unwrapped.state.winner
            win_condition = self.env.unwrapped.state.win_condition
            
            bonus = 0.0
            if winner in [1, 2]:
                bonus = cfg.env.annihilation_bonus if win_condition == 'annihilation' else cfg.env.attrition_bonus
                
                if winner == 1:
                    reward += bonus
                else: # winner == 2
                    reward -= bonus # Penalize losing to encourage winning
            elif winner == 0:
                reward += cfg.env.draw_penalty

            return obs, reward, terminated, truncated, info
            
        # 2. Opponent (P2) moves
        # We need to loop until it's P1's turn again or game over
        # (In this game, turns alternate strictly 1->2->1, so one step is usually enough,
        # but let's be safe)
        
        while self.env.unwrapped.current_player == 2:
            # Get opponent action
            if self.opponent_policy:
                # Opponent sees canonical obs relative to themselves (P2)
                # The env.step() already updated obs to be canonical for the NEXT player (P2)
                # So 'obs' is already what P2 sees.
                p2_action, _ = self.opponent_policy.predict(obs, deterministic=False)
            else:
                # Fallback: random action
                p2_action = self.env.action_space.sample()
                
            obs, p2_reward, terminated, truncated, p2_info = self.env.step(p2_action)
            
            if terminated or truncated:
                winner = self.env.unwrapped.state.winner
                win_condition = self.env.unwrapped.state.win_condition
                info['winner'] = winner
                
                # If game ended during opponent turn, we still need to give terminal reward to P1
                # P1 is the learning agent.
                bonus = 0.0
                if winner in [1, 2]:
                    bonus = cfg.env.annihilation_bonus if win_condition == 'annihilation' else cfg.env.attrition_bonus
                    
                    if winner == 1:
                        reward += bonus
                    else: # winner == 2
                        reward -= bonus
                elif winner == 0:
                    reward += cfg.env.draw_penalty
                
                break
                
        return obs, reward, terminated, truncated, info