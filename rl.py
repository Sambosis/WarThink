import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from copy import deepcopy
from typing import List, Optional

from stable_baselines3.common.vec_env import DummyVecEnv
from env import WarGameEnv, SelfPlayWrapper
from config import cfg

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for 10x10 grid observations.
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))

class PPOSelfPlayAgent:
    """
    Handles PPO model with asymmetric self-play, population pool, and noise injection.
    
    pool[0]: main clean policy for P1
    pool[1:]: noisy variants for P2 (fixed idx per episode)
    Evolves every 100 episodes: top-2 selected, others mutated, best -> pool[0]
    """

    def __init__(self, env: WarGameEnv, pool_size: int = cfg.rl.pool_size, noise_std: float = cfg.rl.noise_std):
        self.env = env
        
        # Device detection including TPU
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            try:
                import torch_xla.core.xla_model as xm
                self.device = xm.xla_device()
                print(f"TPU detected: {self.device}")
            except ImportError:
                pass

        # Policy pool setup
        # We need a base model to initialize the pool
        # We use a dummy env for initialization, but actual training uses VecEnv
        self.policy_pool: List[PPO] = []
        
        # Create the vectorized environment for training
        # The lambda captures self.policy_pool, so the wrapper will have access to the pool
        # even as it gets updated.
        n_envs = cfg.rl.n_envs
        self.vec_env = DummyVecEnv([lambda: SelfPlayWrapper(WarGameEnv(), self.policy_pool) for _ in range(n_envs)])

        # Base PPO model with standard params
        ppo_params = cfg.rl.ppo_params
        ppo_params.update({
            'policy': 'CnnPolicy',
            'env': self.vec_env,
            'verbose': 0,
            'device': self.device,
            'policy_kwargs': {
                'normalize_images': False,
                'features_extractor_class': CustomCNN,
                'features_extractor_kwargs': {'features_dim': cfg.rl.features_dim}
            }
        })
        
        base_model = PPO(**ppo_params)
        self.policy_pool.append(base_model)
        
        # Create other pool models
        # They don't strictly need the VecEnv, but they need compatible observation space
        # We can just use the same params.
        for _ in range(1, pool_size):
            new_model = PPO(**ppo_params)
            new_model.set_parameters(base_model.get_parameters())
            self.policy_pool.append(new_model)
            
        self.model = self.policy_pool[0]  # Main model reference

        # Tracking stats
        self.episode_rewards: List[float] = []
        self.episode_count: int = 0
        self.pool_performances: List[float] = [0.0] * pool_size
        self.pool_usage_count: List[int] = [0] * pool_size
        self.noise_std = noise_std
        self.current_p2_idx: Optional[int] = None

    def start_episode(self):
        """Select fixed P2 policy index for entire episode."""
        self.current_p2_idx = np.random.randint(1, len(self.policy_pool))

    def act(self, obs: np.ndarray, player: int) -> np.ndarray:
        """
        Args:
            obs: np.ndarray (10,10,5)
            player: 1 or 2
            
        Returns:
            actions: np.ndarray (5,) Discrete(9)
        """
        if player == 1:
            model = self.policy_pool[0]
        else:
            if self.current_p2_idx is None:
                # Fallback if start_episode wasn't called (e.g. in eval)
                model = self.policy_pool[np.random.randint(1, len(self.policy_pool))]
            else:
                model = self.policy_pool[self.current_p2_idx]
        
        # Stochastic prediction
        actions, _ = model.predict(obs, deterministic=False)
        actions = actions.astype(np.int32)
        
        # Action noise for P2 only
        if player == 2 and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=(5,))
            actions = np.clip(actions + noise, 0, 8).astype(np.int32)
        
        return actions

    def update_pool(self, rewards: List[float]):
        """
        Credit episode rewards to used policies, increment usage.
        
        Args:
            rewards: [p1_reward, p2_reward]
        """
        if len(rewards) != 2:
            print("Warning: update_pool called with invalid rewards.")
            return
            
        # If we are in training loop via learn(), this might not be called manually per episode
        # But Trainer.play_episode calls it.
        # If we use learn(), the VecEnv wrapper handles the game, but it doesn't update pool stats directly?
        # The wrapper returns rewards to the agent.
        # We need a way to track pool performance during 'learn()'.
        # The SelfPlayWrapper could track it, or we just rely on the manual evaluation episodes for evolution stats.
        # Given the structure, let's rely on Trainer's manual episodes for stats for now.
        
        if self.current_p2_idx is None:
             return
        
        p1_reward, p2_reward = rewards
        p1_idx = 0
        p2_idx = self.current_p2_idx
        
        self.episode_rewards.extend(rewards)
        
        self.pool_performances[p1_idx] += p1_reward
        self.pool_performances[p2_idx] += p2_reward
        
        self.pool_usage_count[p1_idx] += 1
        self.pool_usage_count[p2_idx] += 1
        
        self.episode_count += 1
        self.current_p2_idx = None

    def evolve_pool(self, logger: Optional[callable] = None):
        """Evolve if episode_count % 100 == 0: normalize perf/usage, top-2 elite, mutate others, promote best to [0]."""
        # Note: episode_count here tracks MANUALLY played episodes.
        
        def log(msg):
            if logger:
                logger(msg)
            else:
                print(msg)

        recent_rewards = self.episode_rewards[-200:] if len(self.episode_rewards) >= 200 else self.episode_rewards
        if not recent_rewards:
            return
            
        recent_avg = np.mean(recent_rewards)
        log(f"Pool evolution triggered. Recent avg reward: {recent_avg:.3f}")
        
        # Normalize performances by usage
        normalized_performances = np.array([
            perf / max(usage, 1.0)
            for perf, usage in zip(self.pool_performances, self.pool_usage_count)
        ])
        
        # Log all performances for visibility
        perf_str = ", ".join([f"P{i}:{p:.3f}" for i, p in enumerate(normalized_performances)])
        log(f"Pool Performance: {perf_str}")

        top_indices = np.argsort(normalized_performances)[-2:]
        top_scores = normalized_performances[top_indices]
        scores_str = ", ".join([f"{s:.3f}" for s in top_scores])
        log(f"Top performers: {top_indices} (scores: {scores_str})")
        
        # Mutate only the worst performer OR save P0
        worst_idx = np.argmin(normalized_performances)
        best_idx = int(top_indices[-1])
        
        if best_idx == 0:
            # Case 1: Active Learner is Best
            # Keep P0. Mutate worst to maintain diversity.
            target_idx = worst_idx
            if target_idx == 0: # If pool size 1 or all equal
                 non_top = [i for i in range(len(self.policy_pool)) if i != 0]
                 if non_top: target_idx = np.random.choice(non_top)
            
            if target_idx != 0:
                log(f"Active Learner (P0) is Best! Keeping P0. Mutating worst policy P{target_idx} (score: {normalized_performances[target_idx]:.3f})")
                self.mutate_policy(self.policy_pool[target_idx])
            else:
                log("Active Learner (P0) is Best and dominant. No mutation performed.")
                
        else:
            # Active Learner is NOT the best.
            # Check if it is better than the worst
            p0_score = normalized_performances[0]
            worst_score = normalized_performances[worst_idx]
            
            if p0_score > worst_score:
                # Case 2: Active Learner is Middle (Better than Worst)
                # Save P0 by overwriting the Worst
                log(f"Active Learner (P0) is decent (score: {p0_score:.3f} > {worst_score:.3f}). Saving P0 to slot P{worst_idx} (replacing worst).")
                p0_params = self.policy_pool[0].get_parameters()
                self.policy_pool[worst_idx].set_parameters(p0_params)
            else:
                # Case 3: Active Learner is Worst
                log(f"Active Learner (P0) is worst (score: {p0_score:.3f}). Discarding.")

            # Promote Best to P0
            best_params = self.policy_pool[best_idx].get_parameters()
            self.policy_pool[0].set_parameters(best_params)
            self.model = self.policy_pool[0]
            log(f"Best policy {best_idx} promoted to main (slot 0) - This is now the ACTIVE LEARNER")
        
        # Reset stats
        self.pool_performances = [0.0] * len(self.policy_pool)
        self.pool_usage_count = [0] * len(self.policy_pool)

    def mutate_policy(self, model: PPO):
        """Apply Gaussian noise to policy parameters."""
        with torch.no_grad():
            for param in model.policy.parameters():
                if param.requires_grad:
                    noise = torch.normal(0, self.noise_std, param.shape, device=self.device)
                    param.add_(noise)

    def learn(self, total_timesteps: int):
        """PPO learning: full on main."""
        if total_timesteps <= 0:
            return
        # Main policy training using VecEnv
        # The VecEnv wrapper handles opponent moves using the pool
        self.model.set_env(self.vec_env)
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str):
        """Save main model (pool[0])."""
        self.model.save(path)

    def load(self, path: str):
        """Load main model and deepcopy to pool."""
        # We need to load into the model attached to the VecEnv or just load params
        loaded_model = PPO.load(path, env=self.vec_env)
        self.model = loaded_model
        self.policy_pool[0] = self.model
        
        # Load params into other models
        base_params = self.model.get_parameters()
        for i in range(1, len(self.policy_pool)):
            self.policy_pool[i].set_parameters(base_params)