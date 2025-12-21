import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from copy import deepcopy
from typing import List, Optional

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch.multiprocessing as mp
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
        n_envs = cfg.rl.n_envs
        def make_env():
            return SelfPlayWrapper(WarGameEnv(), self.policy_pool)
            
        if n_envs > 1:
            # SubprocVecEnv for multi-core speedup
            # Note: We use a wrapper or ensure the pool is shared/synced if needed
            self.vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
        else:
            self.vec_env = DummyVecEnv([make_env])

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
        self.p1_rewards_history: List[float] = []

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
        self.p1_rewards_history.append(p1_reward)
        
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

        recent_rewards = self.p1_rewards_history[-200:] if len(self.p1_rewards_history) >= 200 else self.p1_rewards_history
        if not recent_rewards:
            return
            
        recent_avg = np.mean(recent_rewards)
        recent_max = np.max(recent_rewards)
        log(f"Pool evolution triggered. Active Learner Recent Avg Reward: {recent_avg:.3f}, Max Reward: {recent_max:.3f}")
        
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
        
        # Determine number of policies to mutate (25% of pool, rounded up, at least 1)
        n_mutations = max(1, int(np.ceil(len(self.policy_pool) * 0.25)))
        
        # Sort indices by performance (ascending: worst -> best)
        sorted_indices = np.argsort(normalized_performances)
        worst_idx = sorted_indices[0] # Absolute worst
        best_idx = sorted_indices[-1] # Absolute best
        
        # --- Step 1: Handle P0 Preservation ---
        # If P0 is NOT the best, but is better than the worst, save it.
        # We overwrite the 'worst' slot. Note: The 'worst' slot will be a candidate for mutation later?
        # Ideally, we save P0 to a slot that ISN'T about to be mutated immediately, 
        # OR we save it to a slot and protect that slot.
        # LIMITATION: If we overwrite worst, it might be mutated in Step 3 if we select bottom N.
        # FIX: We will exclude 'saved_p0_slot' from mutation if we just saved to it.
        
        saved_p0_slot = None
        
        if best_idx != 0:
            p0_score = normalized_performances[0]
            worst_score = normalized_performances[worst_idx]
            
            if p0_score > worst_score:
                log(f"Active Learner (P0) is decent (score: {p0_score:.3f} > {worst_score:.3f}). Saving P0 to slot P{worst_idx}.")
                p0_params = self.policy_pool[0].get_parameters()
                self.policy_pool[worst_idx].set_parameters(p0_params)
                saved_p0_slot = worst_idx
            else:
                log(f"Active Learner (P0) is worst (score: {p0_score:.3f}). Discarding.")

        # --- Step 2: Promote Best to P0 ---
        # Note: If best_idx was 0, this is a no-op, which is fine.
        if best_idx != 0:
            best_params = self.policy_pool[best_idx].get_parameters()
            self.policy_pool[0].set_parameters(best_params)
            self.model = self.policy_pool[0]
            log(f"Best policy {best_idx} promoted to main (slot 0) - This is now the ACTIVE LEARNER")
        else:
             log(f"Active Learner (P0) is already the Best. Keeping it.")

        # --- Step 3: Mutate Weakest ---
        # We want to mutate the bottom N performers.
        # But we must NOT mutate:
        # 1. P0 (The new active learner, our best hope)
        # 2. saved_p0_slot (Unless we didn't save anything) -- WAIT. 
        #    If we saved P0 to worst_idx, effectively worst_idx is now a COPY of the old P0.
        #    Do we want to mutate it immediately? NO. We saved it because it was decent.
        #    So we should treat 'saved_p0_slot' as protected.
        
        candidates = []
        # Walk through sorted indices from worst to best
        for idx in sorted_indices:
            idx = int(idx)
            if len(candidates) >= n_mutations:
                break
            
            # Exclusions
            if idx == 0: continue # Don't mutate active learner
            if idx == saved_p0_slot: continue # Don't mutate the just-saved decent policy
            
            candidates.append(idx)
            
        log(f"Mutating bottom {len(candidates)} policies: {candidates}")
        for idx in candidates:
             self.mutate_policy(self.policy_pool[idx])
        
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

    def run_parallel_episodes(self, min_episodes: int) -> List[dict]:
        """
        Run episodes in parallel using the vectorized environment.
        Collects at least min_episodes results.
        
        Returns list of dicts with:
            {
                'p1_reward': float,
                'p2_reward': float,
                'winner': int,
                'steps': int,
                'condition': str
            }
        """
        n_envs = self.vec_env.num_envs
        current_rewards = np.zeros((n_envs, 2)) # [p1, p2]
        current_steps = np.zeros(n_envs, dtype=int)
        results = []
        
        # We need to manually reset or assume reset state.
        # VecEnv is always running, but we might want to start fresh or just stream.
        # Streaming is better for throughput. 
        # But we need an initial 'obs' if not tracking it.
        # VecEnv doesn't expose 'current_obs' easily unless we tracked it.
        # Let's force a reset for simplicity of logic, roughly okay.
        obs = self.vec_env.reset()
        
        while len(results) < min_episodes:
            actions, _ = self.model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = self.vec_env.step(actions)
            
            for i in range(n_envs):
                p1_r = rewards[i]
                current_rewards[i, 0] += p1_r
                # Assuming P2 reward is symmetric or we don't track it precisely here without info
                # But we need it for update_pool.
                
                current_steps[i] += 1
                
                if dones[i]:
                    info = infos[i]
                    winner = info.get('winner', 0)
                    
                    # Estimate P2 reward if not provided (zero-sum approx)
                    total_p1 = current_rewards[i, 0]
                    total_p2 = 0.0 
                    
                    self.update_pool([total_p1, total_p2])
                    
                    results.append({
                        'winner': winner,
                        'steps': current_steps[i],
                        'condition': 'unknown', 
                        'p1_reward': total_p1,
                        'p2_reward': total_p2
                    })
                    
                    current_rewards[i] = 0
                    current_steps[i] = 0
                    
        return results

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass