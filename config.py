
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class EnvConfig:
    render_fps: int = 60
    grid_size: int = 8
    n_units: int = 5
    max_turns: int = 500
    
    # Rewards
    damage_scale: float = 2.0
    kill_bonus: float = 300.0
    step_penalty: float = -7.0
    
    # Terminal Bonuses (handled in Trainer usually, but good to have here)
    annihilation_bonus: float = 1000.0
    attrition_bonus: float = 10.0
    draw_penalty: float = -10.0

@dataclass
class RLConfig:
    pool_size: int = 12
    noise_std: float = 0.1
    n_envs: int = 10
    
    # PPO Hyperparameters
    n_steps: int = 400        # Steps per env per update (buffer size = n_steps * n_envs = 4000)
    batch_size: int = 1000     # Minibatch size for gradient update
    learning_rate: float = 5e-4 # Optimizer step size
    n_epochs: int = 5         # Number of passes over the buffer per update
    gamma: float = 0.995       # Discount factor for future rewards (0.995 = long horizon)
    gae_lambda: float = 0.995  # Factor for Generalized Advantage Estimation (bias-variance trade-off)
    clip_range: float = 0.8   # PPO clip parameter for trust region (prevents large policy updates)
    ent_coef: float = 0.03     # Entropy coefficient (higher = more exploration)
    features_dim: int = 256    # Dimension of the feature vector output by the CNN
    
    @property
    def ppo_params(self) -> Dict[str, Any]:
        return {
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
        }

@dataclass
class TrainerConfig:
    stats_window: int = 250
    
    # Training Loop
    # Training Loop
    quick_learn_steps: int = 4000
    intensive_learn_steps: int = 32000
    
    quick_learn_freq: int = 100
    intensive_learn_freq: int = 1000
    checkpoint_freq: int = 1000
    
    eval_fps: int = 60

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

# Global configuration instance
cfg = Config()
