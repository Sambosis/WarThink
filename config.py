
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class EnvConfig:
    render_fps: int = 60
    grid_size: int = 8
    n_units: int = 5
    max_turns: int = 600
    
    # Rewards
    damage_scale: float = 1.0
    kill_bonus: float = 100.0
    step_penalty: float = -3.0
    
    # Terminal Bonuses (handled in Trainer usually, but good to have here)
    annihilation_bonus: float = 1000.0
    attrition_bonus: float = 10.0
    draw_penalty: float = -50.0

@dataclass
class RLConfig:
    pool_size: int = 4
    noise_std: float = 0.05
    n_envs: int = 4
    
    # PPO Hyperparameters
    n_steps: int = 2048
    batch_size: int = 4096
    learning_rate: float = 2.5e-4
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.995
    clip_range: float = 0.2
    ent_coef: float = 0.02
    features_dim: int = 256
    
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
    stats_window: int = 100
    
    # Training Loop
    quick_learn_steps: int = 8192
    intensive_learn_steps: int = 40960
    
    quick_learn_freq: int = 50
    intensive_learn_freq: int = 500
    checkpoint_freq: int = 500
    
    eval_fps: int = 60

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

# Global configuration instance
cfg = Config()
