from trainer import Trainer
import sys

try:
    print("Initializing Trainer...")
    trainer = Trainer()
    print("Starting short training run (5 episodes)...")
    trainer.train(max_episodes=5)
    print("Training run complete.")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
