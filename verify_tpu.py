import torch
from rl import PPOSelfPlayAgent
from env import WarGameEnv

def test_device_detection():
    env = WarGameEnv()
    agent = PPOSelfPlayAgent(env)
    print(f"Detected device: {agent.device}")
    
    expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # On this machine, we expect CPU or CUDA, not TPU.
    # But we want to ensure the code runs without error.
    
    if agent.device == expected_device:
        print("Device detection fallback working correctly.")
    else:
        print(f"Unexpected device: {agent.device} (expected {expected_device})")

if __name__ == "__main__":
    test_device_detection()
