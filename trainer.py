import os
import re
import pygame
import numpy as np
from typing import List, Optional, Tuple
from rich import print as rr
import csv
import datetime

from env import WarGameEnv
from rl import PPOSelfPlayAgent
from renderer import Renderer
from recorder import record_pygame
from ui import TrainingDashboard
from rich.live import Live
from config import cfg

class Trainer:
    def __init__(self, load_model_path: Optional[str] = None):
        self.env = WarGameEnv()
        self.agent = PPOSelfPlayAgent(self.env)
        self.renderer = Renderer()
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_details = [] # List of dicts: {winner, condition, steps}
        self.episode_winners = []
        self.p1_total_annihilations = 0
        self.p2_total_annihilations = 0
        os.makedirs('models', exist_ok=True)
        
        # Initialize CSV logging
        self.csv_log_path = 'training_log.csv'
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'AvgReward', 'P1_WinRate', 'P1_Annihil', 'P1_Attrit', 'P2_Annihil', 'P2_Attrit', 'DrawRate', 'AvgSteps', 'Timestamp'])

        if load_model_path:
            if os.path.exists(load_model_path):
                print(f"Loading model from: {load_model_path}")
                self.agent.load(load_model_path)
                
                # Try to parse episode count from filename
                match = re.search(r'_(\d+)\.zip', load_model_path)
                if match:
                    self.episode_count = int(match.group(1))
                    print(f"Resuming from episode {self.episode_count}")
            else:
                print(f"Warning: Model path not found, starting new training: {load_model_path}")

    def get_recent_stats(self, window: int = cfg.trainer.stats_window):
        if len(self.episode_rewards) < window:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        recent_rewards = np.array(self.episode_rewards[-window:])
        recent_details = self.episode_details[-window:]
        
        avg_reward = np.mean(recent_rewards)
        
        # Calculate detailed stats
        total = len(recent_details)
        p1_wins = sum(1 for d in recent_details if d['winner'] == 1)
        p1_win_rate = p1_wins / total
        
        p1_annihil = sum(1 for d in recent_details if d['winner'] == 1 and d['condition'] == 'annihilation') / total
        p1_attrit = sum(1 for d in recent_details if d['winner'] == 1 and d['condition'] == 'attrition') / total
        
        p2_annihil = sum(1 for d in recent_details if d['winner'] == 2 and d['condition'] == 'annihilation') / total
        p2_attrit = sum(1 for d in recent_details if d['winner'] == 2 and d['condition'] == 'attrition') / total
        
        draw_rate = sum(1 for d in recent_details if d['winner'] == 0) / total
        avg_steps = np.mean([d['steps'] for d in recent_details])
        
        return avg_reward, p1_win_rate, p1_annihil, p1_attrit, p2_annihil, p2_attrit, draw_rate, avg_steps

    def play_episode(self, dashboard: Optional[TrainingDashboard] = None):
        obs, _ = self.env.reset()
        self.agent.start_episode()
        p1_total_reward = 0.0
        p2_total_reward = 0.0
        steps = 0
        done = False
        while not done:
            player = self.env.state.current_player
            actions = self.agent.act(obs, player)
            obs, reward, terminated, truncated, _ = self.env.step(actions)
            done = terminated or truncated
            steps += 1
            if player == 1:
                p1_total_reward += reward
            else:
                p2_total_reward += reward
        # --- End of episode loop ---
        
        self.total_steps += steps

        # Add terminal rewards symmetrically based on win condition
        winner = self.env.state.winner
        win_condition = self.env.state.win_condition

        if winner in [1, 2]:
            # Annihilation wins get a larger bonus
            bonus = cfg.env.annihilation_bonus if win_condition == 'annihilation' else cfg.env.attrition_bonus
            if dashboard and win_condition == 'annihilation':
                 # Only log annihilation to reduce spam, or maybe just rare events
                 pass
            # temporarily disabled annihilation negative rewards
            if winner == 1:
                p1_total_reward += bonus
                # p2_total_reward -= bonus
                if win_condition == 'annihilation':
                    self.p1_total_annihilations += 1
            else: # winner == 2
                p2_total_reward += bonus
                # p1_total_reward -= bonus
                if win_condition == 'annihilation':
                    self.p2_total_annihilations += 1
        elif winner == 0:
            # Apply draw penalty to both players
            p1_total_reward += cfg.env.draw_penalty
            p2_total_reward += cfg.env.draw_penalty
        
        # Log the final total reward (sum of dense + terminal rewards)
        final_total_reward = p1_total_reward + p2_total_reward
        self.episode_rewards.append(final_total_reward)
        self.episode_details.append({
            'winner': winner,
            'condition': win_condition,
            'steps': steps
        })
        
        self.episode_winners.append(winner)
        self.agent.update_pool([p1_total_reward, p2_total_reward])

    def eval_render(self, dashboard: Optional[TrainingDashboard] = None):
        # if dashboard:
        #     # dashboard.log_event("Eval render started...", style="blue")
        # else:
        #     print("Starting evaluation render...")
            
        obs, _ = self.env.reset()
        self.agent.start_episode()
        done = False
        
        # Restore video recording
        with record_pygame(f"videos/WarWatch_{self.episode_count}.mp4", fps=cfg.trainer.eval_fps):
            while not done:
                if not self.renderer.handle_events():
                    if dashboard: dashboard.log_event("Eval interrupted.", style="red")
                    return
                player = self.env.state.current_player
                actions = self.agent.act(obs, player)
                obs, reward, terminated, truncated, _ = self.env.step(actions)
                done = terminated or truncated
                self.renderer.render(self.env.state)
                
                # Just get basic stats for renderer
                stats = self.get_recent_stats(cfg.trainer.stats_window)
                avg_reward, win_rate = stats[0], stats[1]
                
                self.renderer.draw_stats(
                    self.episode_count, avg_reward, win_rate,
                    self.env.state.turn_count, self.env.state.winner
                )
                self.renderer.clock.tick(10)
            
        if dashboard:
            dashboard.log_event("Eval complete.", style="blue")

    def train(self, max_episodes=float('inf')):
        dashboard = TrainingDashboard()
        dashboard.log_event('Starting asymmetric self-play PPO training...', style="bold green")
        dashboard.log_event(f"Device: {self.agent.model.device}", style="cyan")
        
        with Live(dashboard.get_renderable(), refresh_per_second=4) as live:
            try:
                while self.episode_count < max_episodes:
                    dashboard.set_status("Playing episode...")
                    live.update(dashboard.get_renderable())
                    
                    self.play_episode(dashboard=dashboard)
                    self.episode_count += 1
                    
                    # Update stats every episode
                    stats = self.get_recent_stats(cfg.trainer.stats_window)
                    # Unpack all stats
                    (avg_reward, p1_win_rate, p1_annihil, p1_attrit, 
                     p2_annihil, p2_attrit, draw_rate, avg_steps) = stats
                     
                    dashboard.update_stats(
                        self.episode_count, avg_reward, p1_win_rate,
                        p1_annihil, p1_attrit, p2_annihil, p2_attrit,
                        draw_rate, avg_steps, self.total_steps,
                        self.p1_total_annihilations, self.p2_total_annihilations
                    )

                    # CSV Logging every 100 episodes
                    if self.episode_count % 100 == 0:
                        with open(self.csv_log_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                self.episode_count, 
                                f"{avg_reward:.2f}", 
                                f"{p1_win_rate:.2f}",
                                f"{p1_annihil:.2f}",
                                f"{p1_attrit:.2f}",
                                f"{p2_annihil:.2f}",
                                f"{p2_attrit:.2f}",
                                f"{draw_rate:.2f}",
                                f"{avg_steps:.1f}",
                                datetime.datetime.now().isoformat()
                            ])
                    
                    if self.episode_count % cfg.trainer.quick_learn_freq == 0:
                        dashboard.set_status("Quick learning...")
                        # dashboard.log_event(f"Quick learn at ep {self.episode_count}...", style="dim")
                        live.update(dashboard.get_renderable())
                        self.agent.learn(total_timesteps=cfg.trainer.quick_learn_steps)
                        
                    if self.episode_count % cfg.trainer.intensive_learn_freq == 0:
                        dashboard.log_event(f"--- Episode {self.episode_count} ---", style="bold")
                        
                        dashboard.set_status("Evolving pool...")
                        # dashboard.log_event("Evolving policy pool...", style="yellow")
                        live.update(dashboard.get_renderable())
                        self.agent.evolve_pool(logger=lambda msg: dashboard.log_event(msg, style="yellow"))
                        
                        dashboard.set_status("Intensive learning...")
                        dashboard.log_event("Performing intensive PPO learning...", style="magenta")
                        live.update(dashboard.get_renderable())
                        self.agent.learn(total_timesteps=cfg.trainer.intensive_learn_steps)
                        
                        dashboard.set_status("Evaluating...")
                        dashboard.log_event("Rendering evaluation game...", style="blue")
                        live.update(dashboard.get_renderable())
                        self.eval_render(dashboard=dashboard)
                        
                    if self.episode_count % cfg.trainer.checkpoint_freq == 0:
                        checkpoint_path = f"models/checkpoint_{self.episode_count}.zip"
                        self.agent.save(checkpoint_path)
                        dashboard.log_event(f"Checkpoint saved: {checkpoint_path}", style="bold green")
                        
                    live.update(dashboard.get_renderable())
                    
            except KeyboardInterrupt:
                dashboard.log_event("Interrupted by user.", style="bold red")
                if self.episode_count > 0:
                    checkpoint_path = f"models/interrupt_checkpoint_{self.episode_count}.zip"
                    self.agent.save(checkpoint_path)
                    dashboard.log_event(f"Final checkpoint saved: {checkpoint_path}", style="bold green")
                live.update(dashboard.get_renderable())