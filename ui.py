from datetime import datetime
from collections import deque
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.console import Console
from rich.style import Style

class TrainingDashboard:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        self.layout["body"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="log", ratio=2)
        )
        
        self.log_history = deque(maxlen=20)
        self.status_text = "Initializing..."
        self.stats = {
            "Episode": 0,
            "Avg Reward (100)": 0.0,
            "P1 Win Rate (100)": 0.0,
            "P1 Annihilation": 0.0,
            "P1 Attrition": 0.0,
            "P2 Annihilation": 0.0,
            "P2 Attrition": 0.0,
            "Draw Rate": 0.0,
            "Avg Steps": 0.0,
            "Total Steps": 0,
            "Best Reward": -float('inf'),
            "P1 Total Annihilations": 0,
            "P2 Total Annihilations": 0
        }
        
    def update_stats(self, episode: int, avg_reward: float, win_rate: float, 
                     p1_annihil: float, p1_attrit: float, 
                     p2_annihil: float, p2_attrit: float,
                     draw_rate: float, avg_steps: float, total_steps: int,
                     p1_total_annihil: int, p2_total_annihil: int):
        self.stats["Episode"] = episode
        self.stats["Avg Reward (100)"] = avg_reward
        self.stats["P1 Win Rate (100)"] = win_rate
        self.stats["P1 Annihilation"] = p1_annihil
        self.stats["P1 Attrition"] = p1_attrit
        self.stats["P2 Annihilation"] = p2_annihil
        self.stats["P2 Attrition"] = p2_attrit
        self.stats["Draw Rate"] = draw_rate
        self.stats["Avg Steps"] = avg_steps
        self.stats["Total Steps"] = total_steps
        self.stats["P1 Total Annihilations"] = p1_total_annihil
        self.stats["P2 Total Annihilations"] = p2_total_annihil
        
        if avg_reward > self.stats["Best Reward"]:
            self.stats["Best Reward"] = avg_reward

    def log_event(self, message: str, style: str = "white"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_history.append((timestamp, message, style))

    def set_status(self, status: str):
        self.status_text = status

    def _generate_header(self) -> Panel:
        return Panel(
            Text("WarThink AI - Asymmetric Self-Play PPO", justify="center", style="bold cyan"),
            style="cyan"
        )

    def _generate_stats_table(self) -> Panel:
        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right", style="bold")
        
        # Group stats for better readability
        groups = [
            ("Main Stats", ["Episode", "Avg Reward (100)", "Best Reward", "Avg Steps", "Total Steps"]),
            ("Win Rates", ["P1 Win Rate (100)", "Draw Rate"]),
            ("P1 Details", ["P1 Annihilation", "P1 Attrition", "P1 Total Annihilations"]),
            ("P2 Details", ["P2 Annihilation", "P2 Attrition", "P2 Total Annihilations"]),
        ]

        for group_name, keys in groups:
            table.add_row(Text(group_name, style="bold underline"), "")
            for key in keys:
                value = self.stats[key]
                if "Total" in key:
                    # Total counts should be integers
                    val_str = str(value)
                    style = "cyan"
                elif "Rate" in key or "Annihilation" in key or "Attrition" in key:
                    val_str = f"{value:.1%}"
                    if "P1" in key and value > 0.5: style = "green"
                    elif "P2" in key and value > 0.5: style = "red"
                    else: style = "white"
                elif "Reward" in key:
                    val_str = f"{value:.2f}"
                    style = "yellow"
                else:
                    val_str = str(value)
                    style = "white"
                
                table.add_row(f"  {key}", Text(val_str, style=style))
            table.add_row("", "") # Spacer
            
        return Panel(
            table,
            title="Training Stats (Last 100 Eps)",
            border_style="blue"
        )

    def _generate_log_panel(self) -> Panel:
        log_text = Text()
        for ts, msg, style in self.log_history:
            log_text.append(f"[{ts}] ", style="dim")
            log_text.append(f"{msg}\n", style=style)
            
        return Panel(
            log_text,
            title="Event Log",
            border_style="green"
        )

    def _generate_footer(self) -> Panel:
        return Panel(
            Text(f"Status: {self.status_text}", style="italic"),
            style="dim"
        )

    def get_renderable(self) -> Layout:
        self.layout["header"].update(self._generate_header())
        self.layout["stats"].update(self._generate_stats_table())
        self.layout["log"].update(self._generate_log_panel())
        self.layout["footer"].update(self._generate_footer())
        return self.layout
