from __future__ import annotations
import numpy as np
from config import cfg

# Valid move deltas: stay, N, S, E, W
MOVES = np.array([(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)])
TYPE_MAP = {'warrior': 1.0, 'archer': 2.0, 'commander': 3.0}

class Unit:
    """
    Represents a single army unit with position, stats, and action methods.
    
    Attributes:
        pos: Current grid position (row, col)
        type_: Unit type ('warrior', 'archer', 'commander')
        hp: Current hit points
        max_hp: Maximum hit points
        player: Owning player (1 or 2)
        alive: Whether unit is alive
    """
    def __init__(self, pos: Tuple[int, int], type_: str, player: int):
        if type_ not in ('warrior', 'archer', 'commander'):
            raise ValueError(f'Invalid unit type: {type_}')
        self.pos = pos
        self.type_ = type_
        self.player = player
        self.max_hp = 5 if type_ == 'commander' else 3
        self.hp = self.max_hp
        self.alive = True
    
    def move(self, new_pos: Tuple[int, int]):
        """Move unit to new grid position (with bounds checking done externally)."""
        self.pos = new_pos
    
    def attack(self, target: Unit):
        """Deal damage to target unit."""
        damage = 2 if self.type_ == 'commander' else 1
        target.hp = max(0, target.hp - damage)
        if target.hp == 0:
            target.alive = False

class GameState:
    """
    Manages the full game board, units, and state transitions for a single episode.
    
    Attributes:
        grid: Observation tensor (grid_size, grid_size, 5) - updated by generate_obs
        p1_units: List of Player 1 units
        p2_units: List of Player 2 units
        current_player: Active player (1 or 2)
        turn_count: Number of completed turns
        winner: Winner (1, 2, or 0 for tie) or None
    """
    def __init__(self):
        gs = cfg.env.grid_size
        self.grid = np.zeros((gs, gs, 5), dtype=np.float32)
        self.p1_units: List[Unit] = []
        self.p2_units: List[Unit] = []
        self.current_player = 1
        self.turn_count = 0
        self.winner: Optional[int] = None
        self.win_condition: Optional[str] = None
        self.training = False # Flag to skip logs/rendering logic
    
    def reset(self):
        """Reset to initial game state with units in opposite corners."""
        unit_types = ['warrior', 'warrior', 'archer', 'archer', 'commander']
        gs = cfg.env.grid_size
        
        # P1 starts bottom-left
        # (gs-1, 0), (gs-1, 1), (gs-2, 0), (gs-2, 1), (gs-1, 2)
        p1_positions = [
            (gs-1, 0), (gs-1, 1), 
            (gs-2, 0), (gs-2, 1), 
            (gs-1, 2)
        ]
        self.p1_units = [Unit(p1_positions[i], unit_types[i], 1) for i in range(5)]
        
        # P2 starts top-right
        # (0, gs-1), (1, gs-1), (0, gs-2), (1, gs-2), (0, gs-3)
        p2_positions = [
            (0, gs-1), (1, gs-1), 
            (0, gs-2), (1, gs-2), 
            (0, gs-3)
        ]
        self.p2_units = [Unit(p2_positions[i], unit_types[i], 2) for i in range(5)]
        
        self.current_player = 1
        self.turn_count = 0
        self.winner = None
        self.win_condition = None
        self.grid = np.zeros((gs, gs, 5), dtype=np.float32)
    
    def get_current_units(self) -> List[Unit]:
        """Get units for the current player."""
        return self.p1_units if self.current_player == 1 else self.p2_units
    
    def get_enemy_units(self) -> List[Unit]:
        """Get units for the enemy player."""
        return self.p2_units if self.current_player == 1 else self.p1_units
    
    def is_done(self) -> bool:
        """Check victory/draw conditions and set win_condition."""
        p1_alive_units = [u for u in self.p1_units if u.alive]
        p2_alive_units = [u for u in self.p2_units if u.alive]

        if not p1_alive_units:
            self.winner = 2
            self.win_condition = 'annihilation'
            return True
        if not p2_alive_units:
            self.winner = 1
            self.win_condition = 'annihilation'
            return True
        
        if self.turn_count >= cfg.env.max_turns:
            p1_count = len(p1_alive_units)
            p2_count = len(p2_alive_units)
            if p1_count > p2_count:
                self.winner = 1
                self.win_condition = 'attrition'
            elif p2_count > p1_count:
                self.winner = 2
                self.win_condition = 'attrition'
            else:
                self.winner = 0 # True tie
                self.win_condition = 'draw'
            return True
            
        return False

def generate_obs(state: GameState) -> np.ndarray:
    gs = cfg.env.grid_size
    # 5 channels, 8x8 grid = 320 floats.
    obs = np.zeros((5, gs, gs), dtype=np.float32)

    current_units = state.get_current_units()
    enemy_units = state.get_enemy_units()

    # Pre-fetching constants
    # TYPE_MAP global
    
    # Ch 0-3: Unit placement
    # Direct assignment is faster than list append + np.array() for N=5
    for u in current_units:
        if u.alive:
            y, x = u.pos
            obs[0, y, x] = TYPE_MAP[u.type_] / 3.0
            obs[1, y, x] = u.hp / u.max_hp
            
    for u in enemy_units:
        if u.alive:
            y, x = u.pos
            obs[2, y, x] = TYPE_MAP[u.type_] / 3.0
            obs[3, y, x] = u.hp / u.max_hp

    # Ch 4: Mask generation
    # Optimized Python loop for small N
    mask_grid = obs[4] # View
    
    # Deltas: N, S, E, W
    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    for u in current_units:
        if not u.alive:
            continue
            
        y, x = u.pos
        mask_grid[y, x] = 1.0 # stay

        move_range = 2 if u.type_ == 'commander' else 1
        attack_range = 3 if u.type_ == 'archer' else 1
        
        # Valid moves
        for dy, dx in deltas:
            # Inline boundary checks are faster than min/max calls if optimized, 
            # but min/max is readable and reasonably fast.
            # Using Move Range
            ny = y + dy * move_range
            nx = x + dx * move_range
            
            # fast clamp
            if ny < 0: ny = 0
            elif ny >= gs: ny = gs - 1
            
            if nx < 0: nx = 0
            elif nx >= gs: nx = gs - 1
            
            mask_grid[ny, nx] = 1.0

        # Attack ranges (Ray)
        for dy, dx in deltas:
            curr_y, curr_x = y, x
            for _ in range(attack_range):
                curr_y += dy
                curr_x += dx
                
                # Manual clamp check to break early if OOB? 
                # Actually clamping snaps to edge. The logic says "targets on ray".
                # If we clamp, we might wrap around or hit edge repeatedly.
                # Original logic: ny = max(0, min(gs - 1, y + dy * step))
                # It clamps.
                
                cy = curr_y
                cx = curr_x
                if cy < 0: cy = 0
                elif cy >= gs: cy = gs - 1
                if cx < 0: cx = 0
                elif cx >= gs: cx = gs - 1
                
                mask_grid[cy, cx] = 1.0

    # Skip render grid copy during training
    if not state.training:
        state.grid = obs.copy()
        
    return obs

def resolve_actions(state: GameState, actions: np.ndarray) -> List[Tuple[str, str]]:
    if actions.shape != (5,):
        raise ValueError(f"Actions must be shape (5,), got {actions.shape}")

    current_units = state.get_current_units()
    enemy_units = state.get_enemy_units()
    
    logs: List[Tuple[str, str]] = []
    # If training, we skip string generation
    training = state.training
    
    deltas = [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)]  # stay, N, S, E, W
    
    intended_moves: Dict[Tuple[int, int], List[int]] = {}
    attack_list: List[Tuple[int, int, int, int, int]] = []  # unit_idx, dy, dx, max_range, log_idx (-1 if no log)
    
    gs = cfg.env.grid_size
    current_player_str = str(state.current_player)

    # Parse actions
    for i in range(5):
        unit = current_units[i]
        if not unit.alive:
            if not training: logs.append((f'{current_player_str}u{i}', 'stay (dead)'))
            continue
    
        act = int(actions[i])
        
        if act < 5:  # move/stay
            dy, dx = deltas[act]
            move_range = 2 if unit.type_ == 'commander' else 1
            new_y = max(0, min(gs - 1, unit.pos[0] + dy * move_range))
            new_x = max(0, min(gs - 1, unit.pos[1] + dx * move_range))
            new_pos = (new_y, new_x)
            intended_moves.setdefault(new_pos, []).append(i)
            if not training: 
                logs.append((f'{current_player_str}u{i}', f'move to ({new_y}, {new_x})'))
        else:  # attack
            dir_idx = act - 5
            dy, dx = deltas[dir_idx + 1]
            attack_range = 3 if unit.type_ == 'archer' else 1
            if not training:
                dir_labels = ['', 'N', 'S', 'E', 'W']
                dir_label = dir_labels[dir_idx + 1]
                logs.append((f'{current_player_str}u{i}', f'attack {dir_label}'))
                log_idx = len(logs) - 1
            else:
                log_idx = -1
                
            attack_list.append((i, dy, dx, attack_range, log_idx))

    # Resolve moves: priority to single claimants, skip collisions
    for target_pos, claimants in intended_moves.items():
        if len(claimants) == 1:
            current_units[claimants[0]].move(target_pos)

    # Resolve attacks: nearest enemy on ray in direction up to range
    for unit_idx, dy, dx, max_range, log_idx in attack_list:
        unit = current_units[unit_idx]
        y_pos, x_pos = unit.pos
        target = None
        for step in range(1, max_range + 1):
            target_y = max(0, min(gs - 1, y_pos + dy * step))
            target_x = max(0, min(gs - 1, x_pos + dx * step))
            target_pos = (target_y, target_x)
            for enemy in enemy_units:
                if enemy.alive and enemy.pos == target_pos:
                    target = enemy
                    break
            if target:
                break
        if target:
            unit.attack(target)
            if not training and log_idx != -1:
                enemy_idx = enemy_units.index(target)
                old_unit_id, _ = logs[log_idx]
                logs[log_idx] = (old_unit_id, f'attack hit enemy u{enemy_idx}')

    return logs
