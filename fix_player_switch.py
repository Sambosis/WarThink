#!/usr/bin/env python3
"""Fix env.py by moving player switch after observation generation."""

with open('env.py', 'r') as f:
    content = f.read()

# Remove the player switch lines that are in the wrong place
content = content.replace(
    """        # Increment turn count\r
        self.state.turn_count += 1\r
        \r
        # Switch current player\r
        self.state.current_player = 2 if self.state.current_player == 1 else 1\r
""",
    """        # Increment turn count\r
        self.state.turn_count += 1\r
""")

# Add the player switch AFTER observation generation (after "obs = generate_obs(self.state)")
content = content.replace(
    """        # Next observation\r
        obs = generate_obs(self.state)\r
""",
    """        # Next observation\r
        obs = generate_obs(self.state)\r
        \r
        # Switch current player AFTER generating observation\r
        self.state.current_player = 2 if self.state.current_player == 1 else 1\r
""")

with open('env.py', 'w') as f:
    f.write(content)

print("Successfully fixed player switching order in env.py")
