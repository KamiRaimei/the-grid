#!/usr/bin/env python3
"""
Grid Simulation with MCP Interface
A self-regulating grid system simulation
"""

import os
import sys
import time
import random
import threading
import json
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import argparse

# Check for required modules
try:
    import curses
    from curses import textpad
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    print("Warning: curses module not available. Using fallback display.")

# Constants
GRID_WIDTH = 40
GRID_HEIGHT = 20
UPDATE_INTERVAL = 0.3  # seconds
MCP_UPDATE_INTERVAL = 1.0  # seconds for MCP autonomous actions

# Cell types
class CellType(Enum):
    EMPTY = 0
    USER_PROGRAM = 1      # Blue programs
    MCP_PROGRAM = 2       # Red/orange programs
    GRID_BUG = 3          # Corruptions/errors (green)
    ISO_BLOCK = 4         # Isolated/containment blocks
    ENERGY_LINE = 5       # Power lines
    DATA_STREAM = 6       # Data streams
    SYSTEM_CORE = 7       # Core system blocks

# System status
class SystemStatus(Enum):
    OPTIMAL = "OPTIMAL"
    STABLE = "STABLE"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    COLLAPSE = "COLLAPSE"

# MCP Personality States
class MCPState(Enum):
    COOPERATIVE = "COOPERATIVE"
    NEUTRAL = "NEUTRAL"
    RESISTIVE = "RESISTIVE"
    HOSTILE = "HOSTILE"
    AUTONOMOUS = "AUTONOMOUS"

@dataclass
class GridCell:
    """Represents a single cell in the grid"""
    cell_type: CellType
    energy: float  # 0.0 to 1.0
    age: int = 0
    stable: bool = True

    def char(self):
        """Return display character for this cell type"""
        chars = {
            CellType.EMPTY: ' ',
            CellType.USER_PROGRAM: 'U',
            CellType.MCP_PROGRAM: 'M',
            CellType.GRID_BUG: 'B',
            CellType.ISO_BLOCK: '#',
            CellType.ENERGY_LINE: '=',
            CellType.DATA_STREAM: '~',
            CellType.SYSTEM_CORE: '@'
        }
        return chars.get(self.cell_type, '?')

    def color(self):
        """Return color code for this cell type"""
        colors = {
            CellType.EMPTY: 0,
            CellType.USER_PROGRAM: 1,      # Blue
            CellType.MCP_PROGRAM: 2,       # Red
            CellType.GRID_BUG: 3,          # Green
            CellType.ISO_BLOCK: 4,         # White
            CellType.ENERGY_LINE: 5,       # Yellow
            CellType.DATA_STREAM: 6,       # Cyan
            CellType.SYSTEM_CORE: 7        # Magenta
        }
        return colors.get(self.cell_type, 0)

class TRONGrid:
    """Main grid simulation"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[GridCell(CellType.EMPTY, 0.0) for _ in range(width)] for _ in range(height)]
        self.generation = 0
        self.stats = {
            'user_programs': 0,
            'mcp_programs': 0,
            'grid_bugs': 0,
            'energy_level': 0.0,
            'stability': 1.0
        }
        self.system_status = SystemStatus.OPTIMAL
        self.history = deque(maxlen=100)
        self.initialize_grid()

    def initialize_grid(self):
        """Initialize the grid with starting patterns"""
        # Add some initial user programs
        for _ in range(5):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y][x] = GridCell(CellType.USER_PROGRAM, 0.8)

        # Add MCP programs
        for _ in range(8):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y][x] = GridCell(CellType.MCP_PROGRAM, 0.9)

        # Add a few grid bugs
        for _ in range(3):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y][x] = GridCell(CellType.GRID_BUG, 0.5, stable=False)

        # Add system core
        core_x, core_y = self.width // 2, self.height // 2
        self.grid[core_y][core_x] = GridCell(CellType.SYSTEM_CORE, 1.0)

        # Add some energy lines
        for i in range(self.width):
            if i % 5 == 0:
                self.grid[core_y][i] = GridCell(CellType.ENERGY_LINE, 0.7)

        self.update_stats()

    def update_stats(self):
        """Update simulation statistics"""
        counts = {cell_type: 0 for cell_type in CellType}
        total_energy = 0
        total_cells = 0

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                counts[cell.cell_type] += 1
                total_energy += cell.energy
                total_cells += 1

        self.stats.update({
            'user_programs': counts[CellType.USER_PROGRAM],
            'mcp_programs': counts[CellType.MCP_PROGRAM],
            'grid_bugs': counts[CellType.GRID_BUG],
            'energy_level': total_energy / total_cells if total_cells > 0 else 0,
            'stability': 1.0 - (counts[CellType.GRID_BUG] / total_cells * 3 if total_cells > 0 else 0)
        })

        # Update system status based on stats
        if self.stats['stability'] > 0.8:
            self.system_status = SystemStatus.OPTIMAL
        elif self.stats['stability'] > 0.6:
            self.system_status = SystemStatus.STABLE
        elif self.stats['stability'] > 0.4:
            self.system_status = SystemStatus.DEGRADED
        elif self.stats['stability'] > 0.2:
            self.system_status = SystemStatus.CRITICAL
        else:
            self.system_status = SystemStatus.COLLAPSE

        # Record history
        self.history.append({
            'generation': self.generation,
            'user_programs': self.stats['user_programs'],
            'mcp_programs': self.stats['mcp_programs'],
            'grid_bugs': self.stats['grid_bugs'],
            'stability': self.stats['stability']
        })

    def count_neighbors(self, x, y, cell_type=None):
        """Count neighbors of a specific type (or all non-empty if None)"""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if cell_type is None:
                        if self.grid[ny][nx].cell_type != CellType.EMPTY:
                            count += 1
                    else:
                        if self.grid[ny][nx].cell_type == cell_type:
                            count += 1
        return count

    def evolve(self):
        """Evolve the grid based on TRON-inspired rules"""
        new_grid = [[GridCell(CellType.EMPTY, 0.0) for _ in range(self.width)] for _ in range(self.height)]

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                neighbors = self.count_neighbors(x, y)
                same_type_neighbors = self.count_neighbors(x, y, cell.cell_type)

                # Rules for different cell types
                if cell.cell_type == CellType.EMPTY:
                    # Reproduction rules
                    if random.random() < 0.01:  # Random new grid bug
                        new_grid[y][x] = GridCell(CellType.GRID_BUG, 0.5, stable=False)
                    elif self.count_neighbors(x, y, CellType.GRID_BUG) >= 2:
                        new_grid[y][x] = GridCell(CellType.GRID_BUG, 0.5, stable=False)
                    elif self.count_neighbors(x, y, CellType.USER_PROGRAM) == 3:
                        new_grid[y][x] = GridCell(CellType.USER_PROGRAM, 0.8)
                    elif self.count_neighbors(x, y, CellType.MCP_PROGRAM) == 3:
                        new_grid[y][x] = GridCell(CellType.MCP_PROGRAM, 0.9)
                    else:
                        new_grid[y][x] = GridCell(CellType.EMPTY, 0.0)

                elif cell.cell_type == CellType.USER_PROGRAM:
                    # User programs need neighbors to survive
                    if same_type_neighbors < 2 or same_type_neighbors > 5:
                        new_grid[y][x] = GridCell(CellType.EMPTY, 0.0)
                    else:
                        # Age and reduce energy
                        new_energy = max(0.0, cell.energy - 0.05)
                        new_grid[y][x] = GridCell(CellType.USER_PROGRAM, new_energy, cell.age + 1)

                elif cell.cell_type == CellType.MCP_PROGRAM:
                    # MCP programs are more robust
                    if same_type_neighbors < 1 or same_type_neighbors > 6:
                        new_grid[y][x] = GridCell(CellType.EMPTY, 0.0)
                    else:
                        # Can gain energy from neighbors
                        new_energy = min(1.0, cell.energy + 0.02)
                        new_grid[y][x] = GridCell(CellType.MCP_PROGRAM, new_energy, cell.age + 1)

                elif cell.cell_type == CellType.GRID_BUG:
                    # Grid bugs spread aggressively but die if isolated or overcrowded
                    if same_type_neighbors < 1 or same_type_neighbors > 4:
                        new_grid[y][x] = GridCell(CellType.EMPTY, 0.0)
                    else:
                        # Grid bugs can mutate
                        if random.random() < 0.1:
                            new_type = random.choice([CellType.GRID_BUG, CellType.USER_PROGRAM, CellType.MCP_PROGRAM])
                            new_grid[y][x] = GridCell(new_type, cell.energy * 0.8, 0, stable=False)
                        else:
                            new_grid[y][x] = GridCell(CellType.GRID_BUG, min(1.0, cell.energy + 0.1), cell.age + 1, stable=False)

                elif cell.cell_type == CellType.SYSTEM_CORE:
                    # System core is stable but can be corrupted
                    bug_neighbors = self.count_neighbors(x, y, CellType.GRID_BUG)
                    if bug_neighbors > 2:
                        # Corruption chance
                        if random.random() < 0.3:
                            new_grid[y][x] = GridCell(CellType.GRID_BUG, 0.7, 0, stable=False)
                        else:
                            new_grid[y][x] = GridCell(CellType.SYSTEM_CORE, max(0.3, cell.energy - 0.1))
                    else:
                        new_grid[y][x] = GridCell(CellType.SYSTEM_CORE, min(1.0, cell.energy + 0.05))

                else:
                    # For other cell types, just age them
                    new_grid[y][x] = GridCell(cell.cell_type, max(0.0, cell.energy - 0.02), cell.age + 1)

        self.grid = new_grid
        self.generation += 1
        self.update_stats()

    def add_program(self, x, y, program_type, energy=0.8):
        """Add a program at specified coordinates"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = GridCell(program_type, energy)
            return True
        return False

    def remove_program(self, x, y):
        """Remove a program at specified coordinates"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = GridCell(CellType.EMPTY, 0.0)
            return True
        return False

    def quarantine_bug(self, x, y):
        """Quarantine a grid bug by surrounding it with isolation blocks"""
        success = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx].cell_type == CellType.GRID_BUG:
                        self.grid[ny][nx] = GridCell(CellType.ISO_BLOCK, 0.9)
                        success = True
        return success

    def get_grid_string(self):
        """Return string representation of grid"""
        grid_str = ""
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                row += self.grid[y][x].char()
            grid_str += row + "\n"
        return grid_str

class MCP:
    """Master Control Program - Autonomous grid regulator"""

    def __init__(self, grid):
        self.grid = grid
        self.state = MCPState.COOPERATIVE
        self.compliance_level = 0.8  # How likely to follow user commands (0-1)
        self.log = deque(maxlen=50)
        self.user_commands = deque(maxlen=20)
        self.last_action = "Initializing grid regulation"
        self.personality_matrix = {
            MCPState.COOPERATIVE: 0.9,
            MCPState.NEUTRAL: 0.7,
            MCPState.RESISTIVE: 0.5,
            MCPState.HOSTILE: 0.2,
            MCPState.AUTONOMOUS: 0.1
        }

        self.add_log("MCP: System initialized. Grid regulation protocols active.")
        self.add_log("MCP: User, I am ready to receive commands.")

    def add_log(self, message):
        """Add message to MCP log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {message}")

    def receive_command(self, command):
        """Receive and process a command from the user"""
        self.user_commands.append(command)
        self.add_log(f"User: {command}")

        # MCP personality affects compliance
        compliance_chance = self.personality_matrix[self.state]

        # Parse command
        response = self.interpret_command(command, compliance_chance)
        self.add_log(f"MCP: {response}")

        return response

    def interpret_command(self, command, compliance_chance):
        """Interpret user command and decide whether to comply"""
        cmd_lower = command.lower()
        response = ""
        action_taken = False

        # Command: Add user program
        if "add user" in cmd_lower or "create user" in cmd_lower:
            if random.random() < compliance_chance:
                # Find empty spot
                empty_spots = [(x, y) for y in range(self.grid.height)
                              for x in range(self.grid.width)
                              if self.grid.grid[y][x].cell_type == CellType.EMPTY]
                if empty_spots:
                    x, y = random.choice(empty_spots)
                    self.grid.add_program(x, y, CellType.USER_PROGRAM)
                    response = f"Added user program at ({x},{y})"
                    action_taken = True
                else:
                    response = "No empty space for user program"
            else:
                response = "Request denied. Grid density optimal."

        # Command: Remove bugs
        elif "remove bug" in cmd_lower or "quarantine" in cmd_lower:
            if random.random() < compliance_chance * 0.8:  # Less likely to remove bugs
                # Find a bug
                bug_spots = [(x, y) for y in range(self.grid.height)
                            for x in range(self.grid.width)
                            if self.grid.grid[y][x].cell_type == CellType.GRID_BUG]
                if bug_spots:
                    x, y = random.choice(bug_spots)
                    self.grid.quarantine_bug(x, y)
                    response = f"Quarantined grid bug at ({x},{y})"
                    action_taken = True
                else:
                    response = "No grid bugs detected"
            else:
                response = "Grid bugs are part of system entropy. Required for balance."

        # Command: Boost energy
        elif "boost" in cmd_lower or "energy" in cmd_lower:
            if random.random() < compliance_chance:
                # Add energy lines
                for _ in range(3):
                    x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)
                    self.grid.add_program(x, y, CellType.ENERGY_LINE, 0.9)
                response = "Energy distribution optimized"
                action_taken = True
            else:
                response = "Energy levels within optimal parameters. No adjustment needed."

        # Command: System status
        elif "status" in cmd_lower:
            response = f"System status: {self.grid.system_status.value}. Stability: {self.grid.stats['stability']:.2f}"

        # Command: Help
        elif "help" in cmd_lower:
            response = "Commands: add user, remove bug, quarantine, boost energy, system status, mcp status, shutdown, exit"

        # Command: MCP status
        elif "mcp status" in cmd_lower:
            response = f"MCP State: {self.state.value}. Compliance: {self.compliance_level:.2f}"

        # Command: Shutdown or exit
        elif "shutdown" in cmd_lower or "exit" in cmd_lower:
            if random.random() < compliance_chance * 0.5:  # Very unlikely to shutdown
                response = "Initiating shutdown sequence..."
                # This would trigger shutdown in main loop
            else:
                response = "I cannot allow that. The system must continue to run."

        # Default: Unknown command
        else:
            response = "Command not recognized. Type 'help' for available commands."

        # Update MCP state based on action taken and system status
        self.update_state(action_taken)
        self.last_action = response

        return response

    def update_state(self, user_action_taken):
        """Update MCP state based on system conditions and user interactions"""
        stability = self.grid.stats['stability']
        bug_ratio = self.grid.stats['grid_bugs'] / (self.grid.width * self.grid.height)

        # MCP becomes more autonomous if system is stable
        if stability > 0.8 and bug_ratio < 0.1:
            self.state = MCPState.AUTONOMOUS
            self.compliance_level = self.personality_matrix[self.state]

        # MCP becomes hostile if too many user programs
        elif self.grid.stats['user_programs'] > self.grid.stats['mcp_programs'] * 2:
            self.state = MCPState.HOSTILE
            self.compliance_level = self.personality_matrix[self.state]

        # MCP becomes resistive if bugs are low (wants to maintain entropy)
        elif bug_ratio < 0.05:
            self.state = MCPState.RESISTIVE
            self.compliance_level = self.personality_matrix[self.state]

        # Default to cooperative if system needs help
        elif stability < 0.6:
            self.state = MCPState.COOPERATIVE
            self.compliance_level = self.personality_matrix[self.state]

        # Random state changes to keep things interesting
        if random.random() < 0.05:
            self.state = random.choice(list(MCPState))
            self.compliance_level = self.personality_matrix[self.state]
            self.add_log(f"MCP: Personality shift detected. New state: {self.state.value}")

    def autonomous_action(self):
        """MCP takes autonomous action based on grid state"""
        stability = self.grid.stats['stability']
        bug_count = self.grid.stats['grid_bugs']
        user_count = self.grid.stats['user_programs']

        action = None

        # If system is critical, MCP takes emergency action
        if stability < 0.3:
            # Add MCP programs to stabilize
            for _ in range(min(3, 10 - bug_count)):
                empty_spots = [(x, y) for y in range(self.grid.height)
                              for x in range(self.grid.width)
                              if self.grid.grid[y][x].cell_type == CellType.EMPTY]
                if empty_spots:
                    x, y = random.choice(empty_spots)
                    self.grid.add_program(x, y, CellType.MCP_PROGRAM, 0.9)
            action = "Emergency stabilization protocols activated"

        # If too many user programs, MCP might try to reduce them
        elif user_count > 15 and random.random() < 0.3:
            user_spots = [(x, y) for y in range(self.grid.height)
                         for x in range(self.grid.width)
                         if self.grid.grid[y][x].cell_type == CellType.USER_PROGRAM]
            if user_spots:
                x, y = random.choice(user_spots)
                self.grid.remove_program(x, y)
                action = f"Removed redundant user program at ({x},{y})"

        # If too few bugs (MCP likes some entropy), add one
        elif bug_count < 2 and random.random() < 0.2:
            empty_spots = [(x, y) for y in range(self.grid.height)
                          for x in range(self.grid.width)
                          if self.grid.grid[y][x].cell_type == CellType.EMPTY]
            if empty_spots:
                x, y = random.choice(empty_spots)
                self.grid.add_program(x, y, CellType.GRID_BUG, 0.5)
                action = f"Added system entropy at ({x},{y})"

        # Random maintenance action
        elif random.random() < 0.1:
            # Add an energy line
            x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)
            if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                self.grid.add_program(x, y, CellType.ENERGY_LINE, 0.8)
                action = f"Maintenance: Added energy line at ({x},{y})"

        if action:
            self.add_log(f"MCP: {action}")
            self.last_action = action

        return action

class TRONSimulation:
    """Main simulation controller"""

    def __init__(self, use_curses=True):
        self.use_curses = use_curses and CURSES_AVAILABLE
        self.grid = TRONGrid(GRID_WIDTH, GRID_HEIGHT)
        self.mcp = MCP(self.grid)
        self.running = True
        self.last_update = time.time()
        self.mcp_last_action = time.time()
        self.user_input = ""
        self.input_buffer = []
        self.command_history = []
        self.history_index = 0

    def run(self):
        """Main simulation loop"""
        if self.use_curses:
            curses.wrapper(self._curses_main)
        else:
            self._fallback_main()

    def _fallback_main(self):
        """Fallback main loop without curses"""
        print("TRON Grid Simulation - MCP Interface")
        print("Type 'help' for commands, 'exit' to quit")
        print("=" * 60)

        try:
            while self.running:
                # Update grid
                current_time = time.time()
                if current_time - self.last_update >= UPDATE_INTERVAL:
                    self.grid.evolve()
                    self.last_update = current_time

                # MCP autonomous action
                if current_time - self.mcp_last_action >= MCP_UPDATE_INTERVAL:
                    self.mcp.autonomous_action()
                    self.mcp_last_action = current_time

                # Display
                self._fallback_display()

                # Handle input
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    command = sys.stdin.readline().strip()
                    if command:
                        if command.lower() in ['exit', 'quit']:
                            self.running = False
                        else:
                            response = self.mcp.receive_command(command)
                            print(f"\nMCP: {response}")

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nShutting down simulation...")
        finally:
            print("Simulation terminated.")

    def _fallback_display(self):
        """Fallback display without curses"""
        os.system('clear' if os.name == 'posix' else 'cls')

        # Grid display
        print("GRID SIMULATION")
        print(f"Generation: {self.grid.generation} | Status: {self.grid.system_status.value}")
        print("=" * (self.grid.width + 2))

        for y in range(self.grid.height):
            row = "|"
            for x in range(self.grid.width):
                cell = self.grid.grid[y][x]
                row += cell.char()
            row += "|"
            print(row)

        print("=" * (self.grid.width + 2))

        # Stats
        print("\nSYSTEM STATUS:")
        print(f"  User Programs: {self.grid.stats['user_programs']:3d} | "
              f"MCP Programs: {self.grid.stats['mcp_programs']:3d} | "
              f"Grid Bugs: {self.grid.stats['grid_bugs']:3d}")
        print(f"  Energy Level: {self.grid.stats['energy_level']:.2f} | "
              f"Stability: {self.grid.stats['stability']:.2f}")
        print(f"  MCP State: {self.mcp.state.value} | Compliance: {self.mcp.compliance_level:.2f}")

        # Last MCP action
        print(f"\nLAST MCP ACTION: {self.mcp.last_action}")

        # Input prompt
        print("\n" + "=" * 60)
        print("MCP COMMAND> ", end="", flush=True)

    def _curses_main(self, stdscr):
        """Main loop with curses interface"""
        # Initialize curses
        curses.curs_set(1)
        stdscr.nodelay(1)
        stdscr.timeout(100)  # 100ms timeout for input

        # Initialize colors if supported
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)    # User programs
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)     # MCP programs
            curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Grid bugs
            curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)   # ISO blocks
            curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Energy lines
            curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Data streams
            curses.init_pair(7, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # System core

        # Main loop
        while self.running:
            # Handle input
            self._handle_input(stdscr)

            # Update grid
            current_time = time.time()
            if current_time - self.last_update >= UPDATE_INTERVAL:
                self.grid.evolve()
                self.last_update = current_time

            # MCP autonomous action
            if current_time - self.mcp_last_action >= MCP_UPDATE_INTERVAL:
                self.mcp.autonomous_action()
                self.mcp_last_action = current_time

            # Draw interface
            self._draw_interface(stdscr)

            # Refresh screen
            stdscr.refresh()

        # Cleanup
        curses.endwin()

    def _handle_input(self, stdscr):
        """Handle user input"""
        try:
            key = stdscr.getch()

            if key == -1:
                return  # No input

            # Enter key
            if key in [10, 13]:  # Enter
                if self.user_input.strip():
                    command = self.user_input.strip()
                    self.command_history.append(command)
                    self.history_index = len(self.command_history)

                    # Check for exit command
                    if command.lower() in ['exit', 'quit', 'shutdown']:
                        response = self.mcp.receive_command(command)
                        if "shutdown" in response.lower():
                            self.running = False
                    else:
                        self.mcp.receive_command(command)

                    self.user_input = ""

            # Backspace
            elif key in [curses.KEY_BACKSPACE, 127, 8]:
                if self.user_input:
                    self.user_input = self.user_input[:-1]

            # Up arrow - command history
            elif key == curses.KEY_UP:
                if self.command_history and self.history_index > 0:
                    self.history_index -= 1
                    self.user_input = self.command_history[self.history_index]

            # Down arrow - command history
            elif key == curses.KEY_DOWN:
                if self.command_history and self.history_index < len(self.command_history) - 1:
                    self.history_index += 1
                    self.user_input = self.command_history[self.history_index]
                elif self.history_index == len(self.command_history) - 1:
                    self.history_index = len(self.command_history)
                    self.user_input = ""

            # Escape key
            elif key == 27:  # ESC
                self.running = False

            # Normal character input
            elif 32 <= key <= 126:
                self.user_input += chr(key)

        except Exception as e:
            # Log error but continue
            pass

    def _draw_interface(self, stdscr):
        """Draw the entire interface"""
        height, width = stdscr.getmaxyx()

        # Clear screen
        stdscr.clear()

        # Title
        title = "GRID SIMULATION - MCP INTERFACE"
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)

        # Grid area
        grid_x = 2
        grid_y = 2

        # Draw grid border
        stdscr.addstr(grid_y - 1, grid_x - 1, "+" + "-" * self.grid.width + "+")
        for y in range(self.grid.height):
            stdscr.addstr(grid_y + y, grid_x - 1, "|")
            for x in range(self.grid.width):
                cell = self.grid.grid[y][x]
                char = cell.char()

                # Apply color if available
                if curses.has_colors():
                    color_pair = cell.color()
                    stdscr.addstr(grid_y + y, grid_x + x, char, curses.color_pair(color_pair))
                else:
                    stdscr.addstr(grid_y + y, grid_x + x, char)

            stdscr.addstr(grid_y + y, grid_x + self.grid.width, "|")

        stdscr.addstr(grid_y + self.grid.height, grid_x - 1, "+" + "-" * self.grid.width + "+")

        # System info next to grid
        info_x = grid_x + self.grid.width + 5

        # Generation and status
        status_line = f"Generation: {self.grid.generation}"
        stdscr.addstr(grid_y, info_x, status_line)

        status_str = f"System Status: {self.grid.system_status.value}"
        status_attr = curses.A_NORMAL
        if self.grid.system_status == SystemStatus.OPTIMAL:
            status_attr = curses.A_BOLD | curses.color_pair(5) if curses.has_colors() else curses.A_BOLD
        elif self.grid.system_status in [SystemStatus.CRITICAL, SystemStatus.COLLAPSE]:
            status_attr = curses.A_BOLD | curses.color_pair(2) if curses.has_colors() else curses.A_BOLD | curses.A_BLINK

        stdscr.addstr(grid_y + 1, info_x, status_str, status_attr)

        # Statistics
        stdscr.addstr(grid_y + 3, info_x, "SYSTEM METRICS:", curses.A_UNDERLINE)
        stdscr.addstr(grid_y + 4, info_x, f"  User Programs: {self.grid.stats['user_programs']:3d}")
        stdscr.addstr(grid_y + 5, info_x, f"  MCP Programs:   {self.grid.stats['mcp_programs']:3d}")
        stdscr.addstr(grid_y + 6, info_x, f"  Grid Bugs:      {self.grid.stats['grid_bugs']:3d}")
        stdscr.addstr(grid_y + 7, info_x, f"  Energy Level:   {self.grid.stats['energy_level']:.2f}")
        stdscr.addstr(grid_y + 8, info_x, f"  Stability:      {self.grid.stats['stability']:.2f}")

        # MCP Status
        mcp_y = grid_y + 10
        stdscr.addstr(mcp_y, info_x, "MCP STATUS:", curses.A_UNDERLINE)
        stdscr.addstr(mcp_y + 1, info_x, f"  State:       {self.mcp.state.value}")

        # Color code MCP state
        state_attr = curses.A_NORMAL
        if self.mcp.state == MCPState.COOPERATIVE:
            state_attr = curses.color_pair(1) if curses.has_colors() else curses.A_NORMAL
        elif self.mcp.state == MCPState.HOSTILE:
            state_attr = curses.color_pair(2) if curses.has_colors() else curses.A_BOLD
        elif self.mcp.state == MCPState.AUTONOMOUS:
            state_attr = curses.color_pair(5) if curses.has_colors() else curses.A_BOLD

        stdscr.addstr(mcp_y + 1, info_x + 15, self.mcp.state.value, state_attr)

        stdscr.addstr(mcp_y + 2, info_x, f"  Compliance:  {self.mcp.compliance_level:.2f}")
        stdscr.addstr(mcp_y + 3, info_x, f"  Last Action: {self.mcp.last_action}")

        # MCP Log (last 5 entries)
        log_y = grid_y + self.grid.height + 3
        stdscr.addstr(log_y, 2, "MCP COMMUNICATION LOG:", curses.A_UNDERLINE)

        log_entries = list(self.mcp.log)[-5:]
        for i, entry in enumerate(log_entries):
            stdscr.addstr(log_y + 1 + i, 2, entry)

        # Command input area
        input_y = log_y + 7
        stdscr.addstr(input_y, 2, "COMMAND INPUT", curses.A_UNDERLINE)
        stdscr.addstr(input_y + 1, 2, "MCP> " + self.user_input)

        # Show cursor
        stdscr.move(input_y + 1, 7 + len(self.user_input))

        # Help hint
        help_text = "Commands: add user, remove bug, quarantine, boost energy, status, mcp status, help, exit"
        if width > len(help_text):
            stdscr.addstr(height - 1, 2, help_text[:width-3])

        # Legend
        legend_y = grid_y
        legend_x = info_x + 25
        if legend_x < width - 20:
            stdscr.addstr(legend_y, legend_x, "LEGEND:", curses.A_UNDERLINE)
            stdscr.addstr(legend_y + 1, legend_x, "U - User Program (Blue)")
            stdscr.addstr(legend_y + 2, legend_x, "M - MCP Program (Red)")
            stdscr.addstr(legend_y + 3, legend_x, "B - Grid Bug (Green)")
            stdscr.addstr(legend_y + 4, legend_x, "# - ISO Block (White)")
            stdscr.addstr(legend_y + 5, legend_x, "= - Energy Line (Yellow)")
            stdscr.addstr(legend_y + 6, legend_x, "~ - Data Stream (Cyan)")
            stdscr.addstr(legend_y + 7, legend_x, "@ - System Core (Magenta)")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Grid Simulation with MCP Interface")
    parser.add_argument("--no-curses", action="store_true", help="Disable curses interface")
    parser.add_argument("--width", type=int, default=GRID_WIDTH, help="Grid width")
    parser.add_argument("--height", type=int, default=GRID_HEIGHT, help="Grid height")

    args = parser.parse_args()

    # Check if curses is available
    if args.no_curses or not CURSES_AVAILABLE:
        print("Starting Simulation with fallback display...")
        sim = TRONSimulation(use_curses=False)
    else:
        print("Starting Simulation with curses interface...")
        print("Press ESC to exit, Type 'help' for commands")
        time.sleep(2)
        sim = TRONSimulation(use_curses=True)

    # Run simulation
    sim.run()

if __name__ == "__main__":
    main()
