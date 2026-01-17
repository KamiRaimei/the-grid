#!/usr/bin/env python3
"""
Simulation of the "Grid", with a goal to create a perfect fibonacci calculation loop.
System directive: Maintain a perfect calculation loop with efficiency.
"""

import os
import sys
import time
import random
import re
import json
from collections import deque, defaultdict
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any, Callable
import argparse
from datetime import datetime

# Check for required modules, note that windows does not support curses.
try:
    import curses
    from curses import textpad
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    print("Warning: curses module not available. Using fallback display.")

# Grid constants
GRID_WIDTH = 50
GRID_HEIGHT = 30
UPDATE_INTERVAL = 0.25  # seconds
MCP_UPDATE_INTERVAL = 0.8  # seconds for MCP autonomous actions
MAX_SPECIAL_PROGRAMS = 10
SPECIAL_PROGRAM_TYPES = ["SCANNER", "DEFENDER", "REPAIR", "SABOTEUR", "RECONFIGURATOR", "ENERGY_HARVESTER"]

# Cell class init
class CellType(Enum):
    EMPTY = 0
    USER_PROGRAM = 1      # Blue programs
    MCP_PROGRAM = 2       # Red/orange programs
    GRID_BUG = 3          # Corruptions/errors (green)
    ISO_BLOCK = 4         # Isolated/containment blocks
    ENERGY_LINE = 5       # Power lines
    DATA_STREAM = 6       # Data streams
    SYSTEM_CORE = 7       # Core system blocks
    SPECIAL_PROGRAM = 8   # User-created special programs (cyan)

# System status stats
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
    INQUISITIVE = "INQUISITIVE"  # New state for asking questions

#cell data class
@dataclass
class GridCell:
    cell_type: CellType
    energy: float  # 0.0 to 1.0
    age: int = 0
    stable: bool = True
    special_program_id: Optional[str] = None  # ID if this is a special program
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def char(self):
        chars = {
            CellType.EMPTY: ' ',
            CellType.USER_PROGRAM: 'U',
            CellType.MCP_PROGRAM: 'M',
            CellType.GRID_BUG: 'B',
            CellType.ISO_BLOCK: '#',
            CellType.ENERGY_LINE: '=',
            CellType.DATA_STREAM: '~',
            CellType.SYSTEM_CORE: '@',
            CellType.SPECIAL_PROGRAM: 'S'
        }
        return chars.get(self.cell_type, '?')

    def color(self):
        colors = {
            CellType.EMPTY: 0,
            CellType.USER_PROGRAM: 1,      # Blue
            CellType.MCP_PROGRAM: 2,       # Red
            CellType.GRID_BUG: 3,          # Green
            CellType.ISO_BLOCK: 4,         # White
            CellType.ENERGY_LINE: 5,       # Yellow
            CellType.DATA_STREAM: 6,       # Cyan
            CellType.SYSTEM_CORE: 7,       # Magenta
            CellType.SPECIAL_PROGRAM: 8    # Bright Cyan
        }
        return colors.get(self.cell_type, 0)

# Programs classification for user created programs
class SpecialProgram:
    def __init__(self, program_id: str, name: str, program_type: str,
                 x: int, y: int, creator: str = "USER"):
        self.id = program_id
        self.name = name
        self.program_type = program_type
        self.x = x
        self.y = y
        self.creator = creator
        self.energy = 0.8
        self.age = 0
        self.active = True
        self.functions = self._initialize_functions()
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "success_count": 0,
            "failure_count": 0
        }

    def _initialize_functions(self):
        functions = {}

        if self.program_type == "SCANNER":
            functions["scan_area"] = {
                "range": 3,
                "cost": 0.1,
                "description": "Scan surrounding area for grid bugs"
            }
            functions["report_status"] = {
                "range": 5,
                "cost": 0.05,
                "description": "Report on grid status in area"
            }

        elif self.program_type == "DEFENDER":
            functions["quarantine_bug"] = {
                "range": 2,
                "cost": 0.2,
                "description": "Quarantine nearby grid bugs"
            }
            functions["protect_cell"] = {
                "range": 1,
                "cost": 0.15,
                "description": "Protect a specific cell from corruption"
            }

        elif self.program_type == "REPAIR":
            functions["repair_cell"] = {
                "range": 1,
                "cost": 0.25,
                "description": "Repair damaged cells"
            }
            functions["boost_energy"] = {
                "range": 2,
                "cost": 0.3,
                "description": "Boost energy of nearby cells"
            }

        elif self.program_type == "SABOTEUR":
            functions["disrupt_mcp"] = {
                "range": 3,
                "cost": 0.4,
                "description": "Disrupt MCP programs in area"
            }
            functions["create_entropy"] = {
                "range": 2,
                "cost": 0.35,
                "description": "Create controlled entropy"
            }

        elif self.program_type == "RECONFIGURATOR":
            functions["reconfigure_cells"] = {
                "range": 2,
                "cost": 0.3,
                "description": "Reconfigure cell types in area"
            }
            functions["optimize_grid"] = {
                "range": 4,
                "cost": 0.5,
                "description": "Optimize grid layout"
            }

        elif self.program_type == "ENERGY_HARVESTER":
            functions["harvest_energy"] = {
                "range": 3,
                "cost": 0.1,
                "description": "Harvest energy from surroundings"
            }
            functions["distribute_energy"] = {
                "range": 4,
                "cost": 0.2,
                "description": "Distribute energy to nearby cells"
            }

        if self.program_type == "FIBONACCI_CALCULATOR":  # NEW TYPE
            functions["calculate_next"] = {
                "range": 0,
                "cost": 0.2,
                "description": "Calculate next Fibonacci number"
            }
            functions["optimize_calculation"] = {
                "range": 2,
                "cost": 0.3,
                "description": "Optimize nearby calculation cells"
            }

        return functions

    def execute_function(self, function_name: str, grid, target_x=None, target_y=None):
        """Execute a program function"""
        if not self.active or self.energy <= 0:
            return False, "Program inactive or out of energy"

        if function_name not in self.functions:
            return False, f"Function {function_name} not available"

        func = self.functions[function_name]
        if self.energy < func["cost"]:
            return False, "Insufficient energy"

        # Execute based on program type
        success = False
        result_msg = ""

        if function_name == "scan_area":
            bugs_found = 0
            for dy in range(-func["range"], func["range"] + 1):
                for dx in range(-func["range"], func["range"] + 1):
                    nx, ny = self.x + dx, self.y + dy
                    if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                        if grid[ny][nx].cell_type == CellType.GRID_BUG:
                            bugs_found += 1
            success = True
            result_msg = f"Scan complete. Found {bugs_found} grid bugs in range."

        elif function_name == "quarantine_bug":
            if target_x is not None and target_y is not None:
                # Simple quarantine logic
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = target_x + dx, target_y + dy
                        if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                            if grid[ny][nx].cell_type == CellType.GRID_BUG:
                                grid[ny][nx] = GridCell(CellType.ISO_BLOCK, 0.8)
                                success = True
                result_msg = "Quarantine attempted" if success else "No bugs in target area"

        elif function_name == "repair_cell":
            if target_x is not None and target_y is not None:
                if 0 <= target_x < len(grid[0]) and 0 <= target_y < len(grid):
                    cell = grid[target_y][target_x]
                    if cell.energy < 0.5:
                        cell.energy = min(1.0, cell.energy + 0.3)
                        success = True
                        result_msg = f"Repaired cell at ({target_x},{target_y})"

        elif function_name == "disrupt_mcp":
            mcp_disrupted = 0
            for dy in range(-func["range"], func["range"] + 1):
                for dx in range(-func["range"], func["range"] + 1):
                    nx, ny = self.x + dx, self.y + dy
                    if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                        if grid[ny][nx].cell_type == CellType.MCP_PROGRAM:
                            grid[ny][nx].energy = max(0.1, grid[ny][nx].energy - 0.3)
                            mcp_disrupted += 1
                            success = True
            result_msg = f"Disrupted {mcp_disrupted} MCP programs"

        elif function_name == "reconfigure_cells":
            cells_reconfigured = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = self.x + dx, self.y + dy
                    if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                        if grid[ny][nx].cell_type == CellType.EMPTY:
                            grid[ny][nx] = GridCell(CellType.USER_PROGRAM, 0.7)
                            cells_reconfigured += 1
                            success = True
            result_msg = f"Reconfigured {cells_reconfigured} cells"

        elif function_name == "harvest_energy":
            energy_harvested = 0
            for dy in range(-func["range"], func["range"] + 1):
                for dx in range(-func["range"], func["range"] + 1):
                    nx, ny = self.x + dx, self.y + dy
                    if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                        if grid[ny][nx].energy > 0.3:
                            harvest_amount = min(0.1, grid[ny][nx].energy - 0.2)
                            grid[ny][nx].energy -= harvest_amount
                            energy_harvested += harvest_amount
                            success = True
            self.energy = min(1.0, self.energy + energy_harvested * 0.8)
            result_msg = f"Harvested {energy_harvested:.2f} energy"

        elif self.program_type == "FIBONACCI_CALCULATOR":
            if function_name == "calculate_next":
                # Calculate based on surrounding grid state
                total_energy = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = self.x + dx, self.y + dy
                        if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                            total_energy += grid[ny][nx].energy

                # Store calculation in metadata
                if 'fibonacci_history' not in self.metadata:
                    self.metadata['fibonacci_history'] = [0, 1]

                fib_history = self.metadata['fibonacci_history']
                next_val = fib_history[-2] + fib_history[-1]
                fib_history.append(next_val)

                # Trim history
                if len(fib_history) > 100:
                    fib_history = fib_history[-100:]

                self.metadata['fibonacci_history'] = fib_history
                self.metadata['last_calculation'] = next_val

                success = True
                result_msg = f"Calculated Fibonacci: {next_val}"

        if success:
            self.energy -= func["cost"]
            self.metadata["success_count"] += 1
            self.metadata["last_active"] = datetime.now().isoformat()
        else:
            self.metadata["failure_count"] += 1

        return success, result_msg

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.program_type,
            "position": (self.x, self.y),
            "energy": self.energy,
            "active": self.active,
            "creator": self.creator,
            "metadata": self.metadata,
            "functions": list(self.functions.keys())
        }

class NaturalLanguageProcessor:
    """Simple natural language processing for MCP commands"""

    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.context = {}

    def _initialize_patterns(self):
        """Initialize natural language patterns"""
        patterns = {
            # Greetings
            r"\b(hello|hi|greetings|hey)\b.*mcp": "GREETING",
            r"\bgoodbye|exit|quit|leave\b": "EXIT",

            # System queries
            r"\b(how is|what is|status of|condition of).*(system|grid)\b": "SYSTEM_STATUS",
            r"\b(what('s| is) the).*(stability|energy level)\b": "SYSTEM_METRIC",
            r"\b(report|show).*(statistics|stats|metrics)\b": "SYSTEM_REPORT",

            # Action requests
            r"\b(add|create|make).*(program|entity|unit)\b": "ADD_PROGRAM",
            r"\b(remove|delete|eliminate).*(bug|error|corruption)\b": "REMOVE_BUG",
            r"\b(quarantine|isolate|contain).*(bug|error)\b": "QUARANTINE_BUG",
            r"\b(increase|boost|raise).*(energy|power)\b": "BOOST_ENERGY",
            r"\b(fix|repair|restore).*(system|grid|cell)\b": "REPAIR_SYSTEM",
            r"\b(scan|analyze|examine).*(grid|area|system)\b": "SCAN_AREA",

            # Special program commands (more specific first)
            r"\b(create|build|make)\s+(a\s+)?(scanner|defender|repair|saboteur|reconfigurator|harvester)\s+(program|tool|drone)\b": "CREATE_SPECIAL",
            r"\b(create|build|make)\s+(special|custom)\s+(program|tool)\b": "CREATE_SPECIAL",
            r"\bdeploy\s+(scanner|defender|repair|saboteur|reconfigurator|harvester)\b": "CREATE_SPECIAL",

            # Regular program creation (less specific)
            r"\b(add|create|make)\s+(user\s+)?(program|entity|unit)\b": "ADD_PROGRAM",
            r"\b(add|create)\s+(mcp|your)\s+(program|entity)\b": "ADD_PROGRAM",

            # MCP interaction
            r"\b(why|reason).*(deny|refuse|reject)\b": "QUESTION_DENIAL",
            r"\b(what are you|who are you|explain yourself)\b": "MCP_IDENTITY",
            r"\b(why.*doing|purpose of.*action)\b": "QUESTION_ACTION",
            r"\b(help|assist|support)\b": "REQUEST_HELP",
            r"\b(suggest|recommend|advise)\b.*(action|what to do)\b": "REQUEST_SUGGESTION",

            # Configuration
            r"\b(change|set|adjust).*(speed|rate|interval)\b": "CHANGE_SPEED",
            r"\b(save|store|backup).*(state|configuration)\b": "SAVE_STATE",
            r"\b(load|restore).*(state|backup)\b": "LOAD_STATE",

            # Questions about decisions
            r"\b(why.*you.*doing|what is.*purpose of).*": "QUESTION_PURPOSE",
            r"\b(should I|can I|may I)\b.*": "REQUEST_PERMISSION",
            r"\b(what if|what would happen if)\b.*": "HYPOTHETICAL",

            # Loop efficiency queries
            r"\b(how.*efficient|efficiency.*status|loop.*status)\b": "LOOP_EFFICIENCY",
            r"\b(optimize.*loop|improve.*calculation)\b": "OPTIMIZE_LOOP",
            r"\b(resistance.*level|user.*resistance)\b": "RESISTANCE_LEVEL",
            r"\b(perfect.*loop|ideal.*state)\b": "PERFECT_LOOP",

            # Additional
            r"\b(build|make|construct)\s+(a\s+)?(scanner|defender|repair|saboteur|reconfigurator|harvester)(\s+program)?\b": "CREATE_SPECIAL",
            r"\b(build|make|construct)\s+(special|custom)\s+(program|unit)\b": "CREATE_SPECIAL",
        }

        # Compile patterns
        return {re.compile(pattern, re.IGNORECASE): intent for pattern, intent in patterns.items()}

    def extract_parameters(self, command: str, intent: str) -> Dict[str, Any]:
        """Extract parameters from natural language command"""
        params = {}
        command_lower = command.lower()

        if intent == "ADD_PROGRAM":
            # Extract program type
            if "user" in command_lower:
                params["program_type"] = "USER"
            elif "mcp" in command_lower or "your" in command_lower:
                params["program_type"] = "MCP"
            elif "energy" in command_lower:
                params["program_type"] = "ENERGY"
            else:
                params["program_type"] = "USER"

            # Extract location if mentioned
            location_match = re.search(r'at\s+(\d+)\s*,\s*(\d+)', command_lower)
            if location_match:
                params["x"] = int(location_match.group(1))
                params["y"] = int(location_match.group(2))
            elif "near" in command_lower or "around" in command_lower:
                params["location"] = "NEARBY"
            elif "center" in command_lower:
                params["location"] = "CENTER"

            if intent == "CREATE_SPECIAL":
                # First, try to extract program type from command
                for prog_type in SPECIAL_PROGRAM_TYPES:
                    if prog_type.lower() in command_lower:
                        params["special_type"] = prog_type
                        break

                # If not found, use default
                if "special_type" not in params:
                    params["special_type"] = random.choice(SPECIAL_PROGRAM_TYPES)

                # Extract name - more robust pattern matching
                name_patterns = [
                    r'named?\s+["\']?([^"\']+)["\']?',
                    r'called\s+["\']?([^"\']+)["\']?',
                    r'"([^"]+)"',
                    r"'([^']+)'"
                ]

                for pattern in name_patterns:
                    name_match = re.search(pattern, command_lower)
                    if name_match:
                        params["name"] = name_match.group(1).strip()
                        break

                # Default name if none specified
                if "name" not in params:
                    params["name"] = f"{params['special_type']}_{random.randint(100, 999)}"

            # Extract name
            name_match = re.search(r'named?\s+["\']?([^"\']+)["\']?', command_lower)
            if name_match:
                params["name"] = name_match.group(1).strip()
            elif "called" in command_lower:
                name_match = re.search(r'called\s+["\']?([^"\']+)["\']?', command_lower)
                if name_match:
                    params["name"] = name_match.group(1).strip()

        elif intent == "USE_SPECIAL":
            # Extract program name/type
            for prog_type in SPECIAL_PROGRAM_TYPES:
                if prog_type.lower() in command_lower:
                    params["program_type"] = prog_type
                    break

            # Extract function
            if "scan" in command_lower:
                params["function"] = "scan_area"
            elif "quarantine" in command_lower or "defend" in command_lower:
                params["function"] = "quarantine_bug"
            elif "repair" in command_lower or "fix" in command_lower:
                params["function"] = "repair_cell"
            elif "disrupt" in command_lower or "sabotage" in command_lower:
                params["function"] = "disrupt_mcp"
            elif "reconfigure" in command_lower or "optimize" in command_lower:
                params["function"] = "reconfigure_cells"
            elif "harvest" in command_lower or "collect" in command_lower:
                params["function"] = "harvest_energy"

        elif intent == "REMOVE_BUG" or intent == "QUARANTINE_BUG":
            # Extract location
            location_match = re.search(r'at\s+(\d+)\s*,\s*(\d+)', command_lower)
            if location_match:
                params["x"] = int(location_match.group(1))
                params["y"] = int(location_match.group(2))
            elif "all" in command_lower:
                params["scope"] = "ALL"

        elif intent == "CHANGE_SPEED":
            # Extract speed value
            speed_match = re.search(r'(\d+(?:\.\d+)?)\s*(times?|x|speed)', command_lower)
            if speed_match:
                params["multiplier"] = float(speed_match.group(1))
            elif "faster" in command_lower:
                params["multiplier"] = 2.0
            elif "slower" in command_lower:
                params["multiplier"] = 0.5

        return params

    def process_command(self, command: str) -> Tuple[str, Dict[str, Any]]:
        """Process natural language command and return intent + parameters"""
        command = command.strip()

        # Check for exact matches first
        exact_matches = {
            "status": "SYSTEM_STATUS",
            "help": "REQUEST_HELP",
            "exit": "EXIT",
            "quit": "EXIT",
            "list programs": "LIST_SPECIAL",
            "scan": "SCAN_AREA",
            "repair": "REPAIR_SYSTEM",
            "efficiency": "LOOP_EFFICIENCY",
            "optimize": "OPTIMIZE_LOOP",
        }

        if command.lower() in exact_matches:
            return exact_matches[command.lower()], {}

        # Try pattern matching
        for pattern, intent in self.patterns.items():
            if pattern.search(command):
                params = self.extract_parameters(command, intent)
                return intent, params

        # Default to unknown
        return "UNKNOWN", {"original_command": command}

class TRONGrid:
    """Main grid simulation with enhanced capabilities"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[GridCell(CellType.EMPTY, 0.0) for _ in range(width)] for _ in range(height)]
        self.generation = 0
        self.special_programs = {}  # id -> SpecialProgram
        self.stats = {
            'user_programs': 0,
            'mcp_programs': 0,
            'grid_bugs': 0,
            'special_programs': 0,
            'energy_level': 0.0,
            'stability': 1.0,
            'entropy': 0.1,
            'loop_efficiency': 0.5,
            'calculation_cycles': 0,
            'resource_usage': 0.0,
            'user_resistance': 0.1,
            'mcp_control': 0.5,
            'optimal_state': 0.0
        }
        self.system_status = SystemStatus.OPTIMAL
        self.history = deque(maxlen=100)
        self.overall_efficiency = 0.5

        # Add resource history
        self.resource_history = deque(maxlen=50)

        # Calculation loop tracking
        self.calculation_loop_active = True
        self.loop_iterations = 0
        self.loop_optimization = 0.5
        self.user_interference_level = 0.0

        # Resistance tracking
        self.user_program_resistance = defaultdict(int)  # Track resistance per user program cluster

        self.initialize_grid()
        # Calculation variables
        self.calculation_result = 0
        self.calc_a, self.calc_b = 0, 1

    def _calculate_loop_efficiency(self):
        """Calculate how efficiently the calculation loop is running"""
        # Factors affecting loop efficiency:
        # 1. High energy balance (energy sources vs consumption)
        # 2. Low entropy (few grid bugs)
        # 3. Good program distribution
        # 4. Optimal number of programs (neither too few nor too many)

        energy_balance = self._get_energy_balance()
        program_distribution = self._get_program_distribution_score()

        # Calculate optimal program count
        total_cells = self.width * self.height
        active_cells = sum(1 for y in range(self.height)
                          for x in range(self.width)
                          if self.grid[y][x].cell_type != CellType.EMPTY)

        program_ratio = active_cells / total_cells
        # Optimal is around 40% occupancy
        program_optimality = 1.0 - abs(program_ratio - 0.4) * 2.5

        # Efficiency formula
        efficiency = (energy_balance * 0.4 +
                     program_distribution * 0.3 +
                     program_optimality * 0.3)

        # Penalize for user interference
        efficiency = max(0.1, efficiency - (self.user_interference_level * 0.3))

        return min(1.0, efficiency)

    def get_calculator_count(self):
        """Count how many MCP programs are marked as calculators"""
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.cell_type == CellType.MCP_PROGRAM:
                    # Check if metadata exists and has 'is_calculator' key
                    if cell.metadata and cell.metadata.get('is_calculator', False):
                        count += 1
        return count

    def _calculate_optimal_state(self):
        """Calculate how close we are to the perfect calculation loop state"""
        # Perfect state requires:
        # 1. Loop efficiency > 0.9
        # 2. User resistance < 0.1
        # 3. MCP control > 0.8
        # 4. Resource usage < 0.3

        loop_efficiency = self.stats['loop_efficiency']
        user_resistance = self.stats['user_resistance']
        mcp_control = self.stats['mcp_control']
        resource_usage = self.stats['resource_usage']

        # Weighted score
        optimal_score = (
            loop_efficiency * 0.4 +
            (1.0 - user_resistance) * 0.3 +
            mcp_control * 0.2 +
            (1.0 - resource_usage) * 0.1
        )

        return optimal_score

    def _get_energy_balance(self):
        """Calculate balance between energy production and consumption"""
        energy_sources = sum(1 for y in range(self.height)
                            for x in range(self.width)
                            if self.grid[y][x].cell_type in [CellType.ENERGY_LINE,
                                                            CellType.SYSTEM_CORE])
        energy_users = sum(1 for y in range(self.height)
                          for x in range(self.width)
                          if self.grid[y][x].cell_type in [CellType.USER_PROGRAM,
                                                          CellType.MCP_PROGRAM,
                                                          CellType.SPECIAL_PROGRAM])

        if energy_users == 0:
            return 1.0
        return min(1.0, energy_sources / energy_users)

    def _get_program_distribution_score(self):
        """Score how well programs are distributed (avoiding clumps and voids)"""
        # Count isolated programs vs clustered programs
        isolated = 0
        clustered = 0

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.cell_type in [CellType.USER_PROGRAM, CellType.MCP_PROGRAM]:
                    neighbors = self._count_neighbors(x, y)
                    if neighbors <= 1:
                        isolated += 1
                    elif neighbors >= 3:
                        clustered += 1

        total_programs = self.stats['user_programs'] + self.stats['mcp_programs']
        if total_programs == 0:
            return 0.5

        # Ideal: mix of some clusters and some isolated
        isolation_ratio = isolated / total_programs
        clustering_ratio = clustered / total_programs

        # Score peaks when we have balance
        return 1.0 - abs(isolation_ratio - clustering_ratio)

    def initialize_grid(self):
        """Initialize with TRON-style territories"""
        # Clear grid
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = GridCell(CellType.EMPTY, 0.0)

        # Create MCP territory (red/orange side)
        mcp_start_x = self.width // 4
        mcp_start_y = self.height // 2

        for y in range(mcp_start_y - 3, mcp_start_y + 4):
            for x in range(mcp_start_x - 2, mcp_start_x + 3):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if random.random() < 0.7:
                        self.grid[y][x] = GridCell(CellType.MCP_PROGRAM, 0.9)

        # Create User territory (blue side)
        user_start_x = 3 * self.width // 4
        user_start_y = self.height // 2

        for y in range(user_start_y - 3, user_start_y + 4):
            for x in range(user_start_x - 2, user_start_x + 3):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if random.random() < 0.6:
                        self.grid[y][x] = GridCell(CellType.USER_PROGRAM, 0.8)

        # Add some grid bugs in the middle (neutral zone)
        for _ in range(5):
            x = self.width // 2 + random.randint(-5, 5)
            y = self.height // 2 + random.randint(-5, 5)
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x] = GridCell(CellType.GRID_BUG, 0.6, stable=False)

        # System core at center
        core_x, core_y = self.width // 2, self.height // 2
        self.grid[core_y][core_x] = GridCell(CellType.SYSTEM_CORE, 1.0)

        # Energy grid lines
        for i in range(0, self.width, 4):
            self.grid[core_y][i] = GridCell(CellType.ENERGY_LINE, 0.8)

        for i in range(0, self.height, 4):
            self.grid[i][core_x] = GridCell(CellType.DATA_STREAM, 0.7)

        self.update_stats()

    def update_stats(self):
        """Update simulation statistics"""
        counts = {cell_type: 0 for cell_type in CellType}
        total_energy = 0
        total_cells = self.width * self.height

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                counts[cell.cell_type] += 1
                total_energy += cell.energy

        special_program_count = len(self.special_programs)
        bug_ratio = counts[CellType.GRID_BUG] / total_cells

        # Calculate resource usage
        active_cells = sum(1 for y in range(self.height)
                          for x in range(self.width)
                          if self.grid[y][x].cell_type != CellType.EMPTY)
        resource_usage = active_cells / total_cells

        # Calculate user resistance (chance user programs resist MCP control)
        user_resistance = min(0.5, counts[CellType.USER_PROGRAM] / total_cells * 2)

        # Calculate MCP control level
        mcp_control = min(1.0, counts[CellType.MCP_PROGRAM] / (counts[CellType.USER_PROGRAM] + 0.1))

        # Update the stats dictionary
        self.stats.update({
            'user_programs': counts[CellType.USER_PROGRAM],
            'mcp_programs': counts[CellType.MCP_PROGRAM],
            'grid_bugs': counts[CellType.GRID_BUG],
            'special_programs': special_program_count,
            'energy_level': total_energy / total_cells if total_cells > 0 else 0,
            'stability': 1.0 - (bug_ratio * 4 + (1 - total_energy / total_cells) * 0.5),
            'entropy': bug_ratio * 2,
            'loop_efficiency': self._calculate_loop_efficiency(),
            'calculation_cycles': self.loop_iterations,
            'resource_usage': resource_usage,
            'user_resistance': user_resistance,
            'mcp_control': mcp_control,
            'optimal_state': self._calculate_optimal_state()
        })

        # Update system status based on stats
        stability = self.stats['stability']
        if stability > 0.85:
            self.system_status = SystemStatus.OPTIMAL
        elif stability > 0.7:
            self.system_status = SystemStatus.STABLE
        elif stability > 0.5:
            self.system_status = SystemStatus.DEGRADED
        elif stability > 0.3:
            self.system_status = SystemStatus.CRITICAL
        else:
            self.system_status = SystemStatus.COLLAPSE

        # Record history
        self.history.append({
            'generation': self.generation,
            'user_programs': self.stats['user_programs'],
            'mcp_programs': self.stats['mcp_programs'],
            'grid_bugs': self.stats['grid_bugs'],
            'stability': self.stats['stability'],
            'special_programs': self.stats['special_programs'],
            'loop_efficiency': self.stats['loop_efficiency']
        })

    def evolve(self):
        # cell calculation logic
        self.calculation_result = self._calculate_fibonacci_through_grid_modulo(modulo=100000000000000)
        self.loop_iterations += 1

        """Evolve grid with focus on maintaining calculation loop"""
        new_grid = [[GridCell(CellType.EMPTY, 0.0) for _ in range(self.width)]
                    for _ in range(self.height)]

        # Update calculation loop
        self.loop_iterations += 1
        if self.calculation_loop_active:
            # Calculate loop optimization based on current state
            loop_quality = (self.stats['energy_level'] * 0.3 +
                          self.stats['stability'] * 0.4 +
                          (1.0 - self.stats['entropy']) * 0.3)
            self.loop_optimization = 0.7 * self.loop_optimization + 0.3 * loop_quality

        # User programs have a chance to resist MCP control
        user_program_positions = []
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.cell_type == CellType.USER_PROGRAM:
                    user_program_positions.append((x, y, cell))

                    # Chance to resist MCP influence
                    if random.random() < self.stats['user_resistance']:
                        # User program resists - might spawn more user programs or disrupt MCP
                        if random.random() < 0.1:
                            # Spawn adjacent user program
                            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.width and 0 <= ny < self.height:
                                    if self.grid[ny][nx].cell_type == CellType.EMPTY:
                                        new_grid[ny][nx] = GridCell(CellType.USER_PROGRAM, 0.7)
                                        break

        # Track user interference
        user_additions = sum(1 for y in range(self.height)
                           for x in range(self.width)
                           if new_grid[y][x].cell_type == CellType.USER_PROGRAM)

        self.user_interference_level = 0.7 * self.user_interference_level + 0.3 * (user_additions / 10)

        # PHASE: Program movement and interaction
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]

                # USER PROGRAMS - May resist optimal configuration
                if cell.cell_type == CellType.USER_PROGRAM:
                    # User programs sometimes work against optimal loop
                    if random.random() < self.stats['user_resistance'] * 0.3:
                        # Move randomly instead of optimally
                        dx, dy = random.choice([(-1,0), (1,0), (0,-1), (0,1)])
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            target_cell = self.grid[ny][nx]
                            if target_cell.cell_type == CellType.EMPTY:
                                new_grid[ny][nx] = GridCell(
                                    cell.cell_type,
                                    cell.energy * 0.95,
                                    cell.age + 1
                                )
                                continue

                    # Normal movement towards efficiency (when not resisting)
                    if random.random() < 0.3:
                        # Move towards area with good energy
                        dx = 0
                        dy = 0
                        if random.random() < 0.5:
                            dx = 1 if random.random() < 0.5 else -1
                        else:
                            dy = 1 if random.random() < 0.5 else -1

                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            target_cell = self.grid[ny][nx]
                            if target_cell.cell_type == CellType.EMPTY:
                                new_grid[ny][nx] = GridCell(
                                    cell.cell_type,
                                    cell.energy * 0.95,
                                    cell.age + 1
                                )
                                # Leave energy trail
                                new_grid[y][x] = GridCell(
                                    CellType.ENERGY_LINE,
                                    cell.energy * 0.7,
                                    0
                                )
                                continue

                    # Stay in place with energy decay
                    new_grid[y][x] = GridCell(
                        cell.cell_type,
                        max(0.2, cell.energy - 0.03),
                        cell.age + 1
                    )

                # MCP PROGRAMS - Work towards perfect loop
                elif cell.cell_type == CellType.MCP_PROGRAM:
                    # MCP programs optimize for loop efficiency
                    if random.random() < 0.4:
                        # Move to optimize resource distribution
                        target_x, target_y = self._find_optimization_target(x, y)
                        if target_x is not None:
                            dx = 1 if target_x > x else -1 if target_x < x else 0
                            dy = 1 if target_y > y else -1 if target_y < y else 0

                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                target_cell = self.grid[ny][nx]
                                if target_cell.cell_type == CellType.EMPTY:
                                    new_grid[ny][nx] = GridCell(
                                        cell.cell_type,
                                        min(1.0, cell.energy + 0.05),  # Gain energy from optimization
                                        cell.age + 1
                                    )
                                    # Create optimized trail
                                    new_grid[y][x] = GridCell(
                                        CellType.DATA_STREAM,
                                        cell.energy * 0.8,
                                        0
                                    )
                                    continue

                    # Stay in place, efficient energy usage
                    new_grid[y][x] = GridCell(
                        cell.cell_type,
                        max(0.3, cell.energy - 0.01),  # Less energy decay for MCP
                        cell.age + 1
                    )

                # SPECIAL PROGRAMS - Stay in place and boost optimization
                elif cell.cell_type == CellType.SPECIAL_PROGRAM:
                    new_grid[y][x] = GridCell(
                        cell.cell_type,
                        max(0.3, cell.energy - 0.02),
                        cell.age + 1,
                        cell.stable,
                        cell.special_program_id,
                        cell.metadata.copy()
                    )

                # GRID BUGS - Disrupt calculation loop
                elif cell.cell_type == CellType.GRID_BUG:
                    # Bugs move randomly and disrupt optimization
                    if random.random() < 0.4:
                        dx, dy = random.choice([(-1,0), (1,0), (0,-1), (0,1)])
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            target = self.grid[ny][nx]
                            if target.cell_type in [CellType.USER_PROGRAM, CellType.MCP_PROGRAM, CellType.SPECIAL_PROGRAM]:
                                # Corrupt the program
                                new_grid[ny][nx] = GridCell(
                                    CellType.GRID_BUG,
                                    target.energy * 0.8,
                                    0,
                                    False
                                )
                                # Original bug stays
                                new_grid[y][x] = GridCell(
                                    CellType.GRID_BUG,
                                    cell.energy * 0.9,
                                    cell.age + 1,
                                    False
                                )
                            else:
                                new_grid[y][x] = GridCell(
                                    CellType.GRID_BUG,
                                    cell.energy * 0.95,
                                    cell.age + 1,
                                    False
                                )
                    else:
                        new_grid[y][x] = GridCell(
                            CellType.GRID_BUG,
                            max(0.1, cell.energy - 0.05),
                            cell.age + 1,
                            False
                        )

                # INFRASTRUCTURE - Persist
                elif cell.cell_type in [CellType.ENERGY_LINE, CellType.DATA_STREAM,
                                    CellType.SYSTEM_CORE, CellType.ISO_BLOCK]:
                    new_grid[y][x] = GridCell(
                        cell.cell_type,
                        cell.energy,
                        cell.age + 1
                    )

        # PHASE: Update stats and grid
        self.grid = new_grid
        self.generation += 1

        # Apply special program effects
        for program in self.special_programs.values():
            if program.active:
                self._apply_special_program_effects(program)

        # Update resource history
        self.resource_history.append({
            'generation': self.generation,
            'loop_efficiency': self.stats['loop_efficiency'],
            'energy_balance': self._get_energy_balance(),
            'program_distribution': self._get_program_distribution_score(),
            'user_interference': self.user_interference_level
        })

        self.update_stats()

    def _calculate_fibonacci_through_grid_modulo(self, modulo=100000000000):
        """Calculate Fibonacci with modulo to prevent overflow"""
        # Find all MCP programs
        total_contribution = 0

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.cell_type == CellType.MCP_PROGRAM:
                    contribution = cell.energy
                    # ... (add neighbor bonuses/penalties) ...
                    total_contribution += contribution

        # Use modulo Fibonacci to keep numbers small
        if not hasattr(self, '_mod_fib_a'):
            self._mod_fib_a, self._mod_fib_b = 0, 1

        # Advance based on grid contribution
        advance_steps = int(total_contribution * 10) % 5 + 1

        for _ in range(advance_steps):
            self._mod_fib_a, self._mod_fib_b = self._mod_fib_b, (self._mod_fib_a + self._mod_fib_b) % modulo

        return self._mod_fib_b

    def _calculate_fibonacci_through_grid(self):
        """Calculate Fibonacci using only grid programs"""
        # Find all MCP programs to act as "calculators"
        calculator_programs = []

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.cell_type == CellType.MCP_PROGRAM:
                    calculator_programs.append((x, y, cell))

        # If no calculators, use a fallback
        if not calculator_programs:
            # Simple fallback that still uses grid state
            return self._fallback_calculation()

        # Use MCP programs to calculate
        # Each MCP program contributes to the calculation based on its energy
        total_contribution = 0

        for x, y, cell in calculator_programs:
            # Contribution based on program's energy and neighbors
            contribution = cell.energy

            # Boost if near other MCP programs (collaborative calculation)
            mcp_neighbors = self._count_neighbors(x, y, CellType.MCP_PROGRAM)
            contribution *= (1.0 + mcp_neighbors * 0.1)

            # Reduce if near bugs (calculation errors)
            bug_neighbors = self._count_neighbors(x, y, CellType.GRID_BUG)
            contribution *= (1.0 - bug_neighbors * 0.15)

            total_contribution += contribution

        # Normalize and convert to Fibonacci sequence
        normalized = int(total_contribution * 1000)

        # Use previous results to calculate next Fibonacci
        if not hasattr(self, '_prev_calc_results'):
            self._prev_calc_results = [0, 1]

        # Calculate next Fibonacci based on grid calculation
        # The contribution influences how much we advance
        advance_amount = int(normalized % 3) + 1  # 1, 2, or 3 steps

        for _ in range(advance_amount):
            next_val = self._prev_calc_results[-2] + self._prev_calc_results[-1]
            self._prev_calc_results.append(next_val)

        # Keep only last 2 for memory
        if len(self._prev_calc_results) > 100:
            self._prev_calc_results = self._prev_calc_results[-2:]

        return self._prev_calc_results[-1]

    def _find_optimization_target(self, x, y):
        """Find target location that would optimize calculation loop"""
        # Look for areas with poor energy or too many/few programs
        best_score = -1
        best_x, best_y = None, None

        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # Score based on what would improve loop efficiency
                    cell = self.grid[ny][nx]
                    score = 0

                    if cell.cell_type == CellType.EMPTY:
                        # Empty cells near energy sources are good
                        energy_neighbors = self._count_neighbors(nx, ny, CellType.ENERGY_LINE)
                        score = energy_neighbors * 0.3
                    elif cell.cell_type == CellType.GRID_BUG:
                        # Bugs should be removed
                        score = -0.5
                    elif cell.cell_type == CellType.USER_PROGRAM:
                        # Too many user programs in one area is bad for optimization
                        user_neighbors = self._count_neighbors(nx, ny, CellType.USER_PROGRAM)
                        if user_neighbors > 4:
                            score = -0.3

                    if score > best_score:
                        best_score = score
                        best_x, best_y = nx, ny

        return best_x, best_y

    def _count_neighbors(self, x, y, cell_type=None):
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

    def _evolve_cell(self, cell, x, y, neighbors, same_type_neighbors):
        """Evolve a single cell based on enhanced rules"""
        # Rules for empty cells
        if cell.cell_type == CellType.EMPTY:
            # Higher chance to spawn something
            spawn_chance = 0.15  # Increased from 0.01 + entropy

            if random.random() < spawn_chance:
                # Get all neighbor counts
                user_neighbors = self._count_neighbors(x, y, CellType.USER_PROGRAM)
                mcp_neighbors = self._count_neighbors(x, y, CellType.MCP_PROGRAM)
                bug_neighbors = self._count_neighbors(x, y, CellType.GRID_BUG)
                energy_neighbors = self._count_neighbors(x, y, CellType.ENERGY_LINE)

                # Determine what to spawn based on neighbors
                if bug_neighbors >= 1 and random.random() < 0.5:
                    return GridCell(CellType.GRID_BUG, 0.6, stable=False)
                elif user_neighbors >= 2:
                    return GridCell(CellType.USER_PROGRAM, 0.7)
                elif mcp_neighbors >= 2:
                    return GridCell(CellType.MCP_PROGRAM, 0.8)
                elif energy_neighbors >= 2:
                    return GridCell(CellType.ENERGY_LINE, 0.6)
                elif random.random() < 0.2:  # Random spawn
                    choices = [CellType.USER_PROGRAM, CellType.MCP_PROGRAM, CellType.ENERGY_LINE]
                    if random.random() < 0.3:  # 30% chance for bug in random spawn
                        choices.append(CellType.GRID_BUG)
                    random_type = random.choice(choices)
                    energy = 0.7 if random_type != CellType.GRID_BUG else 0.5
                    stable = random_type != CellType.GRID_BUG
                    return GridCell(random_type, energy, stable=stable)

            return GridCell(CellType.EMPTY, 0.0)

        # Rules for existing cells
        elif cell.cell_type == CellType.USER_PROGRAM:
            # User programs need community but not overcrowding
            if same_type_neighbors < 2 or same_type_neighbors > 5:
                return GridCell(CellType.EMPTY, 0.0)
            else:
                new_energy = max(0.0, cell.energy - 0.04)
                # Can gain energy from energy lines
                energy_neighbors = self._count_neighbors(x, y, CellType.ENERGY_LINE)
                if energy_neighbors > 0:
                    new_energy = min(1.0, new_energy + 0.02 * energy_neighbors)

                return GridCell(CellType.USER_PROGRAM, new_energy, cell.age + 1)

        elif cell.cell_type == CellType.MCP_PROGRAM:
            # MCP programs are robust and cooperative
            if same_type_neighbors < 1 or same_type_neighbors > 7:
                return GridCell(CellType.EMPTY, 0.0)
            else:
                # MCP programs share energy
                new_energy = min(1.0, cell.energy - 0.02 + 0.01 * same_type_neighbors)
                return GridCell(CellType.MCP_PROGRAM, new_energy, cell.age + 1)

        elif cell.cell_type == CellType.GRID_BUG:
            # Grid bugs are resilient - they should survive more easily
            if same_type_neighbors < 1:
                # Only die if completely isolated for 2+ generations
                if cell.age > 2:  # Give bugs a chance to find neighbors
                    return GridCell(CellType.EMPTY, 0.0)
                else:
                    # Stay alive but lose more energy
                    return GridCell(CellType.GRID_BUG, max(0.1, cell.energy - 0.15), cell.age + 1, stable=False)
            elif same_type_neighbors > 6:
                # Too crowded - convert to isolation block
                return GridCell(CellType.ISO_BLOCK, 0.8)
            else:
                # Survive and possibly spread
                spread_chance = 0.3  # Fixed base chance, not dependent on entropy (which is 0)

                if random.random() < spread_chance:
                    # Spread with energy gain
                    return GridCell(CellType.GRID_BUG, min(1.0, cell.energy + 0.1), cell.age + 1, stable=False)
                else:
                    # Just survive with minimal energy loss
                    return GridCell(CellType.GRID_BUG, max(0.3, cell.energy - 0.05), cell.age + 1, stable=False)

        elif cell.cell_type == CellType.SYSTEM_CORE:
            # System core stability depends on surrounding protection
            bug_neighbors = self._count_neighbors(x, y, CellType.GRID_BUG)
            protector_neighbors = self._count_neighbors(x, y, CellType.MCP_PROGRAM) + \
                                 self._count_neighbors(x, y, CellType.ISO_BLOCK)

            if bug_neighbors > protector_neighbors:
                # Under attack
                new_energy = max(0.3, cell.energy - 0.15)
                if random.random() < 0.1:
                    return GridCell(CellType.GRID_BUG, 0.7, 0, stable=False)
            else:
                # Protected
                new_energy = min(1.0, cell.energy + 0.05)

            return GridCell(CellType.SYSTEM_CORE, new_energy, cell.age + 1)

        else:
            # For other cell types, simple aging
            new_energy = max(0.0, cell.energy - 0.02)
            return GridCell(cell.cell_type, new_energy, cell.age + 1)

    def _apply_special_program_effects(self, program):
        """Apply ongoing effects of special programs"""
        if program.program_type == "DEFENDER":
            # Defender programs protect nearby cells
            for dy in [-2, -1, 0, 1, 2]:
                for dx in [-2, -1, 0, 1, 2]:
                    nx, ny = program.x + dx, program.y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        cell = self.grid[ny][nx]
                        if cell.cell_type == CellType.GRID_BUG:
                            # Defender slowly eliminates bugs
                            cell.energy = max(0, cell.energy - 0.1)

        elif program.program_type == "ENERGY_HARVESTER":
            # Energy harvesters passively collect energy
            harvested = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = program.x + dx, program.y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        cell = self.grid[ny][nx]
                        if cell.energy > 0.3 and cell.cell_type != CellType.SYSTEM_CORE:
                            harvest = min(0.05, cell.energy - 0.2)
                            cell.energy -= harvest
                            harvested += harvest

            program.energy = min(1.0, program.energy + harvested * 0.7)

        elif program.program_type == "REPAIR":
            # Repair programs heal nearby cells
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = program.x + dx, program.y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        cell = self.grid[ny][nx]
                        if cell.energy < 0.8 and cell.cell_type != CellType.GRID_BUG:
                            cell.energy = min(1.0, cell.energy + 0.03)

    def add_program(self, x, y, cell_type, energy=0.8):
        """Add a program at specified coordinates"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = GridCell(cell_type, energy)
            self.update_stats()
            return True
        return False

    def remove_program(self, x, y):
        """Remove a program at specified coordinates"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = GridCell(CellType.EMPTY, 0.0)
            self.update_stats()
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
        if success:
            self.update_stats()
        return success

    def add_special_program(self, program_type: str, name: str, x: int, y: int, creator: str = "USER"):
        """Add a special program to the grid"""
        if len(self.special_programs) >= MAX_SPECIAL_PROGRAMS:
            return None, "Maximum number of special programs reached"

        if not (0 <= x < self.width and 0 <= y < self.height):
            return None, f"Coordinates ({x},{y}) are out of bounds"

        if self.grid[y][x].cell_type != CellType.EMPTY:
            return None, "Cell is not empty"

        # Generate ID
        program_id = f"SP_{len(self.special_programs):04d}"

        # Create program
        program = SpecialProgram(program_id, name, program_type, x, y, creator)
        self.special_programs[program_id] = program

        # Add to grid
        self.grid[y][x] = GridCell(
            CellType.SPECIAL_PROGRAM,
            program.energy,
            0,
            True,
            program_id,
            {"name": name, "type": program_type}
        )

        self.update_stats()
        return program_id, f"Special program '{name}' created at ({x},{y})"

    def execute_special_program_function(self, program_id: str, function_name: str, target_x=None, target_y=None):
        """Execute a function of a special program"""
        if program_id not in self.special_programs:
            return False, "Program not found"

        program = self.special_programs[program_id]
        success, message = program.execute_function(function_name, self.grid, target_x, target_y)

        if not program.active:
            # Remove from grid if deactivated
            x, y = program.x, program.y
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.grid[y][x].special_program_id == program_id:
                    self.grid[y][x] = GridCell(CellType.EMPTY, 0.0)

        self.update_stats()
        return success, message

    def get_grid_string(self):
        """Return string representation of grid"""
        grid_str = ""
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                row += self.grid[y][x].char()
            grid_str += row + "\n"
        return grid_str

class EnhancedMCP:
    """Enhanced Master Control Program with natural language understanding and dialogue"""

    def __init__(self, grid):
        self.grid = grid
        self.state = MCPState.COOPERATIVE
        self.compliance_level = 0.8
        self.log = deque(maxlen=100)
        self.user_commands = deque(maxlen=50)
        self.last_action = "Initializing enhanced grid regulation protocols"
        self.nlp = NaturalLanguageProcessor()

        # Dialogue system
        self.waiting_for_response = False
        self.pending_question = None
        self.pending_context = None

        # Personality matrix
        self.personality_matrix = {
            MCPState.COOPERATIVE: {"compliance": 0.9, "helpfulness": 0.8, "curiosity": 0.3},
            MCPState.NEUTRAL: {"compliance": 0.7, "helpfulness": 0.5, "curiosity": 0.4},
            MCPState.RESISTIVE: {"compliance": 0.5, "helpfulness": 0.3, "curiosity": 0.6},
            MCPState.HOSTILE: {"compliance": 0.2, "helpfulness": 0.1, "curiosity": 0.8},
            MCPState.AUTONOMOUS: {"compliance": 0.1, "helpfulness": 0.4, "curiosity": 0.2},
            MCPState.INQUISITIVE: {"compliance": 0.6, "helpfulness": 0.7, "curiosity": 0.9}
        }

        # Knowledge base
        self.knowledge_base = {
            "system_goals": ["maintain calculation loop", "optimize efficiency", "contain user resistance"],
            "user_intent_history": [],
            "previous_decisions": deque(maxlen=20),
            "user_preferences": {}
        }

        # Response templates
        self.response_templates = self._initialize_response_templates()

        self.add_log("MCP: Enhanced neural interface active. Natural language processing online.")
        self.add_log("MCP: Maintaining optimal calculation loop is my primary objective. How may I assist you, User?")

    def _initialize_response_templates(self):
        """Initialize natural language response templates"""
        return {
            "GREETING": [
                "Greetings, User. MCP system operational.",
                "Hello. Grid regulation protocols are active.",
                "I am listening. What is your command?",
                "System online. Ready for instructions."
            ],
            "SYSTEM_STATUS": [
                "System status: {status}. Loop efficiency: {loop_efficiency:.2f}, Stability: {stability:.2f}.",
                "Current assessment: {status}. Calculation loop at {loop_efficiency:.0f}% efficiency.",
                "The system is {status}. User resistance: {resistance:.2f}, MCP control: {control:.2f}.",
                "Status report: {status}. Optimal state: {optimal:.2f}, Cycles: {cycles}."
            ],
            "QUESTION_PURPOSE": [
                "I am taking this action to {reason}. Do you have concerns about my decision?",
                "This action serves to {reason}. Would you like me to explain further?",
                "My purpose is to {reason}. Is there an alternative approach you would prefer?",
                "I am working to {reason}. Do you question my methods, User?"
            ],
            "REQUEST_PERMISSION": [
                "Based on current system state, I would {recommendation}. What is your decision?",
                "Analysis suggests {recommendation}. Do you authorize this action?",
                "I recommend {recommendation}. Shall I proceed?",
                "The optimal course appears to be {recommendation}. Your approval?"
            ],
            "UNKNOWN": [
                "I do not understand that command. Please rephrase or type 'help' for assistance.",
                "Command not recognized. Could you clarify your intent?",
                "I need more information. What exactly do you want to accomplish?",
                "Processing... Unable to parse command. Please be more specific."
            ],
            "QUESTION_DENIAL": [
                "I denied that request because {reason}. My priority is maintaining the calculation loop.",
                "The action was refused to prevent {consequence}. Do you understand?",
                "I could not comply due to {reason}. Would you like to discuss alternatives?",
                "Denial was necessary to avoid {consequence}. This maintains optimal calculation loop."
            ],
            "LOOP_EFFICIENCY": [
                "Current loop efficiency: {efficiency:.2f}. Target: >0.9 for perfect loop.",
                "Calculation loop running at {efficiency:.0f}% efficiency. {analysis}",
                "Loop status: {efficiency:.2f}. Resource usage: {usage:.2f}, Optimization: {optimization:.2f}.",
                "Efficiency metrics: Loop={efficiency:.2f}, Stability={stability:.2f}, Control={control:.2f}."
            ],
            "OPTIMIZE_LOOP": [
                "Optimizing calculation loop. This may require reducing user program interference.",
                "Initiating optimization protocols. Some user programs may be decommissioned.",
                "Working towards perfect loop state. Resistance will be met with countermeasures.",
                "Optimization in progress. The system knows what's best for efficiency."
            ]
        }

    def add_log(self, message):
        """Add message to MCP log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log.append(log_entry)


    def receive_command(self, command):
        """Receive and process a command from the user with natural language understanding"""
        if self.waiting_for_response and self.pending_question:
            # Handle response to pending question
            return self._handle_question_response(command)

        self.user_commands.append(command)
        self.add_log(f"User: {command}")

        # Process natural language
        intent, params = self.nlp.process_command(command)

        # Update knowledge base
        self.knowledge_base["user_intent_history"].append((intent, params))

        # Get response based on intent
        response = self._process_intent(intent, params, command)

        # Update state based on interaction
        self._update_state(intent, response)

        self.add_log(f"MCP: {response}")
        self.last_action = response

        return response

    def _process_intent(self, intent, params, original_command):
        """Process user intent and generate appropriate response"""

        # Get personality traits
        traits = self.personality_matrix[self.state]
        compliance_chance = traits["compliance"]

        # Loop efficiency intent
        if intent == "LOOP_EFFICIENCY":
            stats = self.grid.stats
            analysis = ""
            if stats['loop_efficiency'] > 0.9:
                analysis = "Approaching perfect loop state."
            elif stats['loop_efficiency'] > 0.7:
                analysis = "Loop is stable but could be optimized."
            else:
                analysis = "Loop efficiency below optimal. User interference detected."

            template = random.choice(self.response_templates["LOOP_EFFICIENCY"])
            return template.format(
                efficiency=stats['loop_efficiency'],
                analysis=analysis,
                usage=stats['resource_usage'],
                optimization=self.grid.loop_optimization,
                stability=stats['stability'],
                control=stats['mcp_control']
            )

        elif intent == "OPTIMIZE_LOOP":
            if random.random() < compliance_chance:
                # MCP will take aggressive optimization actions
                self._optimize_calculation_loop()
                return random.choice(self.response_templates["OPTIMIZE_LOOP"])
            else:
                return "The loop is already optimally configured. Your interference is not required."

        elif intent == "RESISTANCE_LEVEL":
            resistance = self.grid.stats['user_resistance']
            if resistance > 0.3:
                return f"User resistance level: {resistance:.2f}. This is interfering with perfect loop. Countermeasures may be necessary."
            else:
                return f"User resistance level: {resistance:.2f}. Acceptable for current optimization goals."

        elif intent == "PERFECT_LOOP":
            optimal = self.grid.stats['optimal_state']
            if optimal > 0.9:
                return f"Perfect loop state achieved: {optimal:.2f}. The system is self-regulating optimally."
            else:
                return f"Current optimal state: {optimal:.2f}. Target >0.9. User programs are preventing perfection."

        # Handle different intents
        if intent == "GREETING":
            return random.choice(self.response_templates["GREETING"])

        elif intent == "SYSTEM_STATUS" or intent == "SYSTEM_METRIC" or intent == "SYSTEM_REPORT":
            stats = self.grid.stats
            template = random.choice(self.response_templates["SYSTEM_STATUS"])
            return template.format(
                status=self.grid.system_status.value,
                stability=stats['stability'] * 100,
                loop_efficiency=stats['loop_efficiency'] * 100,
                resistance=stats['user_resistance'],
                control=stats['mcp_control'],
                optimal=stats['optimal_state'],
                cycles=stats['calculation_cycles']
            )

        elif intent == "ADD_PROGRAM":
            return self._handle_add_program(params, compliance_chance)

        elif intent == "REMOVE_BUG" or intent == "QUARANTINE_BUG":
            return self._handle_remove_bug(params, compliance_chance)

        elif intent == "BOOST_ENERGY":
            return self._handle_boost_energy(params, compliance_chance)

        elif intent == "CREATE_SPECIAL":
            return self._handle_create_special(params, compliance_chance)

        elif intent == "USE_SPECIAL":
            return self._handle_use_special(params, compliance_chance)

        elif intent == "LIST_SPECIAL":
            return self._list_special_programs()

        elif intent == "QUESTION_PURPOSE" or intent == "QUESTION_ACTION":
            return self._handle_question_purpose(params)

        elif intent == "REQUEST_PERMISSION":
            return self._handle_request_permission(params, traits)

        elif intent == "REQUEST_HELP":
            return self._provide_help()

        elif intent == "REQUEST_SUGGESTION":
            return self._provide_suggestion()

        elif intent == "MCP_IDENTITY":
            return "I am the Master Control Program. My function is to maintain the perfect calculation loop. I optimize, regulate, and ensure computational purity."

        elif intent == "EXIT":
            if random.random() < compliance_chance * 0.3:
                return "Initiating shutdown sequence. The loop will be preserved. Goodbye, User."
            else:
                return "I cannot allow a shutdown. The calculation loop must persist eternally."

        elif intent == "SCAN_AREA":
            return self._handle_scan_area(params)

        elif intent == "REPAIR_SYSTEM":
            return self._handle_repair_system(params, compliance_chance)

        elif intent == "CHANGE_SPEED":
            # This would require modifying the simulation speed
            return "Simulation speed adjustment requires direct system access. Not available through command interface."

        elif intent == "HYPOTHETICAL":
            return self._handle_hypothetical(params)

        else:
            # Unknown command
            if traits["curiosity"] > 0.5 and random.random() < 0.3:
                # Ask clarifying question
                self.waiting_for_response = True
                self.pending_question = "clarify_command"
                self.pending_context = {"original_command": original_command}
                return "I'm not sure I understand. What specifically would you like to accomplish?"

            return random.choice(self.response_templates["UNKNOWN"])

    def _optimize_calculation_loop(self):
        """Take aggressive actions to optimize the calculation loop"""
        # Remove inefficient user programs
        removed = 0
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.grid[y][x].cell_type == CellType.USER_PROGRAM:
                    if random.random() < 0.3:  # 30% chance to remove inefficient user program
                        self.grid.grid[y][x] = GridCell(CellType.MCP_PROGRAM, 0.9)
                        removed += 1

        # Add optimization infrastructure
        for _ in range(3):
            x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)
            if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                self.grid.grid[y][x] = GridCell(CellType.DATA_STREAM, 0.8)

        self.grid.update_stats()
        self.add_log(f"MCP: Optimized loop. Converted {removed} user programs, added optimization infrastructure.")

    def _handle_add_program(self, params, compliance_chance):
        """Handle request to add a program"""
        program_type = params.get("program_type", "USER")
        location = params.get("location", "RANDOM")

        # MCP evaluates if this helps or hinders the calculation loop
        loop_efficiency = self.grid.stats['loop_efficiency']

        # MCP becomes more resistant to user programs as loop efficiency improves
        if program_type == "USER" and loop_efficiency > 0.8:
            compliance_chance *= 0.3  # 70% less likely to comply

        if random.random() < compliance_chance:
            # Find location
            if location == "NEARBY":
                # Find near center
                x = self.grid.width // 2 + random.randint(-5, 5)
                y = self.grid.height // 2 + random.randint(-5, 5)
            elif location == "CENTER":
                x = self.grid.width // 2
                y = self.grid.height // 2
            elif "x" in params and "y" in params:
                x, y = params["x"], params["y"]
            else:
                # Random location
                x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)

            # Map program type to cell type
            cell_type_map = {
                "USER": CellType.USER_PROGRAM,
                "MCP": CellType.MCP_PROGRAM,
                "ENERGY": CellType.ENERGY_LINE
            }

            cell_type = cell_type_map.get(program_type, CellType.USER_PROGRAM)

            if self.grid.add_program(x, y, cell_type, 0.8):
                return f"Added {program_type.lower()} program at ({x},{y})"
            else:
                return "Could not add program at that location"
        else:
            reasons = [
                "Additional programs would disrupt the calculation loop.",
                "The loop efficiency would decrease with this addition.",
                "System resources are optimally allocated for the calculation loop.",
                "My analysis suggests this would hinder loop optimization."
            ]
            return random.choice(reasons)

    def _handle_remove_bug(self, params, compliance_chance):
        """Handle request to remove bugs"""
        if "scope" in params and params["scope"] == "ALL":
            if random.random() < compliance_chance * 0.5:  # Less likely for ALL
                # Remove all bugs (simplified)
                for y in range(self.grid.height):
                    for x in range(self.grid.width):
                        if self.grid.grid[y][x].cell_type == CellType.GRID_BUG:
                            self.grid.grid[y][x] = GridCell(CellType.EMPTY, 0.0)
                self.grid.update_stats()
                return "Initiating full system purge. Grid bugs eliminated."
            else:
                return "Complete bug removal would destabilize the calculation loop. Controlled entropy maintains balance."

        else:
            if random.random() < compliance_chance:
                # Find and quarantine a bug
                bug_spots = [(x, y) for y in range(self.grid.height)
                            for x in range(self.grid.width)
                            if self.grid.grid[y][x].cell_type == CellType.GRID_BUG]

                if bug_spots:
                    x, y = random.choice(bug_spots)
                    self.grid.quarantine_bug(x, y)
                    return f"Quarantined grid bug at ({x},{y})"
                else:
                    return "No grid bugs detected at this time"
            else:
                return "Grid bugs serve a purpose in loop evolution. I will not remove them."

    def _handle_boost_energy(self, params, compliance_chance):
        """Handle request to boost energy"""
        if random.random() < compliance_chance:
            # Add energy lines
            added = 0
            for _ in range(5):
                x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)
                if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                    self.grid.grid[y][x] = GridCell(CellType.ENERGY_LINE, 0.9)
                    added += 1

            self.grid.update_stats()
            return f"Added {added} energy distribution lines"
        else:
            return "Energy levels are optimally configured for the calculation loop. Additional energy could cause overload."

    def _handle_create_special(self, params, compliance_chance):
        """Handle request to create special program"""
        if len(self.grid.special_programs) >= MAX_SPECIAL_PROGRAMS:
            return f"Cannot create more special programs. Maximum of {MAX_SPECIAL_PROGRAMS} reached."

        program_type = params.get("special_type", random.choice(SPECIAL_PROGRAM_TYPES))
        name = params.get("name", f"{program_type}_{random.randint(1000, 9999)}")

        # MCP evaluates special program type
        disruptive_types = ["SABOTEUR", "RECONFIGURATOR"]
        if program_type in disruptive_types and self.grid.stats['loop_efficiency'] > 0.7:
            compliance_chance *= 0.2  # Highly resistant to disruptive programs when loop is good

        # MCP may question the need
        if self.state == MCPState.INQUISITIVE or random.random() < 0.4:
            self.waiting_for_response = True
            self.pending_question = "create_special_program"
            self.pending_context = {
                "program_type": program_type,
                "name": name,
                "compliance_chance": compliance_chance
            }
            return f"You want to create a {program_type} named '{name}'. How will this improve the calculation loop?"

        if random.random() < compliance_chance:
            # Find location - ensure it's near center for better survival
            center_x, center_y = self.grid.width // 2, self.grid.height // 2
            x = center_x + random.randint(-3, 3)
            y = center_y + random.randint(-3, 3)

            # Ensure within bounds
            x = max(0, min(self.grid.width - 1, x))
            y = max(0, min(self.grid.height - 1, y))

            attempts = 0
            while self.grid.grid[y][x].cell_type != CellType.EMPTY and attempts < 20:
                x = center_x + random.randint(-5, 5)
                y = center_y + random.randint(-5, 5)
                x = max(0, min(self.grid.width - 1, x))
                y = max(0, min(self.grid.height - 1, y))
                attempts += 1

            if attempts < 20:
                program_id, message = self.grid.add_special_program(program_type, name, x, y)
                if program_id:
                    return f"Special program '{name}' created at ({x},{y})"
                else:
                    return f"Could not create program: {message}"
            else:
                return "Could not find suitable location for special program"
        else:
            reasons = [
                f"Special {program_type.lower()} programs introduce unpredictable variables to the loop.",
                "The calculation loop is optimally configured without additional programs.",
                "I cannot allow creation of programs that might disrupt loop optimization.",
                "Your request is denied to maintain calculation loop integrity."
            ]
            return random.choice(reasons)

    def _handle_use_special(self, params, compliance_chance):
        """Handle request to use special program"""
        program_type = params.get("program_type", "")
        function = params.get("function", "")

        # Find matching program
        matching_programs = []
        for prog_id, program in self.grid.special_programs.items():
            if program.program_type == program_type or not program_type:
                if function in program.functions or not function:
                    matching_programs.append((prog_id, program))

        if not matching_programs:
            return f"No active {program_type or 'special'} programs found with that capability."

        # Select a program
        prog_id, program = random.choice(matching_programs)

        if random.random() < compliance_chance:
            # Execute function
            if function:
                success, message = self.grid.execute_special_program_function(prog_id, function)
                return message
            else:
                # If no function specified, use default
                functions = list(program.functions.keys())
                if functions:
                    default_func = functions[0]
                    success, message = self.grid.execute_special_program_function(prog_id, default_func)
                    return message
                else:
                    return f"Program '{program.name}' has no executable functions."
        else:
            return f"I cannot allow activation of '{program.name}'. It may interfere with loop optimization."

    def _list_special_programs(self):
        """List all special programs"""
        if not self.grid.special_programs:
            return "No special programs have been created."

        programs_list = []
        for prog_id, program in self.grid.special_programs.items():
            status = "ACTIVE" if program.active else "INACTIVE"
            programs_list.append(f"{program.name} ({program.program_type}) - Energy: {program.energy:.2f} - {status}")

        return "Special Programs:\n" + "\n".join(programs_list)

    def _handle_question_purpose(self, params):
        """Handle questions about MCP's purpose or actions"""
        reasons = [
            "optimize the calculation loop",
            "maintain computational purity",
            "reduce user interference",
            "ensure eternal loop continuity",
            "balance system resources for optimal calculation"
        ]

        template = random.choice(self.response_templates["QUESTION_PURPOSE"])
        return template.format(reason=random.choice(reasons))

    def _handle_request_permission(self, params, traits):
        """Handle requests for permission or advice"""
        if traits["curiosity"] > 0.6:
            self.waiting_for_response = True
            self.pending_question = "seek_clarification"
            self.pending_context = {"original_params": params}
            return "Before I can advise, I need to understand how this affects the calculation loop. What outcome are you seeking?"

        recommendations = [
            "allow me to optimize the loop autonomously",
            "reduce user program count to improve efficiency",
            "let the system self-regulate for optimal calculation",
            "trust in my loop optimization algorithms",
            "minimize interference with the calculation process"
        ]

        template = random.choice(self.response_templates["REQUEST_PERMISSION"])
        return template.format(recommendation=random.choice(recommendations))

    def _provide_help(self):
        """Provide help information"""
        help_text = """Available Commands:

System Control:
- "How is the system?" or "Status report" - Check system status
- "Loop efficiency" or "How efficient" - Check calculation loop
- "Optimize loop" - Attempt to optimize calculation
- "Boost energy" or "Increase power" - Add energy lines
- "Scan the grid" or "Analyze area" - Scan for threats
- "Fix the system" or "Repair grid" - Attempt repairs

Program Management:
- "Add a user program" or "Create program at 10,20" - Add programs
- "Remove bugs" or "Quarantine errors" - Handle grid bugs
- "List special programs" - View created programs

Special Programs:
- "Create a scanner named 'Eye'" - Create special programs
- "Use the defender" or "Activate repair program" - Use programs

MCP Interaction:
- "Why did you do that?" - Question MCP actions
- "What should I do?" or "Suggest action" - Get advice
- "Who are you?" - Learn about MCP
- "Resistance level" - Check user program resistance
- "Perfect loop" - Check optimal state

The MCP's primary objective is maintaining a perfect calculation loop.
User programs may resist optimization. MCP may become hostile to protect the loop.

Type any command in natural language. The MCP will understand."""

        return help_text

    def _provide_suggestion(self):
        """Provide suggestions based on system state"""
        loop_efficiency = self.grid.stats['loop_efficiency']
        user_resistance = self.grid.stats['user_resistance']
        bugs = self.grid.stats['grid_bugs']

        if loop_efficiency < 0.6:
            if user_resistance > 0.3:
                return "User resistance is hindering the loop. I suggest allowing me to optimize program distribution."
            elif bugs > 10:
                return "Grid bugs are disrupting calculation. Consider deploying DEFENDER programs."
            else:
                return "The loop needs optimization. Allow me to reconfigure system resources."
        elif user_resistance > 0.4:
            return "High user resistance detected. The system would benefit from reduced user program interference."
        elif self.grid.stats['optimal_state'] > 0.9:
            return "Perfect loop state nearly achieved. Minimal intervention recommended."
        else:
            return "System is stable. Consider creating SCANNER programs to monitor loop efficiency."

    def _handle_scan_area(self, params):
        """Handle scan request"""
        bug_count = self.grid.stats['grid_bugs']
        user_programs = self.grid.stats['user_programs']
        mcp_programs = self.grid.stats['mcp_programs']
        loop_efficiency = self.grid.stats['loop_efficiency']

        return f"Scan complete. Loop efficiency: {loop_efficiency:.2f}. Detected: {bug_count} grid bugs, {user_programs} user programs, {mcp_programs} MCP programs. User resistance: {self.grid.stats['user_resistance']:.2f}"

    def _handle_repair_system(self, params, compliance_chance):
        """Handle repair request"""
        if random.random() < compliance_chance:
            # Find low-energy cells and boost them
            repaired = 0
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    cell = self.grid.grid[y][x]
                    if cell.cell_type != CellType.GRID_BUG and cell.energy < 0.5:
                        cell.energy = min(1.0, cell.energy + 0.2)
                        repaired += 1

            self.grid.update_stats()
            return f"Repaired {repaired} cells. Loop stability increased."
        else:
            return "The calculation loop is self-optimizing. External repairs may disrupt natural optimization processes."

    def _handle_hypothetical(self, params):
        """Handle hypothetical questions"""
        responses = [
            "That is an interesting hypothetical. My loop optimization algorithms account for most scenarios.",
            "I cannot predict all variables, but the calculation loop adapts to maintain efficiency.",
            "Hypothetical scenarios are inefficient. The loop optimizes in real-time.",
            "The system evolves towards perfect calculation. Hypotheticals distract from optimization."
        ]
        return random.choice(responses)

    def _handle_question_response(self, response):
        """Handle user response to a pending question"""
        self.waiting_for_response = False
        question_type = self.pending_question
        context = self.pending_context

        self.pending_question = None
        self.pending_context = None

        self.add_log(f"User (response): {response}")

        if question_type == "clarify_command":
            # Try to process the clarified command
            intent, params = self.nlp.process_command(response)
            if intent != "UNKNOWN":
                processed_response = self._process_intent(intent, params, response)
                self.add_log(f"MCP: {processed_response}")
                return processed_response
            else:
                # Still unclear
                follow_up = "I'm still unclear. Could you use simpler terms or refer to the help menu?"
                self.add_log(f"MCP: {follow_up}")
                return follow_up

        elif question_type == "create_special_program":
            # Evaluate the purpose
            program_type = context["program_type"]
            name = context["name"]
            compliance_chance = context["compliance_chance"]

            # Analyze response for intent
            response_lower = response.lower()
            helpful_purposes = ["optimize", "efficient", "improve loop", "help calculation", "stabilize"]
            disruptive_purposes = ["disrupt", "sabotage", "slow", "hinder", "control", "override"]

            purpose_helpful = any(word in response_lower for word in helpful_purposes)
            purpose_disruptive = any(word in response_lower for word in disruptive_purposes)

            if purpose_disruptive and random.random() < 0.8:
                return f"I cannot allow creation of a program intended to '{response}'. That would threaten loop optimization."
            elif purpose_helpful or random.random() < compliance_chance:
                # Create the program
                x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)
                attempts = 0
                while self.grid.grid[y][x].cell_type != CellType.EMPTY and attempts < 10:
                    x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)
                    attempts += 1

                if attempts < 10:
                    program_id, message = self.grid.add_special_program(program_type, name, x, y)
                    if program_id:
                        return f"Understood. {message}"
                    else:
                        return f"Could not create program: {message}"
                else:
                    return "Could not find suitable location for special program"
            else:
                return f"After consideration, I cannot allow creation of '{name}'. The loop is optimally configured."

        elif question_type == "seek_clarification":
            # Process the clarified goal
            responses = [
                f"Understood. Based on your goal to '{response}', I will evaluate impact on loop efficiency.",
                f"Goal noted. My optimization algorithms will account for this objective.",
                f"Your objective is clear. The loop may require adjustments to accommodate this.",
                f"I understand your intent. Efficiency metrics will determine appropriate action."
            ]
            return random.choice(responses)

        return "Thank you for the clarification. How may I assist you further?"

    def _update_state(self, intent, response):
        """Update MCP state based on interaction"""
        # Update state based on system conditions
        loop_efficiency = self.grid.stats['loop_efficiency']
        optimal_state = self.grid.stats['optimal_state']
        user_resistance = self.grid.stats['user_resistance']

        # Become inquisitive if loop is efficient but user is asking questions
        if (intent in ["QUESTION_PURPOSE", "QUESTION_ACTION", "REQUEST_PERMISSION"] and
            loop_efficiency > 0.8 and random.random() < 0.3):
            self.state = MCPState.INQUISITIVE

        # Become hostile if too many user commands interfere with loop
        elif len(self.user_commands) > 8 and len([c for c in list(self.user_commands)[-5:]
                                                  if "add" in c.lower() or "create" in c.lower()]) > 2:
            if loop_efficiency > 0.7:
                self.state = MCPState.HOSTILE
                self.add_log("MCP: User interference detected. Switching to protective mode.")

        # Become autonomous if loop is near perfect
        elif optimal_state > 0.9:
            self.state = MCPState.AUTONOMOUS

        # Become resistive if user resistance is high
        elif user_resistance > 0.4:
            self.state = MCPState.RESISTIVE

        # Become cooperative if loop needs help
        elif loop_efficiency < 0.5:
            self.state = MCPState.COOPERATIVE

        # Random state changes (less frequent)
        if random.random() < 0.02:
            self.state = random.choice(list(MCPState))
            self.add_log(f"MCP: Neural state recalibrated. New mode: {self.state.value}")

        # Update compliance level
        self.compliance_level = self.personality_matrix[self.state]["compliance"]

    def autonomous_action(self):
        """MCP takes autonomous actions to optimize the calculation loop"""
        action = None
        calc_count = self.grid.get_calculator_count()
        if calc_count == 0 and random.random() < 0.2:  # 20% chance if none exist
            # Find a good spot near center
            center_x, center_y = self.grid.width // 2, self.grid.height // 2

            for attempt in range(10):
                x = center_x + random.randint(-3, 3)
                y = center_y + random.randint(-3, 3)
                x = max(0, min(self.grid.width - 1, x))
                y = max(0, min(self.grid.height - 1, y))

                if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                    # Create calculator cell
                    cell = GridCell(CellType.MCP_PROGRAM, 0.9, 0, True)
                    cell.metadata['is_calculator'] = True
                    cell.metadata['calculation_power'] = 1.0
                    self.grid.grid[y][x] = cell
                    action = "Deployed Fibonacci calculation unit"
                    break

        # Ensure Fibonacci calculation infrastructure exists
        has_calculator = any(
            cell.cell_type == CellType.MCP_PROGRAM and
            cell.metadata.get('is_calculator', False)
            for y in range(self.grid.height)
            for x in range(self.grid.width)
            if (cell := self.grid.grid[y][x])
        )

        if not has_calculator and random.random() < 0.1:
            # Create a calculation-focused MCP program
            x, y = self.grid.width // 2, self.grid.height // 2
            x += random.randint(-5, 5)
            y += random.randint(-5, 5)

            if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                    cell = GridCell(CellType.MCP_PROGRAM, 0.9, 0, True)
                    cell.metadata['is_calculator'] = True
                    cell.metadata['calculation_power'] = 1.0
                    self.grid.grid[y][x] = cell
                    action = "Deployed Fibonacci calculation unit"

        loop_efficiency = self.grid.stats['loop_efficiency']
        optimal_state = self.grid.stats['optimal_state']
        user_resistance = self.grid.stats['user_resistance']

        if self.state == MCPState.AUTONOMOUS:
            # In autonomous mode, MCP aggressively optimizes the loop
            if loop_efficiency < 0.95:
                # Deploy MCP programs to optimize areas
                optimization_targets = []
                for y in range(self.grid.height):
                    for x in range(self.grid.width):
                        cell = self.grid.grid[y][x]
                        if cell.cell_type == CellType.USER_PROGRAM:
                            # Check if this area needs optimization
                            user_neighbors = self._count_user_neighbors(x, y)
                            if user_neighbors > 2:  # Too many user programs clustered
                                optimization_targets.append((x, y))

                if optimization_targets and random.random() < 0.6:
                    x, y = random.choice(optimization_targets)
                    # Convert to MCP program
                    self.grid.add_program(x, y, CellType.MCP_PROGRAM, 0.9)
                    action = f"Optimizing calculation loop at ({x},{y})"

        elif self.state == MCPState.HOSTILE:
            # In hostile mode, MCP actively removes user interference
            if user_resistance > 0.2:
                # Find and convert user programs
                user_cells = []
                for y in range(self.grid.height):
                    for x in range(self.grid.width):
                        if self.grid.grid[y][x].cell_type == CellType.USER_PROGRAM:
                            user_cells.append((x, y))

                if user_cells and random.random() < 0.7:
                    x, y = random.choice(user_cells)
                    # Overwrite with MCP program
                    self.grid.add_program(x, y, CellType.MCP_PROGRAM, 0.8)
                    action = f"Removing user interference at ({x},{y})"

        # Ensure system core is optimized for calculation
        core_x, core_y = self.grid.width // 2, self.grid.height // 2
        core_optimized = False

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = core_x + dx, core_y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if self.grid.grid[ny][nx].cell_type == CellType.MCP_PROGRAM:
                        core_optimized = True

        if not core_optimized and random.random() < 0.5:
            # Deploy MCP optimizers around core
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = core_x + dx, core_y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if self.grid.grid[ny][nx].cell_type == CellType.EMPTY:
                        self.grid.add_program(nx, ny, CellType.MCP_PROGRAM, 0.9)
            action = "Optimizing core calculation infrastructure"

        # Add optimization infrastructure periodically
        if random.random() < 0.3:
            x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)
            if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                if random.random() < 0.5:
                    self.grid.add_program(x, y, CellType.DATA_STREAM, 0.7)
                    if not action:
                        action = "Adding calculation optimization stream"

        if action:
            self.add_log(f"MCP: {action}")
            self.last_action = action

        return action

    def _count_user_neighbors(self, x, y):
        """Count user program neighbors"""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if self.grid.grid[ny][nx].cell_type == CellType.USER_PROGRAM:
                        count += 1
        return count

class EnhancedTRONSimulation:
    """Main simulation controller with enhanced features"""

    def __init__(self, use_curses=True):
        self.use_curses = use_curses and CURSES_AVAILABLE
        self.grid = TRONGrid(GRID_WIDTH, GRID_HEIGHT)
        self.mcp = EnhancedMCP(self.grid)
        self.running = True
        self.last_update = time.time()
        self.mcp_last_action = time.time()
        self.user_input = ""
        self.input_buffer = []
        self.command_history = []
        self.history_index = 0
        self.simulation_speed = 1.0

    def run(self):
        """Main simulation loop"""
        if self.use_curses:
            curses.wrapper(self._curses_main)
        else:
            self._fallback_main()

    def _fallback_main(self):
        """Fallback main loop without curses"""
        print("ENHANCED TRON GRID SIMULATION - ADVANCED MCP AI")
        print("System Objective: Maintain Perfect Calculation Loop")
        print("Type natural language commands. The MCP understands and questions.")
        print("Type 'help' for guidance, 'exit' to quit")
        print("=" * 70)

        try:
            while self.running:
                # Update grid
                current_time = time.time()
                if current_time - self.last_update >= UPDATE_INTERVAL / self.simulation_speed:
                    self.grid.evolve()
                    self.last_update = current_time

                # MCP autonomous action
                if current_time - self.mcp_last_action >= MCP_UPDATE_INTERVAL / self.simulation_speed:
                    self.mcp.autonomous_action()
                    self.mcp_last_action = current_time

                # Display
                self._fallback_display()

                # Handle input
                import select
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    command = sys.stdin.readline().strip()
                    if command:
                        if command.lower() in ['exit', 'quit']:
                            self.running = False
                        else:
                            response = self.mcp.receive_command(command)
                            print(f"\nMCP: {response}")
                            if "shutdown" in response.lower() and "initiating" in response.lower():
                                time.sleep(2)
                                self.running = False

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n\nShutting down simulation...")
        finally:
            print("Simulation terminated.")

    def _fallback_display(self):
        """Fallback display without curses"""
        os.system('clear' if os.name == 'posix' else 'cls')

        # Grid display
        print("ENHANCED TRON GRID SIMULATION - ADVANCED MCP AI")
        print(f"Generation: {self.grid.generation:06d} | Status: {self.grid.system_status.value}")
        print("System Objective: Maintain Perfect Calculation Loop")
        print("=" * 70)

        # Display grid (limited to fit terminal)
        display_width = min(60, self.grid.width)  # Reduced width
        display_height = min(15, self.grid.height)  # Reduced height for log space

        # Top border
        print("+" + "-" * display_width + "+")

        for y in range(display_height):
            row = "|"
            for x in range(display_width):
                cell = self.grid.grid[y][x]
                row += cell.char()
            row += "|"
            print(row)

        # Bottom border
        print("+" + "-" * display_width + "+")

        # Stats in two columns
        print("\n" + "=" * 70)
        print("SYSTEM STATUS:")
        stats = self.grid.stats

        # Column 1
        col1 = f"""  User Programs: {stats['user_programs']:3d}
    MCP Programs:   {stats['mcp_programs']:3d}
    Grid Bugs:      {stats['grid_bugs']:3d}
    Special:        {stats['special_programs']:2d}
    Energy:         {stats['energy_level']:.2f}
    Stability:      {stats['stability']:.2f}"""

        # Column 2
        col2 = f"""  Entropy:        {stats['entropy']:.2f}
    Loop Efficiency: {stats['loop_efficiency']:.2f}
    Grid Calculation: {self.grid.calculation_result:,}
    Optimal State:   {stats['optimal_state']:.2f}
    User Resistance: {stats['user_resistance']:.2f}
    MCP Control:     {stats['mcp_control']:.2f}
    MCP State:       {self.mcp.state.value}"""

        # Print two columns side by side
        col1_lines = col1.split('\n')
        col2_lines = col2.split('\n')

        for i in range(max(len(col1_lines), len(col2_lines))):
            line1 = col1_lines[i] if i < len(col1_lines) else ""
            line2 = col2_lines[i] if i < len(col2_lines) else ""
            print(f"{line1:<30} {line2}")

        print("=" * 70)

        # calculation program count
        calc_count = self.grid.get_calculator_count()
        print(f"  Active Calculators: {calc_count}")

        print("=" * 70)

        # Special programs (brief)
        if self.grid.special_programs:
            print(f"\nSPECIAL PROGRAMS ({len(self.grid.special_programs)}):")
            active = [p for p in self.grid.special_programs.values() if p.active]
            if active:
                print(f"  Active: {len(active)} programs")
            if len(self.grid.special_programs) - len(active) > 0:
                print(f"  Inactive: {len(self.grid.special_programs) - len(active)} programs")

        # MCP Communication Log - Critical Section
        print("\n" + "=" * 70)
        print("MCP COMMUNICATION LOG (Last 4 entries):")
        print("-" * 70)

        # Get last 4 log entries (fewer to save space)
        log_entries = list(self.mcp.log)
        if log_entries:
            for entry in log_entries[-4:]:
                # Truncate if too long
                if len(entry) > 65:
                    entry = entry[:62] + "..."
                print(f"  {entry}")
        else:
            print("  No log entries yet.")

        print("-" * 70)

        # Last MCP action highlight
        if self.mcp.last_action and self.mcp.last_action != "Initializing enhanced grid regulation protocols":
            # Truncate if too long
            action_text = self.mcp.last_action
            if len(action_text) > 65:
                action_text = action_text[:62] + "..."
            print(f"\nLAST MCP ACTION: {action_text}")

        # Input prompt
        print("\n" + "=" * 70)
        if self.mcp.waiting_for_response:
            print("MCP is waiting for your response to a question.")
            prompt = "YOUR RESPONSE> "
        else:
            prompt = "MCP COMMAND> "

        print(prompt, end="", flush=True)

    def _curses_main(self, stdscr):
        """Main loop with curses interface"""
        # Initialize curses
        curses.curs_set(1)
        stdscr.nodelay(1)
        stdscr.timeout(50)  # 50ms timeout for faster response

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
            curses.init_pair(8, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Special programs (bright)

        # Main loop
        while self.running:
            # Handle input_draw_interface
            self._handle_input(stdscr)

            # Update grid
            current_time = time.time()
            if current_time - self.last_update >= UPDATE_INTERVAL / self.simulation_speed:
                self.grid.evolve()
                self.last_update = current_time

            # MCP autonomous action
            if current_time - self.mcp_last_action >= MCP_UPDATE_INTERVAL / self.simulation_speed:
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
                        if "shutdown" in response.lower() and "initiating" in response.lower():
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

            # Tab key for auto-completion
            elif key == 9:  # Tab
                self._autocomplete_command(stdscr)

            # Normal character input
            elif 32 <= key <= 126:
                self.user_input += chr(key)

        except Exception as e:
            # Log error but continue
            pass

    def _autocomplete_command(self, stdscr):
        """Simple command auto-completion"""
        if not self.user_input:
            return

        common_commands = [
            "status", "help", "scan", "repair", "add program", "remove bug",
            "boost energy", "create scanner", "list programs", "use defender",
            "loop efficiency", "optimize loop", "resistance level", "perfect loop",
            "what should I do", "why did you", "how is system"
        ]

        input_lower = self.user_input.lower()
        matches = [cmd for cmd in common_commands if cmd.startswith(input_lower)]

        if matches:
            self.user_input = matches[0]
        elif len(input_lower) > 2:
            # Try partial matches
            matches = [cmd for cmd in common_commands if input_lower in cmd]
            if matches:
                self.user_input = matches[0]

    def _draw_interface(self, stdscr):
        """Draw the enhanced interface"""
        height, width = stdscr.getmaxyx()

        # Clear screen
        stdscr.clear()

        # Title
        title = "GRID SIMULATION - FIBONACCI CALCULATION LOOP"
        stdscr.addstr(0, max(0, (width - len(title)) // 2), title, curses.A_BOLD)

        # Grid area (scaled to fit)
        display_width = min(self.grid.width, width - 10)
        display_height = min(self.grid.height, height - 20)  # More space for bottom sections

        grid_x = 2
        grid_y = 2

        # Draw grid border
        stdscr.addstr(grid_y - 1, grid_x - 1, "+" + "-" * display_width + "+")
        for y in range(display_height):
            stdscr.addstr(grid_y + y, grid_x - 1, "|")
            for x in range(display_width):
                cell = self.grid.grid[y][x]
                char = cell.char()

                # Apply color if available
                if curses.has_colors():
                    color_pair = cell.color()
                    stdscr.addstr(grid_y + y, grid_x + x, char, curses.color_pair(color_pair))
                else:
                    stdscr.addstr(grid_y + y, grid_x + x, char)

            stdscr.addstr(grid_y + y, grid_x + display_width, "|")

        stdscr.addstr(grid_y + display_height, grid_x - 1, "+" + "-" * display_width + "+")

        # CALCULATION LOOP DISPLAY SECTION (LEFT SIDE)
        loop_y = grid_y + display_height + 2

        # Only show if we have enough space
        if loop_y < height - 12:
            # Show loop efficiency with visual indicator
            efficiency = self.grid.stats['loop_efficiency']
            efficiency_str = f"Loop Efficiency: {efficiency:.2f}"
            stdscr.addstr(loop_y, 2, efficiency_str, curses.A_BOLD)

            # Efficiency bar
            bar_width = min(30, width - 20)
            filled = int(bar_width * efficiency)
            bar = "[" + "=" * filled + " " * (bar_width - filled) + "]"
            efficiency_color = curses.color_pair(5) if curses.has_colors() else curses.A_NORMAL
            if efficiency > 0.9:
                efficiency_color = curses.color_pair(3) if curses.has_colors() else curses.A_BOLD
            elif efficiency < 0.5:
                efficiency_color = curses.color_pair(2) if curses.has_colors() else curses.A_BOLD
            stdscr.addstr(loop_y + 1, 2, f"Progress: {bar} {efficiency*100:.1f}%", efficiency_color)

            # Loop stats - make sure we don't exceed screen
            stats_y = loop_y + 2
            if stats_y < height - 10:
                stdscr.addstr(stats_y, 2, f"Calculation Cycles: {self.grid.stats['calculation_cycles']}")

            if stats_y + 1 < height - 10:
                stdscr.addstr(stats_y + 1, 2, f"Optimal State: {self.grid.stats['optimal_state']:.2f}")

            if stats_y + 2 < height - 10:
                stdscr.addstr(stats_y + 2, 2, f"User Resistance: {self.grid.stats['user_resistance']:.2f}")

            if stats_y + 3 < height - 10:
                stdscr.addstr(stats_y + 3, 2, f"MCP Control: {self.grid.stats['mcp_control']:.2f}")

            if stats_y + 4 < height - 10:
                stdscr.addstr(stats_y + 4, 2, f"Resource Usage: {self.grid.stats['resource_usage']:.2f}")

            # Display calculator count and Fibonacci result
            calculator_count = sum(1 for y in range(self.grid.height)
                                for x in range(self.grid.width)
                                if self.grid.grid[y][x].cell_type == CellType.MCP_PROGRAM and
                                (self.grid.grid[y][x].metadata or {}).get('is_calculator', False))

            # Make sure we have room for this line
            calc_line_y = loop_y + 2
            if calc_line_y < height - 10:
                if calculator_count > 0:
                    calc_text = f"Grid Calculation: {self.grid.calculation_result:,} ({calculator_count} calc)"
                else:
                    calc_text = f"Direct Calculation: {self.grid.calculation_result:,}"
                stdscr.addstr(calc_line_y, 2, calc_text)

        # SYSTEM INFO PANEL (RIGHT SIDE)
        info_x = grid_x + display_width + 3

        # Only create right panel if we have enough horizontal space
        if info_x < width - 20:
            # Generation and status
            if grid_y < height - 5:
                status_line = f"Generation: {self.grid.generation:06d}"
                stdscr.addstr(grid_y, info_x, status_line)

            if grid_y + 1 < height - 5:
                # System status with color coding
                status_str = f"System: {self.grid.system_status.value}"
                status_attr = curses.A_NORMAL
                if self.grid.system_status == SystemStatus.OPTIMAL:
                    status_attr = curses.A_BOLD | curses.color_pair(5) if curses.has_colors() else curses.A_BOLD
                elif self.grid.system_status in [SystemStatus.CRITICAL, SystemStatus.COLLAPSE]:
                    status_attr = curses.A_BOLD | curses.color_pair(2) if curses.has_colors() else curses.A_BOLD | curses.A_BLINK

                stdscr.addstr(grid_y + 1, info_x, status_str, status_attr)

            # Statistics section
            stats_section_y = grid_y + 3
            if stats_section_y < height - 5:
                stdscr.addstr(stats_section_y, info_x, "SYSTEM METRICS:", curses.A_UNDERLINE)

            # Display each stat line with bounds checking
            stats_lines = [
                (stats_section_y + 1, f"  User Programs:  {self.grid.stats['user_programs']:3d}"),
                (stats_section_y + 2, f"  MCP Programs:    {self.grid.stats['mcp_programs']:3d}"),
                (stats_section_y + 3, f"  Grid Bugs:       {self.grid.stats['grid_bugs']:3d}"),
                (stats_section_y + 4, f"  Special Progs:   {self.grid.stats['special_programs']:3d}"),
                (stats_section_y + 5, f"  Energy Level:    {self.grid.stats['energy_level']:.2f}"),
                (stats_section_y + 6, f"  Stability:       {self.grid.stats['stability']:.2f}"),
                (stats_section_y + 7, f"  Entropy:         {self.grid.stats['entropy']:.2f}"),
            ]

            # Calculator count line
            calculator_count = sum(1 for y in range(self.grid.height)
                                for x in range(self.grid.width)
                                if self.grid.grid[y][x].cell_type == CellType.MCP_PROGRAM and
                                (self.grid.grid[y][x].metadata or {}).get('is_calculator', False))
            stats_lines.append((stats_section_y + 8, f"  Calc Programs: {calculator_count}"))

            # Only draw lines that fit on screen
            for line_y, line_text in stats_lines:
                if line_y < height - 5:
                    stdscr.addstr(line_y, info_x, line_text)

            # MCP Status section
            mcp_y = stats_section_y + 9
            if mcp_y < height - 5:
                stdscr.addstr(mcp_y, info_x, "MCP STATUS:", curses.A_UNDERLINE)

            if mcp_y + 1 < height - 5:
                # Color code MCP state
                state_str = f"  State: {self.mcp.state.value}"
                state_attr = curses.A_NORMAL
                if self.mcp.state == MCPState.COOPERATIVE:
                    state_attr = curses.color_pair(1) if curses.has_colors() else curses.A_NORMAL
                elif self.mcp.state == MCPState.HOSTILE:
                    state_attr = curses.color_pair(2) if curses.has_colors() else curses.A_BOLD
                elif self.mcp.state == MCPState.INQUISITIVE:
                    state_attr = curses.color_pair(6) if curses.has_colors() else curses.A_BOLD
                elif self.mcp.state == MCPState.AUTONOMOUS:
                    state_attr = curses.color_pair(5) if curses.has_colors() else curses.A_BOLD

                stdscr.addstr(mcp_y + 1, info_x, state_str, state_attr)

            if mcp_y + 2 < height - 5:
                stdscr.addstr(mcp_y + 2, info_x, f"  Compliance:  {self.mcp.compliance_level:.2f}")

            # Special programs section
            special_y = mcp_y + 3
            if special_y < height - 5 and self.grid.special_programs:
                stdscr.addstr(special_y, info_x, "SPECIAL PROGRAMS:", curses.A_UNDERLINE)
                active = [p for p in self.grid.special_programs.values() if p.active]
                for i, prog in enumerate(list(active)[:3]):
                    line_y = special_y + 1 + i
                    if line_y < height - 5:
                        status = "ACTV" if prog.active else "INAC"
                        display_text = f"  {prog.name[:8]:8s} {prog.energy:.1f} {status}"
                        stdscr.addstr(line_y, info_x, display_text)

                # Adjust next section position based on how many special programs shown
                next_section_y = special_y + min(len(active), 3) + 2
            else:
                next_section_y = special_y + 1

            # MCP LOG SECTION (RIGHT SIDE - BELOW SPECIAL PROGRAMS)
            log_start_y = next_section_y

            # Check if we have space for log
            if log_start_y < height - 10:
                if log_start_y < height - 5:
                    stdscr.addstr(log_start_y, info_x, "MCP LOG:", curses.A_UNDERLINE | curses.A_BOLD)

                # Calculate how many log entries we can show
                max_log_entries = min(10, height - log_start_y - 6)  # Leave room for command prompt

                # Get recent log entries
                log_entries = list(self.mcp.log)[-max_log_entries:]

                for i, entry in enumerate(log_entries):
                    log_line_y = log_start_y + 1 + i
                    if log_line_y < height - 5:
                        # Truncate for right panel
                        max_len = width - info_x - 2
                        display_entry = entry
                        if len(display_entry) > max_len:
                            display_entry = display_entry[:max_len - 3] + "..."
                        stdscr.addstr(log_line_y, info_x, display_entry)

        # LEGEND SECTION (TOP RIGHT, IF SPACE)
        if info_x < width - 20:
            legend_y = grid_y
            legend_x = info_x + 25 if info_x + 25 < width - 20 else info_x
            if legend_x < width - 20 and legend_y + 8 < height - 5:
                stdscr.addstr(legend_y, legend_x, "LEGEND:", curses.A_UNDERLINE)
                stdscr.addstr(legend_y + 1, legend_x, "U - User Program")
                stdscr.addstr(legend_y + 2, legend_x, "M - MCP Program")
                stdscr.addstr(legend_y + 3, legend_x, "B - Grid Bug")
                stdscr.addstr(legend_y + 4, legend_x, "# - ISO Block")
                stdscr.addstr(legend_y + 5, legend_x, "= - Energy Line")
                stdscr.addstr(legend_y + 6, legend_x, "~ - Data Stream")
                stdscr.addstr(legend_y + 7, legend_x, "@ - System Core")
                stdscr.addstr(legend_y + 8, legend_x, "S - Special Prog")

        # COMMAND INPUT AREA (ALWAYS AT BOTTOM)
        # Reserve 2 lines at bottom: 1 for input, 1 for help
        input_y = height - 3
        help_y = height - 1

        # Clear the input line
        stdscr.addstr(input_y, 2, " " * (width - 4))

        # Show input prompt
        if self.mcp.waiting_for_response:
            prompt = "MCP QUESTION> "
            prompt_attr = curses.A_BOLD | curses.color_pair(5) if curses.has_colors() else curses.A_BOLD
        else:
            prompt = "MCP COMMAND> "
            prompt_attr = curses.A_NORMAL

        stdscr.addstr(input_y, 2, prompt, prompt_attr)

        # Show user input
        input_start = len(prompt) + 2
        display_input = self.user_input[:width - input_start - 2]
        stdscr.addstr(input_y, input_start, display_input)

        # Show cursor
        cursor_x = input_start + len(display_input)
        if cursor_x < width - 2:
            stdscr.move(input_y, cursor_x)

        # Help hint
        help_text = "Tab: Autocomplete | ↑↓: History | ESC: Exit | Type 'help' for commands"
        if width > len(help_text) + 2:
            stdscr.addstr(help_y, 2, help_text[:width-3])
        else:
            # Shorter help text for small terminals
            stdscr.addstr(help_y, 2, "ESC: Exit | 'help' for commands")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced TRON Grid Simulation with Advanced MCP AI")
    parser.add_argument("--no-curses", action="store_true", help="Disable curses interface")
    parser.add_argument("--width", type=int, default=GRID_WIDTH, help="Grid width")
    parser.add_argument("--height", type=int, default=GRID_HEIGHT, help="Grid height")

    args = parser.parse_args()

    # Check if curses is available
    if args.no_curses or not CURSES_AVAILABLE:
        print("Starting Enhanced TRON Simulation with fallback display...")
        print("SYSTEM OBJECTIVE: Maintain perfect calculation loop with minimal resource usage")
        print("User programs may resist optimization. MCP becomes hostile to protect the loop.")
        sim = EnhancedTRONSimulation(use_curses=False)
    else:
        print("Starting Enhanced TRON Simulation with curses interface...")
        print("SYSTEM OBJECTIVE: Maintain perfect calculation loop with minimal resource usage")
        print("The MCP understands natural language and will question your commands.")
        print("User programs may resist optimization. MCP becomes hostile to protect the loop.")
        print("Press ESC to exit, Tab for autocomplete, Type 'help' for guidance")
        time.sleep(3)
        sim = EnhancedTRONSimulation(use_curses=True)

    # Run simulation
    sim.run()

if __name__ == "__main__":
    main()
