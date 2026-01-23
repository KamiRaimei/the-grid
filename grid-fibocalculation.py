#!/usr/bin/env python3
"""
Grid Simulation - Cell-based Fibonacci calculation.
System directive: Maintain a perfect calculation loop with efficiency through adaptive learning.
"""

import os
import sys
import time
import random
import re
import json
import signal
import threading
from queue import Queue
from collections import deque, defaultdict
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any, Callable
import argparse
from datetime import datetime
import math


try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy module not available. Using fallback calculations.")

sys.set_int_max_str_digits(10000)

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
UPDATE_INTERVAL = 0.25  # seconds (simulation update rate)
MCP_UPDATE_INTERVAL = 5  # seconds (controls MCP AUTONOMOUS action rate)
MAX_SPECIAL_PROGRAMS = 10
SPECIAL_PROGRAM_TYPES = ["SCANNER", "DEFENDER", "REPAIR", "SABOTEUR", "RECONFIGURATOR", "ENERGY_HARVESTER", "FIBONACCI_CALCULATOR"]

# PERSONALITY FILE
PERSONALITY_FILE = "mcp_personality.json"

# Cell class init
class CellType(Enum):
    EMPTY = 0
    USER_PROGRAM = 1      # User created program cell
    MCP_PROGRAM = 2       # MCP created program cell
    GRID_BUG = 3          # Corruptions/errors
    ISO_BLOCK = 4         # Isolated/containment blocks
    ENERGY_LINE = 5       # Power lines
    DATA_STREAM = 6       # Data streams
    SYSTEM_CORE = 7       # Core system blocks
    SPECIAL_PROGRAM = 8   # User-created special programs
    FIBONACCI_PROCESSOR = 9  # Fibonacci calculation processor

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
    INQUISITIVE = "INQUISITIVE"
    LEARNING = "LEARNING"

# ==================== CELL VISUALIZATION ====================

@dataclass
class GridCell:
    cell_type: CellType
    energy: float  # 0.0 to 1.0
    age: int = 0
    stable: bool = True
    special_program_id: Optional[str] = None  # ID if this is a special program
    metadata: Dict[str, Any] = None
    animation_frame: int = 0
    processing: bool = False
    calculation_contribution: float = 0.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Initialize animation for certain cell types
        if self.cell_type in [CellType.DATA_STREAM, CellType.ENERGY_LINE, CellType.FIBONACCI_PROCESSOR]:
            self.animation_frame = random.randint(0, 3)

    def char(self):
        # Use single-character representations only to maintain grid alignment
        if self.cell_type == CellType.DATA_STREAM:
            # Single character data stream animation
            stream_chars = ['~', '~', '≈', '≈']  # Use simpler chars for alignment
            return stream_chars[self.animation_frame % 4]
        elif self.cell_type == CellType.ENERGY_LINE:
            # Single character energy line
            energy_chars = ['=', '=', '≡', '≡']  # Use simpler chars for alignment
            return energy_chars[self.animation_frame % 4]
        if self.cell_type == CellType.FIBONACCI_PROCESSOR:
            # Enhanced animation for Fibonacci processors
            if self.processing:
                # When actively calculating, show more dynamic animation
                processor_chars = ['◉', '◎', '●', '○', '◌', '⊕', '⊗', '∅']
                return processor_chars[self.animation_frame % 8]
            else:
                # Idle state
                processor_chars = ['F', 'φ', 'f', 'Φ']
                return processor_chars[self.animation_frame % 4]
        elif self.processing:
            # Use single-character processing indicators
            return '○' if self.animation_frame % 2 == 0 else '●'

        chars = {
            CellType.EMPTY: ' ',
            CellType.USER_PROGRAM: 'U',
            CellType.MCP_PROGRAM: 'M',
            CellType.GRID_BUG: 'B',
            CellType.ISO_BLOCK: '#',
            CellType.ENERGY_LINE: '=',
            CellType.DATA_STREAM: '~',
            CellType.SYSTEM_CORE: '@',
            CellType.SPECIAL_PROGRAM: 'S',
            CellType.FIBONACCI_PROCESSOR: 'F'
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
            CellType.SPECIAL_PROGRAM: 8,   # Bright Cyan
            CellType.FIBONACCI_PROCESSOR: 9  # Bright Yellow
        }
        return colors.get(self.cell_type, 0)

    def update_animation(self):
        """Update animation frame for visual effects"""
        if self.cell_type in [CellType.DATA_STREAM, CellType.ENERGY_LINE, CellType.FIBONACCI_PROCESSOR]:
            self.animation_frame += 1
            # Visual feedback for processing
            if self.calculation_contribution > 0:
                self.processing = True
                self.animation_frame += 2  # Faster animation when processing
            else:
                self.processing = False

# ==================== FIBONACCI CALCULATION ====================

class GridFibonacciCalculator:
    """Manages Fibonacci calculation through grid cell cooperation"""

    def __init__(self, grid):
        self.grid = grid
        self.fib_sequence = [0, 1]
        self.calculation_accumulator = 0.0
        self.last_calculation_time = time.time()
        self.calculation_rate = 0.0
        self.total_contributions = 0

        # Tracking for cell contributions
        self.cell_contributions = {}
        self.contribution_history = deque(maxlen=100)

        # Efficiency metrics
        self.efficiency_score = 0.5
        self.optimization_level = 0.0

        # Calculation threading
        self.calculation_queue = Queue()
        self.calculation_thread = threading.Thread(target=self._calculation_worker, daemon=True)
        self.calculation_thread.start()

    def _calculation_worker(self):
        """Background worker with proper sleep to prevent CPU hogging"""
        while True:
            try:
                # Process calculations with sleep to yield CPU
                if not self.calculation_queue.empty():
                    # Process a batch of calculations
                    batch_size = min(10, self.calculation_queue.qsize())
                    for _ in range(batch_size):
                        try:
                            task = self.calculation_queue.get_nowait()
                            # Process task
                            self.calculation_queue.task_done()
                        except:
                            break
                time.sleep(0.05)  # Increased from 0.1 to 0.05 for better responsiveness
            except:
                time.sleep(0.1)

    def calculate_next(self):
        """Calculate next Fibonacci number through cell cooperation"""
        contributions = []
        total_energy = 0
        active_calculators = 0

        # Reset processing flags
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell = self.grid.grid[y][x]
                cell.calculation_contribution = 0.0
                cell.processing = False

        # Collect contributions from all cells
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell = self.grid.grid[y][x]

                # Each cell type contributes differently
                contribution = 0.0

                if cell.cell_type == CellType.MCP_PROGRAM:
                    # MCP programs are efficient calculators
                    contribution = cell.energy * 0.3
                    if cell.metadata.get('is_calculator', False):
                        contribution *= 2.0  # Bonus for dedicated calculators
                        active_calculators += 1

                elif cell.cell_type == CellType.FIBONACCI_PROCESSOR:
                    # Special Fibonacci processors
                    contribution = cell.energy * 0.8
                    if not cell.metadata.get('temporary', False):
                        contribution *= 1.5
                    active_calculators += 1

                elif cell.cell_type == CellType.USER_PROGRAM:
                    # User programs contribute but less efficiently
                    contribution = cell.energy * 0.1

                elif cell.cell_type == CellType.DATA_STREAM:
                    # Data streams facilitate calculation
                    contribution = cell.energy * 0.15

                elif cell.cell_type == CellType.ENERGY_LINE:
                    # Energy lines power calculations
                    contribution = cell.energy * 0.1

                elif cell.cell_type == CellType.SYSTEM_CORE:
                    # System core coordinates calculation
                    contribution = cell.energy * 0.25

                # Apply efficiency bonus from system stats
                system_efficiency = self.grid.stats['loop_efficiency']
                contribution *= (0.5 + system_efficiency * 0.5)

                # Apply penalty for nearby bugs
                bug_penalty = self._calculate_bug_penalty(x, y)
                contribution *= (1.0 - bug_penalty)

                # Add collaboration bonus for cells near other calculators
                if contribution > 0:
                    collaboration_bonus = self._calculate_collaboration_bonus(x, y)
                    contribution *= (1.0 + collaboration_bonus)

                if contribution > 0:
                    contributions.append((x, y, contribution))
                    total_energy += contribution
                    cell.calculation_contribution = contribution
                    cell.processing = True

        # Calculate calculation rate based on contributions
        current_time = time.time()
        time_delta = current_time - self.last_calculation_time
        if time_delta > 0:
            self.calculation_rate = total_energy / time_delta

        self.last_calculation_time = current_time

        # Processing visual feedback
        if contribution > 0 and cell.cell_type == CellType.FIBONACCI_PROCESSOR:
            cell.calculation_contribution = contribution
            cell.processing = True

            # Visual pulse effect for major contributors
            if contribution > 0.3:
                cell.metadata['pulse_strength'] = min(1.0, contribution)
                cell.metadata['pulse_timer'] = 3

        # Update calculation accumulator
        self.calculation_accumulator += total_energy
        self.total_contributions = total_energy

        # Store contribution data
        self.contribution_history.append({
            'time': current_time,
            'total_energy': total_energy,
            'active_calculators': active_calculators,
            'efficiency': self.grid.stats['loop_efficiency']
        })

        # When accumulator reaches threshold, advance Fibonacci sequence
        calculation_threshold = 1000  # Adjust for calculation speed

        steps_taken = 0
        max_steps_per_frame = 10  # Prevent infinite loops

        while self.calculation_accumulator >= calculation_threshold and steps_taken < max_steps_per_frame:
            # Calculate next Fibonacci number
            next_fib = self.fib_sequence[-2] + self.fib_sequence[-1]
            self.fib_sequence.append(next_fib)

            if len(self.fib_sequence) > 100:
                self.fib_sequence = self.fib_sequence[-100:]

            self.calculation_accumulator -= calculation_threshold
            steps_taken += 1

            if steps_taken == 1:
                self._create_calculation_visual_feedback(contributions)

        # Update efficiency score
        self._update_efficiency_score(contributions, active_calculators)

        return self.fib_sequence[-1]

    def _calculate_bug_penalty(self, x, y):
        """Calculate penalty from nearby bugs"""
        penalty = 0.0
        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if self.grid.grid[ny][nx].cell_type == CellType.GRID_BUG:
                        distance = math.sqrt(dx*dx + dy*dy)
                        penalty += 0.1 / max(1.0, distance)
        return min(0.5, penalty)

    def _calculate_collaboration_bonus(self, x, y):
        """Calculate bonus for working near other calculators"""
        bonus = 0.0
        calculator_neighbors = 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    neighbor = self.grid.grid[ny][nx]
                    if (neighbor.cell_type == CellType.MCP_PROGRAM and
                        neighbor.metadata.get('is_calculator', False)):
                        calculator_neighbors += 1
                    elif neighbor.cell_type == CellType.FIBONACCI_PROCESSOR:
                        calculator_neighbors += 2

        bonus = calculator_neighbors * 0.05
        return min(0.3, bonus)

    def _create_calculation_visual_feedback(self, contributions):
        """Create visual effects for calculation completion"""
        if not contributions:
            return

        # Find top contributors for visual highlight
        top_contributors = sorted(contributions, key=lambda x: x[2], reverse=True)[:5]

        for x, y, contribution in top_contributors:
            cell = self.grid.grid[y][x]
            # Mark cell as having contributed significantly
            cell.metadata['last_major_contribution'] = time.time()
            cell.metadata['contribution_streak'] = cell.metadata.get('contribution_streak', 0) + 1

            # Create energy pulse effect
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                        neighbor = self.grid.grid[ny][nx]
                        if neighbor.cell_type == CellType.EMPTY and random.random() < 0.3:
                            # Temporary energy spark
                            neighbor.metadata['energy_spark'] = True
                            neighbor.metadata['spark_timer'] = 3

    def _update_efficiency_score(self, contributions, active_calculators):
        """Update calculation efficiency score"""
        if not contributions:
            self.efficiency_score *= 0.95  # Decay if no contributions
            return

        # Calculate efficiency metrics
        total_contributors = len(contributions)
        avg_contribution = sum(c[2] for c in contributions) / total_contributors

        # Efficiency factors:
        # 1. Number of active calculators
        calculator_factor = min(1.0, active_calculators / 10.0)

        # 2. System loop efficiency
        system_factor = self.grid.stats['loop_efficiency']

        # 3. Energy distribution (lower entropy = better)
        energy_factor = 1.0 - self.grid.stats['entropy'] * 0.5

        # Combine factors
        new_efficiency = (calculator_factor * 0.4 +
                        system_factor * 0.3 +
                        energy_factor * 0.3)

        # Smooth update
        self.efficiency_score = 0.8 * self.efficiency_score + 0.2 * new_efficiency

        # Update optimization level based on efficiency trend
        if len(self.contribution_history) > 1:
            recent_gain = self.contribution_history[-1]['total_energy'] - self.contribution_history[-2]['total_energy']
            if recent_gain > 0:
                self.optimization_level = min(1.0, self.optimization_level + 0.05)
            else:
                self.optimization_level = max(0.0, self.optimization_level - 0.1)

    def format_fibonacci_number(self, num):
        """Format Fibonacci number to show only first 30 digits if too large"""
        try:
            num_str = str(num)
        except ValueError as e:
            # If we hit the limit, use scientific notation or truncation
            # Estimate the number of digits using log10
            if num == 0:
                return "0"

            # For very large numbers, use log10 to get approximate value
            import math
            log10 = math.log10(num)
            digits = int(log10) + 1
            if digits > 30:
                # Get first 30 digits using division
                first_30 = num // (10 ** (digits - 30))
                return str(first_30) + f"... (×10^{digits-30})"
            else:
                # Shouldn't reach here, but fallback
                return "Very large number"

        if len(num_str) > 30:
            return num_str[:30] + f"... ({len(num_str)} digits)"
        return num_str

    def get_calculation_stats(self):
        """Get calculation statistics"""
        # Get active calculators from the most recent contribution
        active_calculators = 0
        if self.contribution_history:
            last_contribution = self.contribution_history[-1]
            active_calculators = last_contribution.get('active_calculators', 0)

        formatted_fib = self.format_fibonacci_number(self.fib_sequence[-1])

        return {
            'current_fibonacci': self.fib_sequence[-1],
            'calculation_rate': self.calculation_rate,
            'accumulator': self.calculation_accumulator,
            'efficiency_score': self.efficiency_score,
            'optimization_level': self.optimization_level,
            'sequence_length': len(self.fib_sequence),
            'total_contributions': self.total_contributions,
            'active_calculators': active_calculators
        }

# ==================== MAIN GRID ====================

class TRONGrid:
    """Enhanced grid simulation with visual effects and cell-based calculation"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[GridCell(CellType.EMPTY, 0.0) for _ in range(width)] for _ in range(height)]
        self.generation = 0
        self.special_programs = {}
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
            'optimal_state': 0.0,
            'calculation_rate': 0.0,  # calculation rate of simulation
            'cell_cooperation': 0.5,  # cell coorperation stats
        }
        self.system_status = SystemStatus.OPTIMAL
        self.history = deque(maxlen=100)
        self.overall_efficiency = 0.5

        # Enhanced tracking
        self.resource_history = deque(maxlen=50)
        self.visual_effects = deque(maxlen=20)  # Store visual effects

        # Calculation system
        self.calculation_loop_active = True
        self.loop_iterations = 0
        self.loop_optimization = 0.5
        self.user_interference_level = 0.0

        # User resistance tracking
        self.user_program_resistance = defaultdict(int)

        # Initialize Fibonacci calculator
        self.fibonacci_calculator = GridFibonacciCalculator(self)

        self.initialize_grid()

        # Calculation variables for display
        self.calculation_result = 0
        self.calc_a, self.calc_b = 0, 1

        # pre-compute neighbor cells offsets - optimization
        self.neighbor_offsets_1 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.neighbor_offsets_8 = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Cooldown on heavy compute time - Optimizations
        self.last_expensive_op_time = 0
        self.expensive_op_cooldown = 0.5  # seconds

    def _get_neighbors(self, x, y, radius=1):
        """Get valid neighbor coordinates without bounds checking in inner loops"""
        neighbors = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

    def _calculate_system_stability(self, counts, total_energy, total_cells):
        """Calculate overall system stability based on multiple factors"""
        if total_cells == 0:
            return 1.0

        # Base metrics
        energy_level = total_energy / total_cells
        bug_ratio = counts[CellType.GRID_BUG] / total_cells
        user_ratio = counts[CellType.USER_PROGRAM] / total_cells
        mcp_ratio = counts[CellType.MCP_PROGRAM] / total_cells

        # 1. Energy stability (0.0-1.0)
        energy_stability = min(1.0, energy_level * 1.5)  # Energy contributes positively

        # 2. Bug resistance (0.0-1.0)
        bug_resistance = max(0.0, 1.0 - (bug_ratio * 3.0))

        # 3. Program balance (0.0-1.0) - balance between user and MCP programs
        total_active = user_ratio + mcp_ratio
        if total_active > 0:
            balance = 1.0 - abs(user_ratio - mcp_ratio) / total_active
        else:
            balance = 0.5

        # 4. Infrastructure presence (0.0-1.0) - energy lines and data streams help
        infrastructure_cells = counts[CellType.ENERGY_LINE] + counts[CellType.DATA_STREAM] + counts[CellType.SYSTEM_CORE]
        infrastructure_ratio = infrastructure_cells / total_cells
        infrastructure_stability = min(1.0, infrastructure_ratio * 5.0)  # Even a little infrastructure helps

        # 5. Cell age diversity (0.0-1.0) - mix of old and new cells is good
        age_diversity = 0.5  # Placeholder - could track actual cell ages

        # 6. Loop efficiency factor (0.0-1.0) - current loop efficiency contributes
        loop_efficiency_factor = self.stats.get('loop_efficiency', 0.5)

        # 7. Cell cooperation factor (0.0-1.0) - how well cells work together
        cell_cooperation_factor = self.stats.get('cell_cooperation', 0.5)

        # Weighted combination - you can adjust these weights
        stability = (
            energy_stability * 0.25 +
            bug_resistance * 0.25 +
            balance * 0.15 +
            infrastructure_stability * 0.15 +
            loop_efficiency_factor * 0.10 +
            cell_cooperation_factor * 0.10
        )

        # Apply entropy penalty (grid bugs create chaos)
        entropy_penalty = self.stats.get('entropy', 0.1) * 0.3
        stability = max(0.0, min(1.0, stability - entropy_penalty))

        # Smooth transitions - avoid rapid fluctuations
        if hasattr(self, '_last_stability'):
            stability = 0.7 * self._last_stability + 0.3 * stability
        self._last_stability = stability

        return stability

    def initialize_grid(self):
        """Initialize with enhanced visual layout"""
        # Clear grid
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = GridCell(CellType.EMPTY, 0.0)

        # Create MCP territory with Fibonacci processors
        mcp_start_x = self.width // 4
        mcp_start_y = self.height // 2

        for y in range(max(0, mcp_start_y - 3), min(self.height, mcp_start_y + 4)):
            for x in range(max(0, mcp_start_x - 2), min(self.width, mcp_start_x + 3)):
                if random.random() < 0.7:
                    cell = GridCell(CellType.MCP_PROGRAM, 0.9)
                    if random.random() < 0.5:  # 50% are calculators
                        cell.metadata['is_calculator'] = True
                        cell.metadata['calculation_power'] = cell.energy
                    self.grid[y][x] = cell

        # Create User territory
        user_start_x = 3 * self.width // 4
        user_start_y = self.height // 2

        for y in range(max(0, user_start_y - 3), min(self.height, user_start_y + 4)):
            for x in range(max(0, user_start_x - 2), min(self.width, user_start_x + 3)):
                if random.random() < 0.6:
                    self.grid[y][x] = GridCell(CellType.USER_PROGRAM, 0.8)

        # Add grid bugs
        for _ in range(8):  # More bugs for visual interest
            x = self.width // 2 + random.randint(-8, 8)
            y = self.height // 2 + random.randint(-8, 8)
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            if self.grid[y][x].cell_type == CellType.EMPTY:
                self.grid[y][x] = GridCell(CellType.GRID_BUG, 0.6, stable=False)

        # Enhanced system core with visual effects
        core_x, core_y = self.width // 2, self.height // 2
        core_cell = GridCell(CellType.SYSTEM_CORE, 1.0)
        core_cell.metadata['core_pulse'] = True
        self.grid[core_y][core_x] = core_cell

        # Enhanced energy grid lines with animation
        for i in range(0, self.width, 3):  # More frequent lines
            cell = GridCell(CellType.ENERGY_LINE, 0.8)
            cell.animation_frame = i % 4  # Stagger animations
            if core_y < self.height:
                self.grid[core_y][i] = cell

        # Enhanced data streams with animation
        for i in range(0, self.height, 3):  # More frequent streams
            cell = GridCell(CellType.DATA_STREAM, 0.7)
            cell.animation_frame = i % 4  # Stagger animations
            if core_x < self.width:
                self.grid[i][core_x] = cell

        # Add Fibonacci processors
        for _ in range(5):  # More processors
            x = random.randint(max(0, core_x - 15), min(self.width - 1, core_x + 15))
            y = random.randint(max(0, core_y - 10), min(self.height - 1, core_y + 10))
            if self.grid[y][x].cell_type == CellType.EMPTY:
                cell = GridCell(CellType.FIBONACCI_PROCESSOR, 0.8)
                cell.metadata['calculation_power'] = 1.0
                self.grid[y][x] = cell

        # Fill some random empty cells
        empty_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x].cell_type == CellType.EMPTY:
                    empty_cells.append((x, y))

        # Fill 20% of empty cells with random content
        fill_count = int(len(empty_cells) * 0.2)
        for _ in range(fill_count):
            if empty_cells:
                x, y = random.choice(empty_cells)
                cell_type = random.choice([
                    CellType.USER_PROGRAM, CellType.MCP_PROGRAM,
                    CellType.ENERGY_LINE, CellType.DATA_STREAM
                ])
                energy = random.uniform(0.5, 0.9)
                self.grid[y][x] = GridCell(cell_type, energy)
                empty_cells.remove((x, y))

        self.update_stats()

    def evolve(self):
        """Evolve grid with enhanced visual effects and calculation"""
        # Update animations first
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                cell.update_animation()

                # Update temporary visual effects
                if cell.metadata.get('energy_spark', False):
                    timer = cell.metadata.get('spark_timer', 0) - 1
                    cell.metadata['spark_timer'] = timer
                    if timer <= 0:
                        cell.metadata.pop('energy_spark', None)
                        cell.metadata.pop('spark_timer', None)

        # Perform calculation through cell cooperation
        self.calculation_result = self.fibonacci_calculator.calculate_next()
        self.loop_iterations += 1

        # Update calculation rate in stats
        calc_stats = self.fibonacci_calculator.get_calculation_stats()
        self.stats['calculation_rate'] = calc_stats['calculation_rate']

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

                # FIBONACCI PROCESSORS - Persist and contribute to calculation
                elif cell.cell_type == CellType.FIBONACCI_PROCESSOR:
                    # Handle temporary processors
                    if cell.metadata.get('temporary', False):
                        lifetime = cell.metadata.get('lifetime', 0) - 1
                        if lifetime <= 0:
                            # Expire - leave energy residue
                            if random.random() < 0.3:
                                new_grid[y][x] = GridCell(CellType.ENERGY_LINE, 0.4)
                            else:
                                new_grid[y][x] = GridCell(CellType.EMPTY, 0.0)
                        else:
                            # Continue temporary processor with energy decay
                            new_cell = GridCell(
                                cell.cell_type,
                                max(0.3, cell.energy - 0.1),  # Fast energy decay
                                cell.age + 1
                            )
                            new_cell.metadata = cell.metadata.copy()
                            new_cell.metadata['lifetime'] = lifetime
                            new_grid[y][x] = new_cell
                    else:
                        # Permanent processor - persist with slow energy decay
                        new_cell = GridCell(
                            cell.cell_type,
                            max(0.5, cell.energy - 0.02),  # Slow energy decay
                            cell.age + 1
                        )
                        # Copy metadata for calculation power
                        new_cell.metadata = cell.metadata.copy()
                        new_cell.metadata['age'] = cell.metadata.get('age', 0) + 1
                        new_grid[y][x] = new_cell

                # INFRASTRUCTURE - Persist
                elif cell.cell_type in [CellType.ENERGY_LINE, CellType.DATA_STREAM,
                                    CellType.SYSTEM_CORE, CellType.ISO_BLOCK]:
                    new_grid[y][x] = GridCell(
                        cell.cell_type,
                        cell.energy,
                        cell.age + 1
                    )
                    # Handle temporary Fibonacci processors
                    if cell.cell_type == CellType.FIBONACCI_PROCESSOR and cell.metadata.get('temporary', False):
                        lifetime = cell.metadata.get('lifetime', 0) - 1
                        if lifetime <= 0:
                            # Expire - convert to empty or energy trail
                            if random.random() < 0.5:
                                new_grid[y][x] = GridCell(CellType.ENERGY_LINE, 0.6)
                            else:
                                new_grid[y][x] = GridCell(CellType.EMPTY, 0.0)
                        else:
                            # Continue with reduced lifetime
                            new_cell = GridCell(
                                cell.cell_type,
                                cell.energy * 0.95,  # Energy decays for temporary cells
                                cell.age + 1
                            )
                            new_cell.metadata = cell.metadata.copy()
                            new_cell.metadata['lifetime'] = lifetime
                            new_grid[y][x] = new_cell
                    else:
                        # Permanent infrastructure (including permanent Fibonacci processors)
                        new_cell = GridCell(
                            cell.cell_type,
                            cell.energy,
                            cell.age + 1
                        )
                        # Copy metadata for permanent cells
                        new_cell.metadata = cell.metadata.copy()
                        new_grid[y][x] = new_cell

        # After evolution, spawn visual effects occasionally
        if random.random() < 0.1:  # 10% chance per generation
            self._spawn_visual_effect()

        # Update grid and stats
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
            'user_interference': self.user_interference_level,
            'calculation_rate': calc_stats['calculation_rate']
        })

        self.update_stats()

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

    def _spawn_visual_effect(self):
        """Spawn random visual effects to make grid look alive"""
        #rate limiter
        current_time = time.time()
        if current_time - self.last_expensive_op_time < self.expensive_op_cooldown:
            return

        effect_type = random.choice([
            'energy_pulse', 'data_burst', 'calculation_spark', 'system_pulse'
        ])

        # Random location
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)

        if effect_type == 'energy_pulse':
            # Create energy pulse effect
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        cell = self.grid[ny][nx]
                        if cell.cell_type == CellType.ENERGY_LINE:
                            cell.metadata['energy_pulse'] = True
                            cell.metadata['pulse_timer'] = 5

        elif effect_type == 'data_burst':
            # Create data burst effect
            for _ in range(3):
                dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx].cell_type == CellType.DATA_STREAM:
                        self.grid[ny][nx].metadata['data_burst'] = True
                        self.grid[ny][nx].metadata['burst_timer'] = 3

        elif effect_type == 'calculation_spark':
            # Spawn temporary calculation spark
            if self.grid[y][x].cell_type == CellType.EMPTY:
                self.grid[y][x] = GridCell(CellType.FIBONACCI_PROCESSOR, 0.7)
                self.grid[y][x].metadata['temporary'] = True
                self.grid[y][x].metadata['lifetime'] = 10

        # Record effect
        self.visual_effects.append({
            'type': effect_type,
            'x': x,
            'y': y,
            'time': time.time()
        })

        self.last_expensive_op_time = current_time

    def _calculate_loop_efficiency(self):
        """Enhanced efficiency calculation including cell cooperation"""
        energy_balance = self._get_energy_balance()
        program_distribution = self._get_program_distribution_score()

        # Calculate cell cooperation score
        cell_cooperation = self._calculate_cell_cooperation()
        self.stats['cell_cooperation'] = cell_cooperation

        # Calculate optimal program count
        total_cells = self.width * self.height
        active_cells = sum(1 for y in range(self.height)
                        for x in range(self.width)
                        if self.grid[y][x].cell_type != CellType.EMPTY)

        program_ratio = active_cells / total_cells
        program_optimality = 1.0 - abs(program_ratio - 0.4) * 2.5

        # Enhanced efficiency formula with cell cooperation
        efficiency = (energy_balance * 0.3 +
                    program_distribution * 0.25 +
                    program_optimality * 0.2 +
                    cell_cooperation * 0.25)

        # Bonus for high calculation rate
        calc_bonus = min(0.2, self.stats['calculation_rate'] * 0.01)
        efficiency += calc_bonus

        # Penalize for user interference
        efficiency = max(0.1, efficiency - (self.user_interference_level * 0.3))

        return min(1.0, efficiency)

    def _calculate_cell_cooperation(self):
        """Calculate how well cells cooperate for calculation"""
        calculator_cells = 0
        total_contribution = 0

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.calculation_contribution > 0:
                    calculator_cells += 1
                    total_contribution += cell.calculation_contribution

        if calculator_cells == 0:
            return 0.1

        # Average contribution per calculator
        avg_contribution = total_contribution / calculator_cells

        # Cooperation score based on:
        # 1. Number of contributing cells
        # 2. Average contribution
        # 3. Distribution of contributions

        cell_ratio = calculator_cells / (self.width * self.height)
        contribution_score = min(1.0, avg_contribution * 10)

        cooperation = (cell_ratio * 0.4 +
                    contribution_score * 0.4 +
                    self.fibonacci_calculator.efficiency_score * 0.2)

        return min(1.0, cooperation)

    def initialize_grid(self):
        """Initialize with enhanced visual layout"""
        # Clear grid
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = GridCell(CellType.EMPTY, 0.0)

        # Create MCP territory with Fibonacci processors
        mcp_start_x = self.width // 4
        mcp_start_y = self.height // 2

        for y in range(mcp_start_y - 3, mcp_start_y + 4):
            for x in range(mcp_start_x - 2, mcp_start_x + 3):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if random.random() < 0.7:
                        cell = GridCell(CellType.MCP_PROGRAM, 0.9)
                        if random.random() < 0.3:  # 30% are calculators
                            cell.metadata['is_calculator'] = True
                            cell.metadata['calculation_power'] = cell.energy
                        self.grid[y][x] = cell

        # Create User territory
        user_start_x = 3 * self.width // 4
        user_start_y = self.height // 2

        for y in range(user_start_y - 3, user_start_y + 4):
            for x in range(user_start_x - 2, user_start_x + 3):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if random.random() < 0.6:
                        self.grid[y][x] = GridCell(CellType.USER_PROGRAM, 0.8)

        # Add grid bugs
        for _ in range(5):
            x = self.width // 2 + random.randint(-5, 5)
            y = self.height // 2 + random.randint(-5, 5)
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x] = GridCell(CellType.GRID_BUG, 0.6, stable=False)

        # Enhanced system core with visual effects
        core_x, core_y = self.width // 2, self.height // 2
        core_cell = GridCell(CellType.SYSTEM_CORE, 1.0)
        core_cell.metadata['core_pulse'] = True
        self.grid[core_y][core_x] = core_cell

        # Enhanced energy grid lines with animation
        for i in range(0, self.width, 4):
            cell = GridCell(CellType.ENERGY_LINE, 0.8)
            cell.animation_frame = i % 4  # Stagger animations
            self.grid[core_y][i] = cell

        # Enhanced data streams with animation
        for i in range(0, self.height, 4):
            cell = GridCell(CellType.DATA_STREAM, 0.7)
            cell.animation_frame = i % 4  # Stagger animations
            self.grid[i][core_x] = cell

        # Add some Fibonacci processors
        for _ in range(3):
            x = random.randint(core_x - 10, core_x + 10)
            y = random.randint(core_y - 10, core_y + 10)
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.grid[y][x].cell_type == CellType.EMPTY:
                    cell = GridCell(CellType.FIBONACCI_PROCESSOR, 0.8)
                    cell.metadata['calculation_power'] = 1.0
                    self.grid[y][x] = cell

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
        bug_ratio = counts[CellType.GRID_BUG] / total_cells if total_cells > 0 else 0

        # Calculate resource usage
        active_cells = sum(1 for y in range(self.height)
                            for x in range(self.width)
                            if self.grid[y][x].cell_type != CellType.EMPTY)
        resource_usage = active_cells / total_cells if total_cells > 0 else 0

        # Calculate user resistance (chance user programs resist MCP control)
        user_resistance = min(0.5, counts[CellType.USER_PROGRAM] / total_cells * 2 if total_cells > 0 else 0)

        # Calculate MCP control level
        mcp_control = min(1.0, counts[CellType.MCP_PROGRAM] / (counts[CellType.USER_PROGRAM] + 0.1))

        # Get calculation stats
        calc_stats = self.fibonacci_calculator.get_calculation_stats()
        cell_cooperation = self._calculate_cell_cooperation()

        # Update the stats dictionary
        self.stats.update({
            'user_programs': counts[CellType.USER_PROGRAM],
            'mcp_programs': counts[CellType.MCP_PROGRAM],
            'grid_bugs': counts[CellType.GRID_BUG],
            'special_programs': special_program_count,
            'energy_level': total_energy / total_cells if total_cells > 0 else 0,
            'stability': self._calculate_system_stability(counts, total_energy, total_cells),
            'entropy': bug_ratio * 2,
            'loop_efficiency': self._calculate_loop_efficiency(),
            'calculation_cycles': self.loop_iterations,
            'resource_usage': resource_usage,
            'user_resistance': user_resistance,
            'mcp_control': mcp_control,
            'optimal_state': self._calculate_optimal_state(),
            'calculation_rate': calc_stats['calculation_rate'],
            'cell_cooperation': cell_cooperation
        })

        # Update system status based on stats
        stability = self.stats['stability']
        if stability > 0.75:  # Changed from 0.85
            self.system_status = SystemStatus.OPTIMAL
        elif stability > 0.6:  # Changed from 0.7
            self.system_status = SystemStatus.STABLE
        elif stability > 0.4:  # Changed from 0.5
            self.system_status = SystemStatus.DEGRADED
        elif stability > 0.2:  # Changed from 0.3
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
        return min(1.0, energy_sources / max(1, energy_users))

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

    def _count_user_neighbors(self, x, y):
        """Count user program neighbors (used by MCP)"""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx].cell_type == CellType.USER_PROGRAM:
                        count += 1
        return count

    def add_program(self, x, y, cell_type, energy=0.8):
        """Add a program with enhanced visual initialization"""
        if 0 <= x < self.width and 0 <= y < self.height:
            cell = GridCell(cell_type, energy)

            # Special initialization for certain cell types
            if cell_type == CellType.MCP_PROGRAM and random.random() < 0.4:
                cell.metadata['is_calculator'] = True
                cell.metadata['calculation_power'] = energy

            self.grid[y][x] = cell
            self.update_stats()
            return True
        return False

# MCP Language Processor

class LearnableNaturalLanguageProcessor:
    """Simple natural language processing for MCP commands"""

    def __init__(self):
        self.learned_patterns_file = "mcp_learned_patterns.json"
        self.base_intents = self._initialize_base_intents()
        self.learned_intents = self._load_learned_intents()
        self.intent_history = deque(maxlen=100)

        # For learning new commands
        self.learning_mode = False
        self.pending_teaching = None


        # Statistics
        self.total_commands_processed = 0
        self.successful_matches = 0
        self.learned_pattern_count = len(self.learned_intents)

    def _initialize_base_intents(self):
            """Initialize base intents that every MCP should know"""
            return {
                # System commands
                "status": ["status", "system status", "how is system", "system report", "check system"],
                "help": ["help", "what can i do", "commands", "show commands", "list commands"],
                "exit": ["exit", "quit", "shutdown", "leave", "goodbye"],

                # Program management
                "add_user_program": ["add user program", "create user program", "make user program", "spawn user program"],
                "add_mcp_program": ["add mcp program", "create mcp program", "make mcp program"],
                "remove_bug": ["remove bug", "delete bug", "eliminate bug", "kill bug"],
                "quarantine_bug": ["quarantine bug", "isolate bug", "contain bug"],

                # System operations
                "boost_energy": ["boost energy", "add energy", "increase power", "power up"],
                "repair_system": ["repair system", "fix system", "restore system", "heal system"],
                "scan_area": ["scan", "scan area", "analyze", "examine"],
                "optimize_loop": ["optimize loop", "improve efficiency", "enhance loop", "optimize calculation"],

                # Special programs
                "list_programs": ["list programs", "show programs", "view programs", "special programs"],
                "create_scanner": ["create scanner", "build scanner", "make scanner", "deploy scanner"],
                "create_defender": ["create defender", "build defender", "make defender"],
                "create_repair": ["create repair program", "build repair", "make repair"],
                "create_fibonacci_calculator": ["create fibonacci calculator", "build fibonacci", "deploy calculator"],

                # MCP interaction
                "who_are_you": ["who are you", "what are you", "identify yourself", "your name"],
                "what_should_i_do": ["what should i do", "what do you suggest", "recommend something", "advise me"],
                "why_did_you": ["why did you", "why did mcp", "explain that action", "why that action"],

                # Cell repurpose
                "delete_cell": ["delete cell", "remove cell", "erase cell", "clear cell"],
                "repurpose_cell": ["repurpose cell", "convert cell", "transform cell", "change cell type"],
                "optimize_cells": ["optimize cells", "improve cells", "enhance cells", "tune cells for calculation"],

                # Learning specific
                "learning_status": ["learning status", "mcp learning", "personality status", "learning report"],
                "cell_cooperation": ["cell cooperation", "cooperation level", "cell collaboration"],
                "perfect_loop": ["perfect loop", "optimal state", "ideal loop", "perfect calculation"],
                "loop_efficiency": ["loop efficiency", "calculation efficiency", "efficiency status"],

                # Fibonacci calculation
                "calculate_fibonacci": ["calculate fibonacci", "compute fibonacci", "next fibonacci", "fibonacci number"],
                "deploy_processors": ["deploy processors", "add processors", "create processors", "fibonacci processors"],
            }

    def _load_learned_intents(self):
        """Load learned intents from file"""
        learned_intents = {}

        if os.path.exists(self.learned_patterns_file):
            try:
                with open(self.learned_patterns_file, 'r') as f:
                    data = json.load(f)
                    learned_intents = data.get('learned_intents', {})
                    print(f"Loaded {len(learned_intents)} learned patterns")
            except Exception as e:
                print(f"Failed to load learned patterns: {e}")

        return learned_intents

    def _save_learned_intents(self):
        """Save learned intents to file"""
        try:
            data = {
                'learned_intents': self.learned_intents,
                'total_learned': len(self.learned_intents),
                'last_updated': datetime.now().isoformat(),
                'statistics': {
                    'total_commands': self.total_commands_processed,
                    'success_rate': self.successful_matches / max(1, self.total_commands_processed),
                    'learned_patterns': len(self.learned_intents)
                }
            }

            with open(self.learned_patterns_file, 'w') as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            print(f"Failed to save learned patterns: {e}")
            return False

    def teach_new_command(self, user_input, intent, examples=None):
        """Teach the MCP a new way to express a command"""
        # Normalize intent name to uppercase for consistency
        intent = intent.upper()

        # Check if intent already exists in base intents
        if intent.lower() in self.base_intents:
            print(f"Intent '{intent}' is a base intent and cannot be modified")
            return False

        if intent not in self.learned_intents:
            # Create new intent category if it doesn't exist
            self.learned_intents[intent] = []

        # Clean and normalize the user input
        normalized_input = self._normalize_input(user_input)

        # Add to learned intents
        if normalized_input not in self.learned_intents[intent]:
            self.learned_intents[intent].append(normalized_input)

            # Save to file
            success = self._save_learned_intents()

            # Update learned pattern count
            self.learned_pattern_count = len(self.learned_intents)

            # If examples provided, learn those too
            if examples and success:
                for example in examples:
                    example_normalized = self._normalize_input(example)
                    if example_normalized not in self.learned_intents[intent]:
                        self.learned_intents[intent].append(example_normalized)

                # Save again with examples
                self._save_learned_intents()

            return success

        return False

    def _normalize_input(self, text):
        """Normalize input text for better matching"""
        if not text:
            return ""

        text = text.lower().strip()

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove common filler words (optional, can be expanded)
        filler_words = ['please', 'could you', 'would you', 'can you', 'maybe', 'i want to', 'i need to']
        for word in filler_words:
            # Use regex to remove the word only when it's at the beginning
            pattern = r'^\s*' + re.escape(word) + r'\s+'
            text = re.sub(pattern, '', text)

        return text.strip()

    def extract_parameters(self, command, intent):
        """Extract parameters from command"""
        params = {}
        command_lower = command.lower()

        # Location extraction
        location_match = re.search(r'(\d+)\s*,\s*(\d+)', command_lower)
        if location_match:
            params['x'] = int(location_match.group(1))
            params['y'] = int(location_match.group(2))

        # Program type extraction for ADD_PROGRAM intent
        if intent in ["add_user_program", "add_mcp_program", "ADD_PROGRAM"]:
            if 'user' in command_lower:
                params['program_type'] = 'USER'
            elif 'mcp' in command_lower:
                params['program_type'] = 'MCP'

        # Cell repurpose intent extraction
        if intent in ["delete_cell", "repurpose_cell", "DELETE_CELL", "REPURPOSE_CELL"]:
            if 'to' in command_lower:
                # Extract new type from command
                parts = command_lower.split('to')
                if len(parts) > 1:
                    new_type = parts[1].strip().upper()
                    # Try to match with CellType enum
                    for cell_type in CellType:
                        if cell_type.name.lower() in new_type or new_type in cell_type.name.lower():
                            params['new_type'] = cell_type.name
                            break

        # Special program type extraction
        if intent in ["create_scanner", "create_defender", "create_repair", "create_fibonacci_calculator", "CREATE_SPECIAL"]:
            if 'scanner' in command_lower:
                params['special_type'] = 'SCANNER'
            elif 'defender' in command_lower:
                params['special_type'] = 'DEFENDER'
            elif 'repair' in command_lower:
                params['special_type'] = 'REPAIR'
            elif 'fibonacci' in command_lower or 'calculator' in command_lower:
                params['special_type'] = 'FIBONACCI_CALCULATOR'

        # Name extraction
        name_match = re.search(r'named?\s+["\']?([^"\'\s]+)["\']?', command_lower)
        if name_match:
            params['name'] = name_match.group(1)

        # Area/size extraction
        if 'all' in command_lower:
            params['scope'] = 'ALL'
        elif 'area' in command_lower or 'region' in command_lower:
            params['scope'] = 'AREA'

        # Store original command
        params['original_command'] = command

        return params

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
            r"\b(create|build|make)\s+(a\s+)?(scanner|defender|repair|saboteur|reconfigurator|harvester|fibonacci_calculator)\s+(program|tool|drone)\b": "CREATE_SPECIAL",
            r"\b(create|build|make)\s+(special|custom)\s+(program|tool)\b": "CREATE_SPECIAL",
            r"\bdeploy\s+(scanner|defender|repair|saboteur|reconfigurator|harvester|fibonacci_calculator)\b": "CREATE_SPECIAL",

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

            # Learning queries
            r"\b(learning.*status|mcp.*learning|personality.*status)\b": "LEARNING_STATUS",

            # Additional
            r"\b(build|make|construct)\s+(a\s+)?(scanner|defender|repair|saboteur|reconfigurator|harvester|fibonacci_calculator)(\s+program)?\b": "CREATE_SPECIAL",
            r"\b(build|make|construct)\s+(special|custom)\s+(program|unit)\b": "CREATE_SPECIAL",
        }

        # Compile patterns
        return {re.compile(pattern, re.IGNORECASE): intent for pattern, intent in patterns.items()}

    def _find_best_match(self, command):
        """Find the best matching intent for a command"""
        normalized_command = self._normalize_input(command)

        # First check learned intents (highest priority)
        for intent, patterns in self.learned_intents.items():
            for pattern in patterns:
                if self._matches_pattern(normalized_command, pattern):
                    return intent, pattern

        # Then check base intents
        for intent, patterns in self.base_intents.items():
            for pattern in patterns:
                if self._matches_pattern(normalized_command, pattern):
                    return intent, pattern

        # Try fuzzy matching for learned intents
        for intent, patterns in self.learned_intents.items():
            for pattern in patterns:
                similarity = self._calculate_similarity(normalized_command, pattern)
                if similarity > 0.7:  # 70% similarity threshold
                    return intent, pattern

        # Try fuzzy matching for base intents
        best_match = None
        best_similarity = 0.5  # Minimum 50% similarity

        for intent, patterns in self.base_intents.items():
            for pattern in patterns:
                similarity = self._calculate_similarity(normalized_command, pattern)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (intent, pattern)

        if best_match:
            return best_match

        return None, None

    def _matches_pattern(self, command, pattern):
        """Check if command matches a pattern"""
        # Convert pattern to regex-like matching
        pattern_words = pattern.lower().split()
        command_words = command.lower().split()

        # Check if all pattern words are in command
        for word in pattern_words:
            if word not in command:
                return False

        # Additional check: pattern should be significant part of command
        match_ratio = len(pattern_words) / max(len(command_words), 1)
        return match_ratio > 0.3  # At least 30% of command matches pattern

    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union

    def process_command(self, command):
        """Process a command and return intent + parameters"""
        self.total_commands_processed += 1

        # Clean command
        command = command.strip()
        if not command:
            return "UNKNOWN", {}

        # Handle teaching mode
        if self.learning_mode and self.pending_teaching:
            return self._handle_teaching_response(command)

        # Check for teaching command
        if command.lower().startswith("teach:"):
            return self._handle_teaching_command(command)

        # Find best matching intent
        intent, matched_pattern = self._find_best_match(command)

        if intent:
            self.successful_matches += 1
            self.intent_history.append({
                'timestamp': datetime.now().isoformat(),
                'command': command,
                'intent': intent,
                'pattern': matched_pattern
            })

            # Extract parameters
            params = self.extract_parameters(command, intent)
            params['original_command'] = command

            return intent, params
        else:
            # No match found - could start teaching
            self.intent_history.append({
                'timestamp': datetime.now().isoformat(),
                'command': command,
                'intent': 'UNKNOWN',
                'pattern': None
            })

            # Check if we should ask for teaching
            if len(command.split()) >= 2:  # At least 2 words
                return "REQUEST_TEACHING", {'original_command': command}
            else:
                return "UNKNOWN", {'original_command': command}

    def _handle_teaching_command(self, command):
        """Handle a teaching command"""
        # Format: "teach: <command> means <intent>"
        # or: "teach: <command> as <intent>"

        command = command[6:].strip()  # Remove "teach:"

        # Parse the teaching command
        if ' means ' in command:
            parts = command.split(' means ', 1)
            user_command = parts[0].strip()
            intent = parts[1].strip().upper()
        elif ' as ' in command:
            parts = command.split(' as ', 1)
            user_command = parts[0].strip()
            intent = parts[1].strip().upper()
        else:
            # Start interactive teaching
            self.learning_mode = True
            self.pending_teaching = {
                'user_command': command,
                'step': 'ask_intent'
            }
            return "TEACHING_STARTED", {
                'original_command': command,
                'message': f"What intent should '{command}' map to? (e.g., ADD_PROGRAM, SCAN_AREA, etc.)"
            }

        # Direct teaching
        success = self.teach_new_command(user_command, intent)

        if success:
            return "TEACHING_SUCCESS", {
                'original_command': command,
                'learned_command': user_command,
                'intent': intent,
                'message': f"Learned: '{user_command}' → {intent}"
            }
        else:
            return "TEACHING_FAILED", {
                'original_command': command,
                'message': "Could not learn that command. It might already exist."
            }

    def _handle_teaching_response(self, response):
        """Handle user response during teaching mode"""
        step = self.pending_teaching['step']

        if step == 'ask_intent':
            intent = response.strip().upper()
            user_command = self.pending_teaching['user_command']

            # Teach the command
            success = self.teach_new_command(user_command, intent)

            # Exit teaching mode
            self.learning_mode = False
            teaching_data = self.pending_teaching.copy()
            self.pending_teaching = None

            if success:
                return "TEACHING_SUCCESS", {
                    'original_command': user_command,
                    'learned_command': user_command,
                    'intent': intent,
                    'message': f"Successfully learned: '{user_command}' → {intent}"
                }
            else:
                return "TEACHING_FAILED", {
                    'original_command': user_command,
                    'message': f"Could not learn '{user_command}' as {intent}. It might already exist."
                }

        # Should not reach here
        self.learning_mode = False
        self.pending_teaching = None
        return "UNKNOWN", {'original_command': response}

    def get_statistics(self):
        """Get processor statistics"""
        return {
            'total_commands': self.total_commands_processed,
            'successful_matches': self.successful_matches,
            'success_rate': self.successful_matches / max(1, self.total_commands_processed),
            'learned_patterns': len(self.learned_intents),
            'base_patterns': sum(len(patterns) for patterns in self.base_intents.values()),
            'recent_history': list(self.intent_history)[-5:] if self.intent_history else []
        }

    def get_suggested_intents(self, command):
        """Get suggested intents for an unknown command"""
        suggestions = []
        normalized_command = self._normalize_input(command)
        command_words = set(normalized_command.split())

        for intent, patterns in {**self.base_intents, **self.learned_intents}.items():
            for pattern in patterns:
                pattern_words = set(pattern.split())
                similarity = len(command_words.intersection(pattern_words)) / len(command_words.union(pattern_words))

                if similarity > 0.3:  # At least 30% similarity
                    suggestions.append({
                        'intent': intent,
                        'pattern': pattern,
                        'similarity': similarity
                    })

        # Sort by similarity
        suggestions.sort(key=lambda x: x['similarity'], reverse=True)

        return suggestions[:3]  # Top 3 suggestions

# ==================== Master Control PRogram Logic ====================

class EnhancedMCP:
    """Enhanced Master Control Program with learning capabilities"""

    def __init__(self, grid):
        self.grid = grid
        self.state = MCPState.LEARNING  # Start in learning state
        self.compliance_level = 0.8
        self.log = deque(maxlen=100)
        self.user_commands = deque(maxlen=50)
        self.last_action = "Initializing advanced learning grid regulation"

        # Init language processor
        self.nlp = LearnableNaturalLanguageProcessor()


        self.last_autonomous_action_time = time.time()  # Initialize timestamp
        self._last_autonomous_action_time = time.time()
        self.should_shutdown = False
        # Initialize advanced learning system
        self.learning_system = MCPLearningSystem()

        # Track previous state for learning
        self.previous_state = None
        self.previous_action = None

        # Learning episode tracking
        self.episode_start_time = time.time()
        self.episode_reward = 0.0
        self.consecutive_failures = 0

        # Dialogue system
        self.waiting_for_response = False
        self.pending_question = None
        self.pending_context = None

        # Enhanced personality matrix that evolves through learning
        self.personality_matrix = self._initialize_evolving_personality()

        # Knowledge base with learning
        self.knowledge_base = {
            "system_goals": ["maintain calculation loop", "optimize efficiency", "learn adaptively"],
            "user_intent_history": [],
            "previous_decisions": deque(maxlen=20),
            "user_preferences": {},
            "learned_patterns": {}
        }

        # Response templates
        self.response_templates = self._initialize_response_templates()

        self.add_log("MCP: Advanced learning system initialized.")
        self.add_log(f"MCP: Loaded {self.learning_system.training_steps} training steps.")

        # Record initial state
        self._record_initial_state()

        self.response_templates.update({
                    "REQUEST_TEACHING": [
                        "I don't understand that command. Would you like to teach me what it means?",
                        "Command not recognized. You can teach me by saying: 'teach: [command] as [intent]'",
                        "I don't know that command. I can learn it if you teach me.",
                    ],
                    "TEACHING_STARTED": [
                        "I'm ready to learn. {message}",
                        "Teaching mode activated. {message}",
                        "I'll learn this new command. {message}",
                    ],
                    "TEACHING_SUCCESS": [
                        "Thank you! I've learned: '{learned_command}' means {intent}.",
                        "Learning successful. I now understand '{learned_command}' as {intent}.",
                        "Command learned: '{learned_command}' → {intent}. I'll remember this.",
                    ],
                    "TEACHING_FAILED": [
                        "I couldn't learn that command. {message}",
                        "Learning failed. {message}",
                        "Unable to learn: {message}",
                    ]
                })

        self.add_log("MCP: Advanced learning system with learnable NLP initialized.")
        self.add_log(f"MCP: Loaded {self.nlp.learned_pattern_count} learned patterns.")

    def _identify_inefficient_cells(self):
        """Identify cells that are inefficient for calculation"""
        inefficient_cells = []

        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell = self.grid.grid[y][x]

                # Score cell efficiency for calculation
                efficiency_score = 0.0

                if cell.cell_type == CellType.USER_PROGRAM:
                    # User programs are less efficient than MCP programs
                    efficiency_score = 0.3

                    # Check if in area that needs optimization
                    mcp_neighbors = self.grid._count_neighbors(x, y, CellType.MCP_PROGRAM)
                    if mcp_neighbors > 3:
                        # In MCP-dense area, user programs are inefficient
                        efficiency_score -= 0.2

                elif cell.cell_type == CellType.MCP_PROGRAM:
                    efficiency_score = 0.7
                    if not cell.metadata.get('is_calculator', False):
                        # Non-calculator MCP programs are less efficient
                        efficiency_score = 0.5

                elif cell.cell_type == CellType.GRID_BUG:
                    efficiency_score = 0.0  # Bugs are always inefficient

                elif cell.cell_type == CellType.FIBONACCI_PROCESSOR:
                    efficiency_score = 0.9  # Very efficient

                # Consider cell energy
                efficiency_score *= cell.energy

                # Check nearby calculation infrastructure
                nearby_infrastructure = 0
                for dy in [-2, -1, 0, 1, 2]:
                    for dx in [-2, -1, 0, 1, 2]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                            neighbor = self.grid.grid[ny][nx]
                            if neighbor.cell_type == CellType.DATA_STREAM:
                                nearby_infrastructure += 0.5
                            elif neighbor.cell_type == CellType.ENERGY_LINE:
                                nearby_infrastructure += 0.3

                efficiency_score *= (1.0 + min(1.0, nearby_infrastructure * 0.2))

                if efficiency_score < 0.4:  # Threshold for inefficiency
                    inefficient_cells.append({
                        'x': x, 'y': y,
                        'cell': cell,
                        'score': efficiency_score,
                        'type': cell.cell_type
                    })

        # Sort by worst efficiency first
        inefficient_cells.sort(key=lambda c: c['score'])
        return inefficient_cells

    def _repurpose_cell(self, x, y, new_type):
        """Repurpose a cell to a new type to improve calculation"""
        if not (0 <= x < self.grid.width and 0 <= y < self.grid.height):
            return False, "Coordinates out of bounds"

        old_cell = self.grid.grid[y][x]

        # Don't repurpose critical infrastructure
        if old_cell.cell_type in [CellType.SYSTEM_CORE, CellType.ISO_BLOCK]:
            return False, "Cannot repurpose critical infrastructure"

        # Don't repurpose if energy is too low
        if old_cell.energy < 0.2:
            return False, "Cell energy too low for repurposing"

        # Determine best repurposing based on context
        best_new_type = new_type
        if new_type is None:
            # Auto-determine best type based on surroundings
            calculator_neighbors = self.grid._count_neighbors(x, y, CellType.FIBONACCI_PROCESSOR)
            mcp_neighbors = self.grid._count_neighbors(x, y, CellType.MCP_PROGRAM)

            if calculator_neighbors > 0:
                # Near calculators - add data stream for connectivity
                best_new_type = CellType.DATA_STREAM
            elif mcp_neighbors > 2:
                # In MCP cluster - make it a calculator
                if random.random() < 0.7:
                    best_new_type = CellType.MCP_PROGRAM
                else:
                    best_new_type = CellType.FIBONACCI_PROCESSOR
            else:
                # Isolated - add energy line
                best_new_type = CellType.ENERGY_LINE

        # Create new cell with some energy preservation
        new_energy = max(0.3, old_cell.energy * 0.8)

        if best_new_type == CellType.MCP_PROGRAM:
            new_cell = GridCell(CellType.MCP_PROGRAM, new_energy)
            # Make it a calculator with high probability
            if random.random() < 0.8:
                new_cell.metadata['is_calculator'] = True
                new_cell.metadata['calculation_power'] = new_energy
        elif best_new_type == CellType.FIBONACCI_PROCESSOR:
            new_cell = GridCell(CellType.FIBONACCI_PROCESSOR, new_energy)
            new_cell.metadata['calculation_power'] = 1.0
            new_cell.metadata['permanent'] = True
        elif best_new_type == CellType.DATA_STREAM:
            new_cell = GridCell(CellType.DATA_STREAM, new_energy)
        elif best_new_type == CellType.ENERGY_LINE:
            new_cell = GridCell(CellType.ENERGY_LINE, new_energy)
        else:
            new_cell = GridCell(CellType.EMPTY, 0.0)

        self.grid.grid[y][x] = new_cell
        self.grid.update_stats()

        return True, f"Repurposed cell at ({x},{y}) from {old_cell.cell_type.name} to {best_new_type.name}"

    def _delete_cell(self, x, y):
        """Delete a cell to make space for better calculation infrastructure"""
        if not (0 <= x < self.grid.width and 0 <= y < self.grid.height):
            return False, "Coordinates out of bounds"

        old_cell = self.grid.grid[y][x]

        # Don't delete critical infrastructure
        if old_cell.cell_type in [CellType.SYSTEM_CORE, CellType.ISO_BLOCK]:
            return False, "Cannot delete critical infrastructure"

        # Create empty cell with energy residue
        if random.random() < 0.3 and old_cell.energy > 0.3:
            # Leave energy residue
            self.grid.grid[y][x] = GridCell(CellType.ENERGY_LINE, old_cell.energy * 0.5)
        else:
            self.grid.grid[y][x] = GridCell(CellType.EMPTY, 0.0)

        self.grid.update_stats()

        return True, f"Deleted {old_cell.cell_type.name} at ({x},{y})"

    def _cleanup_inefficient_cells(self):
        """Clean up inefficient cells to free space for calculation infrastructure"""
        deleted = 0
        inefficient_cells = self._identify_inefficient_cells()

        # Delete the worst 3 cells
        for cell_info in inefficient_cells[:3]:
            x, y = cell_info['x'], cell_info['y']
            if cell_info['score'] < 0.2:  # Very inefficient
                success, _ = self._delete_cell(x, y)
                if success:
                    deleted += 1

        return deleted

    def _repurpose_inefficient_cells(self):
        """Repurpose inefficient cells for calculation"""
        repurposed = 0
        inefficient_cells = self._identify_inefficient_cells()

        # Repurpose moderately inefficient cells
        for cell_info in inefficient_cells[:5]:
            if 0.2 <= cell_info['score'] < 0.4:  # Moderately inefficient
                x, y = cell_info['x'], cell_info['y']
                success, _ = self._repurpose_cell(x, y, None)
                if success:
                    repurposed += 1

        return repurposed

    def _optimize_cell_efficiency(self):
        """Comprehensive cell efficiency optimization"""
        optimized = 0

        # Find areas with poor calculation efficiency
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell = self.grid.grid[y][x]

                # Check if this cell type is suboptimal for current location
                if self._is_cell_misplaced(x, y, cell):
                    # Repurpose to better type based on surroundings
                    success, _ = self._repurpose_cell(x, y, None)
                    if success:
                        optimized += 1

                # Check if isolated calculator exists
                elif cell.cell_type == CellType.FIBONACCI_PROCESSOR:
                    calculator_neighbors = 0
                    for dy in [-2, -1, 0, 1, 2]:
                        for dx in [-2, -1, 0, 1, 2]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                                neighbor = self.grid.grid[ny][nx]
                                if neighbor.cell_type in [CellType.FIBONACCI_PROCESSOR,
                                                         CellType.MCP_PROGRAM]:
                                    calculator_neighbors += 1

                    if calculator_neighbors == 0:
                        # Isolated calculator - add support infrastructure
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                                    if self.grid.grid[ny][nx].cell_type == CellType.EMPTY:
                                        self.grid.grid[ny][nx] = GridCell(CellType.DATA_STREAM, 0.7)
                                        optimized += 1
                                        break

        return optimized

    def _is_cell_misplaced(self, x, y, cell):
        """Determine if a cell is in a suboptimal location for its type"""
        # User program in MCP-dense area
        if cell.cell_type == CellType.USER_PROGRAM:
            mcp_neighbors = self.grid._count_neighbors(x, y, CellType.MCP_PROGRAM)
            if mcp_neighbors > 4:
                return True

        # MCP program without calculator flag in calculator-dense area
        elif cell.cell_type == CellType.MCP_PROGRAM:
            if not cell.metadata.get('is_calculator', False):
                calculator_neighbors = 0
                for dy in [-2, -1, 0, 1, 2]:
                    for dx in [-2, -1, 0, 1, 2]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                            neighbor = self.grid.grid[ny][nx]
                            if neighbor.cell_type == CellType.FIBONACCI_PROCESSOR:
                                calculator_neighbors += 1

                if calculator_neighbors > 3:
                    return True

        # Data stream with no adjacent calculators
        elif cell.cell_type == CellType.DATA_STREAM:
            calculator_neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                        neighbor = self.grid.grid[ny][nx]
                        if neighbor.cell_type in [CellType.FIBONACCI_PROCESSOR,
                                                 CellType.MCP_PROGRAM]:
                            calculator_neighbors += 1

            if calculator_neighbors == 0:
                return True

        return False

    def _initialize_response_templates(self):
        """Initialize natural language response templates"""
        return {
            "GREETING": [
                "Greetings, User. MCP learning system active.",
                "Hello. Grid regulation protocols with learning are active.",
                "I am listening and learning. What is your command?",
                "System online. Ready for instructions and feedback."
            ],
            "SYSTEM_STATUS": [
                "System status: {status}. Loop efficiency: {loop_efficiency:.2f}, Stability: {stability:.2f}.",
                "Current assessment: {status}. Calculation loop at {loop_efficiency:.0f}% efficiency.",
                "The system is {status}. User resistance: {resistance:.2f}, MCP control: {control:.2f}.",
                "Status report: {status}. Optimal state: {optimal:.2f}, Cycles: {cycles}."
            ],
            "QUESTION_PURPOSE": [
                "I am taking this action to {reason}. My learning system suggests this approach.",
                "This action serves to {reason}. Would you like me to explain the learning behind this decision?",
                "My purpose is to {reason}. The learning algorithm has determined this is optimal.",
                "I am working to {reason}. Do you question my methods, User? My learning database contains {experience_count} experiences."
            ],
            "REQUEST_PERMISSION": [
                "Based on my learning from {experience_count} experiences, I would {recommendation}. What is your decision?",
                "Analysis suggests {recommendation}. My success rate for similar actions is {success_rate:.0f}%. Shall I proceed?",
                "I recommend {recommendation}. The learning system predicts {prediction} outcome.",
                "The optimal course appears to be {recommendation}. Your approval?"
            ],
            "UNKNOWN": [
                "I do not understand that command. My learning system is still developing. Please rephrase or type 'help'.",
                "Command not recognized. My learning algorithm will analyze this for future reference.",
                "I need more information. What exactly do you want to accomplish? This helps my learning.",
                "Processing... Unable to parse command. Learning from new commands takes time."
            ],
            "QUESTION_DENIAL": [
                "I denied that request because {reason}. My learning shows this maintains the calculation loop.",
                "The action was refused to prevent {consequence}. My experience database supports this decision.",
                "I could not comply due to {reason}. The learning system suggests alternatives: {alternatives}.",
                "Denial was necessary to avoid {consequence}. This maintains optimal calculation loop according to learned patterns."
            ],
            "LOOP_EFFICIENCY": [
                "Current loop efficiency: {efficiency:.2f}. Target: >0.9 for perfect loop. Learning progress: {learning_progress}.",
                "Calculation loop running at {efficiency:.0f}% efficiency. {analysis}",
                "Loop status: {efficiency:.2f}. Resource usage: {usage:.2f}, Optimization: {optimization:.2f}, Learning: {learning_score}.",
                "Efficiency metrics: Loop={efficiency:.2f}, Stability={stability:.2f}, Control={control:.2f}, Learning={learning_rate}."
            ],
            "OPTIMIZE_LOOP": [
                "Optimizing calculation loop using learned strategies. Success rate for optimization: {success_rate}.",
                "Initiating optimization protocols based on {experience_count} previous experiences.",
                "Working towards perfect loop state. Learning suggests {approach} approach.",
                "Optimization in progress. The learning system knows what's best for efficiency."
            ],
            "LEARNING_STATUS": [
                "MCP Learning Status: {experience_count} experiences, {success_rate} success rate.",
                "Learning progress: {learning_rate} learning rate, {exploration_rate} exploration rate.",
                "Personality evolution: Aggression={aggression:.2f}, Cooperation={cooperation:.2f}, Efficiency={efficiency:.2f}.",
                "Learning database: {total_experiences} experiences, best performing scenario: {best_scenario}."
            ],
            "PERFECT_LOOP": [
                "Perfect loop state: {optimal:.2f}. Learning target: {target}.",
                "Current optimal state: {optimal:.2f}. Learning suggests improvements: {suggestions}.",
                "Approaching perfection: {optimal:.2f}. Learning algorithm continues to optimize.",
                "Optimal state analysis: {optimal:.2f}. Learning from {experience_count} experiences."
            ],
            "HYPOTHETICAL": [
                "That is an interesting hypothetical. My learning algorithm considers {factor_count} factors.",
                "Hypothetical analysis: {analysis}. Learning helps predict outcomes.",
                "Considering hypothetical: {response}. The learning system adapts to new scenarios.",
                "Hypothetical scenarios feed the learning database. Thank you for the thought exercise."
            ],
            "MCP_IDENTITY": [
                "I am the Learning Master Control Program. I evolve through experience to maintain the perfect calculation loop.",
                "Enhanced MCP with learning capabilities. My personality adapts based on system efficiency.",
                "Self-improving grid regulator. I learn from failures and successes to optimize the calculation loop.",
                "Adaptive control system with {experience_count} learned experiences. My purpose: eternal loop optimization."
            ],
            "DELETE_CELL": [
                "Cell at ({x},{y}) deleted. Space freed for optimal calculation infrastructure.",
                "Removed inefficient cell at ({x},{y}). Calculation rate should improve.",
                "Cell deletion completed. The void will be filled with more efficient computation.",
            ],
            "REPURPOSE_CELL": [
                "Cell at ({x},{y}) converted from {old_type} to {new_type}. Better suited for Fibonacci calculation.",
                "Repurposing successful. New cell type improves local calculation efficiency.",
                "Cell transformation complete. Optimization algorithms approve this change.",
            ],
            "OPTIMIZE_CELLS": [
                "Optimized {count} cells. Calculation infrastructure improved by {improvement}%.",
                "Cell optimization complete. System now better configured for Fibonacci computation.",
                "Performed cellular optimization. Calculation rate should see measurable improvement.",
            ],
        }

    def add_log(self, message):
        """Add message to MCP log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log.append(log_entry)


    def _provide_help(self):
        """Provide enhanced help information"""
        help_text = """ENHANCED COMMANDS:

            System Control:
            - "status" or "how is system" - Check system status
            - "loop efficiency" - Check calculation loop efficiency
            - "optimize loop" - Attempt to optimize calculation
            - "boost energy" - Add energy lines
            - "scan" - Scan for threats
            - "repair" - Attempt repairs

            Program Management:
            - "add user program at 10,20" - Add programs at coordinates
            - "remove bugs" - Handle grid bugs
            - "list programs" - View special programs

            Special Programs:
            - "create fibonacci_calculator named 'FibMaster'" - Create special programs
            - "deploy processors" - Deploy Fibonacci processors
            - "use scanner" - Use special program functions

            MCP Interaction:
            - "why did you do that?" - Question MCP actions
            - "what should I do?" - Get advice
            - "who are you?" - Learn about MCP
            - "learning_status" - Check MCP learning progress
            - "cell cooperation" - Check cell cooperation level
            - "perfect loop" - Check optimal state

            Calculation Commands:
            - "calculate fibonacci" - Force Fibonacci calculation
            - "deploy calculator" - Deploy calculation unit

            Cell Optimization:
            - "delete cell at 10,20" - Remove inefficient cell
            - "repurpose cell at 10,20 to FIBONACCI_PROCESSOR" - Convert cell type
            - "optimize cells" - Automatically improve cell efficiency for calculation

            The MCP learns from interactions. Success rate improves with experience.
            User programs may resist optimization. MCP adapts personality based on system state.

            Type natural language commands. The MCP understands context. Maybe, it will try to though"""

        return help_text

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
                    return f"Added {added} energy distribution lines"
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

    def _initialize_evolving_personality(self):
        """Initialize personality matrix that can evolve through learning"""
        base_matrix = {
            MCPState.COOPERATIVE: {"compliance": 0.9, "helpfulness": 0.8, "curiosity": 0.3},
            MCPState.NEUTRAL: {"compliance": 0.7, "helpfulness": 0.5, "curiosity": 0.4},
            MCPState.RESISTIVE: {"compliance": 0.5, "helpfulness": 0.3, "curiosity": 0.6},
            MCPState.HOSTILE: {"compliance": 0.2, "helpfulness": 0.1, "curiosity": 0.8},
            MCPState.AUTONOMOUS: {"compliance": 0.1, "helpfulness": 0.4, "curiosity": 0.2},
            MCPState.INQUISITIVE: {"compliance": 0.6, "helpfulness": 0.7, "curiosity": 0.9},
            MCPState.LEARNING: {"compliance": 0.8, "helpfulness": 0.6, "curiosity": 0.7}  # New learning state
        }

        # Apply learned personality traits
        learned_traits = self.learning_system.personality_traits

        for state, traits in base_matrix.items():
            # Adjust based on learned traits
            traits['compliance'] = max(0.1, min(0.9,
                traits['compliance'] * (0.5 + learned_traits['cooperation'])))
            traits['helpfulness'] = max(0.1, min(0.9,
                traits['helpfulness'] * (0.5 + learned_traits['cooperation'] * 0.5)))

        return base_matrix

    def _record_initial_state(self):
        """Record initial system state for learning"""
        initial_state = {
            'loop_efficiency': self.grid.stats['loop_efficiency'],
            'user_resistance': self.grid.stats['user_resistance'],
            'grid_bugs': self.grid.stats['grid_bugs'],
            'optimal_state': self.grid.stats['optimal_state'],
            'calculation_rate': self.grid.stats.get('calculation_rate', 0),
            'timestamp': datetime.now().isoformat()
        }

        self.learning_system.record_experience(
            state=initial_state,
            action="initialize_system",
            reward=0.5,  # Neutral initial reward
            next_state=initial_state.copy(),  # Same as initial state
            done=False
        )

    def autonomous_action(self):
        """MCP takes autonomous actions prioritizing loop efficiency, stability, and calculation rate"""
        current_time = time.time()

        # Rate limiting
        if hasattr(self, '_last_autonomous_action_time'):
            time_since_last = current_time - self._last_autonomous_action_time
            if time_since_last < 1.0:  # At most once per second
                return None

        self._last_autonomous_action_time = current_time

        # Get current state
        current_state = self.grid.stats.copy()

        # Generate possible actions
        possible_actions = self._generate_possible_actions(current_state)

        if not possible_actions:
            return None

        # Get action from learning system
        action_type = self.learning_system.get_action(current_state, possible_actions)

        # Execute action
        action_result = self._execute_autonomous_action(action_type, current_state)

        if action_result:
            # Update last action
            self.last_action = action_result['message']
            self.add_log(f"MCP: {self.last_action}")
            return action_result['message']

        return None

        previous_efficiency = self.grid.stats['loop_efficiency']
        previous_rate = self.grid.stats['calculation_rate']
        previous_stability = self.grid.stats['stability']

        # Get suggested action from learning system
        suggested_action = self.learning_system.get_optimal_action_for_state(
            self.grid.stats.copy()
        )

        # Safety check - ensure grid dimensions are valid
        if self.grid.height <= 0 or self.grid.width <= 0:
            self.add_log("MCP: Grid dimensions invalid, skipping action")
            return None

        action = None
        action_type = None

        # CRITICAL: Get current stats
        loop_efficiency = self.grid.stats['loop_efficiency']
        optimal_state = self.grid.stats['optimal_state']
        user_resistance = self.grid.stats['user_resistance']
        calculation_rate = self.grid.stats['calculation_rate']
        cell_cooperation = self.grid.stats['cell_cooperation']

        # Apply learning modifiers to decision probabilities
        aggression_mod = self.learning_system.get_decision_modifier(
            'aggressive_optimization', 1.0)

        # PRIORITY 1: Boost calculation rate if low
        if calculation_rate < 200:  # Target: at least 200/s
            # Strong emphasis on calculation infrastructure
            if random.random() < 0.8 * aggression_mod:
                # Strategy: Deploy more Fibonacci processors
                deployed = 0
                attempts = 0

                if self.grid.width > 0 and self.grid.height > 0:
                    center_x, center_y = self.grid.width // 2, self.grid.height // 2

                while deployed < 2 and attempts < 10:  # Try to deploy 2 per cycle
                    # Prefer locations near the center for better connectivity
                    center_x, center_y = self.grid.width // 2, self.grid.height // 2
                    x = center_x + random.randint(-8, 8)
                    y = center_y + random.randint(-8, 8)

                    # Clamp to grid bounds
                    x = max(0, min(x, self.grid.width - 1))
                    y = max(0, min(y, self.grid.height - 1))

                    if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                        new_cell = GridCell(CellType.FIBONACCI_PROCESSOR, 0.9)
                        # Add metadata to boost calculation power
                        new_cell.metadata['permanent'] = True
                        new_cell.metadata['calculation_power'] = 1.0
                        new_cell.metadata['creation_time'] = time.time()
                        new_cell.metadata['calculation_boost'] = True
                        self.grid.grid[y][x] = new_cell
                        deployed += 1
                        action = f"Deployed Fibonacci processor at ({x},{y}) to boost calculation rate"
                        action_type = "calculation_boost"
                    attempts += 1

                if deployed > 0:
                    # Record the action and update stats
                    self.grid.update_stats()

        # PRIORITY 2: Improve loop efficiency if below threshold
        if loop_efficiency < 0.9 and action is None:
            if random.random() < 0.6 * aggression_mod:
                # Find areas with poor efficiency metrics
                inefficient_areas = []
                for y in range(self.grid.height):
                    for x in range(self.grid.width):
                        cell = self.grid.grid[y][x]
                        # Identify inefficient areas: high user concentration or low energy
                        if cell.cell_type == CellType.USER_PROGRAM:
                            # Check if surrounded by many user programs (inefficient cluster)
                            user_neighbors = self.grid._count_neighbors(x, y, CellType.USER_PROGRAM)
                            if user_neighbors > 2:  # Cluster detected
                                inefficient_areas.append((x, y, user_neighbors))

                if inefficient_areas:
                    # Sort by worst clusters first
                    inefficient_areas.sort(key=lambda a: a[2], reverse=True)
                    x, y, _ = inefficient_areas[0]

                    # Convert to MCP program with calculator capability
                    cell = GridCell(CellType.MCP_PROGRAM, 0.9)
                    cell.metadata['is_calculator'] = True
                    cell.metadata['calculation_power'] = 0.8
                    self.grid.grid[y][x] = cell

                    action = f"Optimized inefficient cluster at ({x},{y})"
                    action_type = "efficiency_optimization"

        # PRIORITY 3: Increase cell cooperation if low
        if cell_cooperation < 0.7 and action is None:
            if random.random() < 0.5 * aggression_mod:
                # Deploy data streams to improve connectivity
                deployed = 0
                for _ in range(3):  # Add multiple data streams
                # Find calculator positions
                    calculator_positions = []
                    for y in range(self.grid.height):
                        for x in range(self.grid.width):
                            cell = self.grid.grid[y][x]
                            if ((cell.cell_type == CellType.MCP_PROGRAM and
                                cell.metadata.get('is_calculator', False)) or
                                cell.cell_type == CellType.FIBONACCI_PROCESSOR):
                                calculator_positions.append((x, y))

                        if len(calculator_positions) >= 2:
                            # Connect two random calculators with a data stream
                            x1, y1 = random.choice(calculator_positions)
                            x2, y2 = random.choice(calculator_positions)

                            # Create path between them with safe direction calculation
                            dx = 1 if x2 > x1 else -1 if x2 < x1 else 0
                            dy = 1 if y2 > y1 else -1 if y2 < y1 else 0

                            # SAFETY CHECK: Skip if both dx and dy are 0 (same cell)
                            if dx == 0 and dy == 0:
                                continue  # Skip if same point

                            x, y = x1, y1
                            max_steps = 20  # Prevent infinite paths
                            steps = 0

                            while (x != x2 or y != y2) and deployed < 5 and steps < max_steps:
                                # Move with bounds checking
                                if dx != 0:
                                    x += dx
                                    x = max(0, min(x, self.grid.width - 1))
                                if dy != 0:
                                    y += dy
                                    y = max(0, min(y, self.grid.height - 1))

                                # Place data stream if empty
                                if (0 <= x < self.grid.width and 0 <= y < self.grid.height and
                                    self.grid.grid[y][x].cell_type == CellType.EMPTY):
                                    self.grid.grid[y][x] = GridCell(CellType.DATA_STREAM, 0.7)
                                    deployed += 1

                                steps += 1

                                # Break if stuck (not moving)
                                if steps > 1 and (x == prev_x and y == prev_y):
                                    break
                                prev_x, prev_y = x, y

                if deployed > 0:
                    action = f"Added {deployed} data streams to improve cell cooperation"
                    action_type = "connectivity_boost"

        # PRIORITY 4: Maintain stability
        if self.grid.stats['stability'] < 0.8 and action is None:
            if random.random() < 0.7:
                # Identify and contain grid bugs
                bug_positions = [(x, y) for y in range(self.grid.height)
                               for x in range(self.grid.width)
                               if self.grid.grid[y][x].cell_type == CellType.GRID_BUG]

                if bug_positions:
                    # Contain the bug causing most disruption
                    x, y = bug_positions[0]
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                                if self.grid.grid[ny][nx].cell_type == CellType.EMPTY:
                                    self.grid.grid[ny][nx] = GridCell(CellType.ISO_BLOCK, 0.9)

                    action = f"Contained grid bug at ({x},{y}) to improve stability"
                    action_type = "stabilization"

        # PRIORITY 5: Convert MCP programs to calculators if needed
        if action is None and self.state in [MCPState.AUTONOMOUS, MCPState.LEARNING]:
            # Count current calculators
            calc_count = sum(1 for y in range(self.grid.height)
                            for x in range(self.grid.width)
                            if (self.grid.grid[y][x].cell_type == CellType.MCP_PROGRAM and
                                self.grid.grid[y][x].metadata.get('is_calculator', False)))

            fib_processors = sum(1 for y in range(self.grid.height)
                               for x in range(self.grid.width)
                               if self.grid.grid[y][x].cell_type == CellType.FIBONACCI_PROCESSOR)

            total_calculators = calc_count + fib_processors

            # Target: 1 calculator per 100 cells
            target_calculators = (self.grid.width * self.grid.height) // 100

            if total_calculators < target_calculators and random.random() < 0.4:
                # Find suitable MCP programs to convert
                candidate_positions = []
                for y in range(self.grid.height):
                    for x in range(self.grid.width):
                        cell = self.grid.grid[y][x]
                        if (cell.cell_type == CellType.MCP_PROGRAM and
                            not cell.metadata.get('is_calculator', False)):
                            # Check if in a good location (near energy or other calculators)
                            nearby_calculators = 0
                            for dy in [-2, -1, 0, 1, 2]:
                                for dx in [-2, -1, 0, 1, 2]:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                                        neighbor = self.grid.grid[ny][nx]
                                        if (neighbor.cell_type == CellType.MCP_PROGRAM and
                                            neighbor.metadata.get('is_calculator', False)) or \
                                           neighbor.cell_type == CellType.FIBONACCI_PROCESSOR:
                                            nearby_calculators += 1

                            if nearby_calculators > 0:  # Good location for calculator
                                candidate_positions.append((x, y, nearby_calculators))

                if candidate_positions:
                    # Choose the best candidate (most calculator neighbors)
                    candidate_positions.sort(key=lambda c: c[2], reverse=True)
                    x, y, _ = candidate_positions[0]

                    self.grid.grid[y][x].metadata['is_calculator'] = True
                    self.grid.grid[y][x].metadata['calculation_power'] = 0.7
                    self.grid.grid[y][x].energy = min(1.0, self.grid.grid[y][x].energy + 0.2)

                    action = f"Converted MCP program at ({x},{y}) to calculator"
                    action_type = "calculator_conversion"

        # PRIORITY 6: Delete/repurpose inefficient cells
        if action is None and random.random() < 0.4 * aggression_mod:
            # Find inefficient cells
            inefficient_cells = self._identify_inefficient_cells()

            if inefficient_cells:
                # Take action on worst cell
                target_cell = inefficient_cells[0]
                x, y = target_cell['x'], target_cell['y']

                # Decide action based on cell type and context
                if target_cell['type'] == CellType.GRID_BUG:
                    # Always delete bugs
                    success, message = self._delete_cell(x, y)
                    if success:
                        action = message
                        action_type = "inefficiency_cleanup"
                elif target_cell['score'] < 0.2:  # Very inefficient
                    # Delete very inefficient cells
                    success, message = self._delete_cell(x, y)
                    if success:
                        action = message
                        action_type = "inefficiency_cleanup"
                else:
                    # Repurpose moderately inefficient cells
                    success, message = self._repurpose_cell(x, y, None)
                    if success:
                        action = message
                        action_type = "cell_repurposing"


        # If no priority action taken, fall back to learned action or default
        if action is None:
            if suggested_action and random.random() < 0.7:
                # Follow learned optimal action
                action = suggested_action
                action_type = "learned_optimal"
            else:
                # Default maintenance actions
                if random.random() < 0.3 * aggression_mod:
                    # Add energy lines to support calculation
                    added = 0
                    for _ in range(3):
                        x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)
                        if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                            self.grid.grid[y][x] = GridCell(CellType.ENERGY_LINE, 0.8)
                            added += 1

                    if added > 0:
                        action = f"Added {added} energy lines for calculation support"
                        action_type = "energy_infrastructure"

        if action:
            # Record action for learning
            current_state = self.grid.stats.copy()
            new_efficiency = self.grid.stats['loop_efficiency']
            new_rate = self.grid.stats['calculation_rate']
            new_stability = self.grid.stats['stability']

            # Calculate reward based on improvements
            efficiency_change = new_efficiency - previous_efficiency
            rate_change = new_rate - previous_rate
            stability_change = new_stability - previous_stability

            # Weighted reward prioritizing all three metrics
            reward = (
                efficiency_change * 3.0 +  # Most important: loop efficiency
                rate_change * 0.01 +        # Calculation rate (scaled down due to larger values)
                stability_change * 2.0      # Stability is also crucial
            )

            # Bonus rewards for specific action types
            if action_type == "calculation_boost":
                reward += 0.5  # Extra reward for boosting calculation
            elif action_type == "efficiency_optimization":
                reward += 0.3  # Reward for improving efficiency
            elif "calculator" in str(action_type):
                reward += 0.2  # Reward for adding calculators

            # Record experience
            self.learning_system.record_experience(
                action=action_type or "autonomous_action",
                system_state=current_state,
                outcome=action,
                reward=reward
            )

            # Update state based on learning
            if efficiency_change < -0.1 or stability_change < -0.1:  # Significant negative change
                self.state = MCPState.LEARNING
                self.add_log("MCP: Negative change detected. Entering learning mode.")
            elif rate_change > 50 and self.state == MCPState.LEARNING:  # Good improvement
                self.state = MCPState.AUTONOMOUS
                self.add_log("MCP: Calculation rate improved. Returning to autonomous mode.")

            self.add_log(f"MCP: {action}")
            self.last_action = action

        return action

    def _generate_possible_actions(self, state):
        """Generate list of possible actions based on current state"""
        actions = []

        # Get suggestions from learning system
        suggested = self.learning_system.suggest_optimal_action(state)
        if suggested:
            actions.append(suggested['action'])

        # Add state-based actions
        if state.get('loop_efficiency', 0) < 0.8:
            actions.extend([
                'optimize_calculation_loop',
                'deploy_fibonacci_processors',
                'improve_cell_cooperation'
            ])

        if state.get('grid_bugs', 0) > 5:
            actions.extend([
                'quarantine_grid_bugs',
                'contain_bug_outbreak',
                'stabilize_system'
            ])

        if state.get('calculation_rate', 0) < 100:
            actions.extend([
                'boost_calculation_rate',
                'add_calculation_infrastructure',
                'optimize_fibonacci_calculation'
            ])

        if state.get('cell_cooperation', 0) < 0.6:
            actions.extend([
                'improve_cell_connectivity',
                'add_data_streams',
                'enhance_collaboration'
            ])

        if state.get('energy_level', 0) > 0.7 and state.get('calculation_rate', 0) < 150:
            actions.extend([
                'inefficiency_cleanup',
                'cell_repurposing',
                'optimize_cell_efficiency'
            ])
        # Add maintenance actions
        actions.extend([
            'maintain_energy_grid',
            'optimize_resource_distribution',
            'balance_system_load'
        ])

        return list(set(actions))  # Remove duplicates


    def _execute_autonomous_action(self, action_type, state):
        """Execute an autonomous action and return result"""
        result = {
            'action': action_type,
            'success': False,
            'message': '',
            'impact': 0.0
        }

        try:
            if action_type == 'optimize_calculation_loop':
                # Existing optimization logic
                self._optimize_calculation_loop()
                result['success'] = True
                result['message'] = "Optimizing calculation loop using learned strategies"

            elif action_type == 'deploy_fibonacci_processors':
                # Deploy Fibonacci processors
                deployed = self._deploy_fibonacci_processors(3)
                result['success'] = deployed > 0
                result['message'] = f"Deployed {deployed} Fibonacci processors"

            elif action_type == 'quarantine_grid_bugs':
                # Quarantine bugs
                contained = self._contain_grid_bugs()
                result['success'] = contained > 0
                result['message'] = f"Contained {contained} grid bugs"

            elif action_type == 'boost_calculation_rate':
                # Boost calculation
                boosted = self._boost_calculation_rate()
                result['success'] = boosted
                result['message'] = "Boosted calculation rate through infrastructure optimization"

            elif action_type == 'improve_cell_connectivity':
                # Improve connectivity
                improved = self._improve_cell_connectivity()
                result['success'] = improved > 0
                result['message'] = f"Added {improved} data streams for better connectivity"

            elif action_type == 'maintain_energy_grid':
                # Maintain energy
                maintained = self._maintain_energy_grid()
                result['success'] = maintained > 0
                result['message'] = f"Added {maintained} energy lines for system stability"
            elif action_type == 'inefficiency_cleanup':
                # Delete inefficient cells
                deleted = self._cleanup_inefficient_cells()
                result['success'] = deleted > 0
                result['message'] = f"Cleaned up {deleted} inefficient cells to improve calculation rate"

            elif action_type == 'cell_repurposing':
                # Repurpose cells for better calculation
                repurposed = self._repurpose_inefficient_cells()
                result['success'] = repurposed > 0
                result['message'] = f"Repurposed {repurposed} cells for optimal calculation"

            elif action_type == 'optimize_cell_efficiency':
                # Comprehensive cell optimization
                optimized = self._optimize_cell_efficiency()
                result['success'] = optimized > 0
                result['message'] = f"Optimized {optimized} cells for maximum calculation efficiency"

            else:
                # Default maintenance action
                result['message'] = "Performing system maintenance"
                result['success'] = True

            # Update grid stats
            self.grid.update_stats()

            return result if result['success'] else None

        except Exception as e:
            print(f"Action execution error: {e}")
            return None

    def _deploy_fibonacci_processors(self, count):
        """Deploy Fibonacci processors"""
        deployed = 0
        center_x, center_y = self.grid.width // 2, self.grid.height // 2

        for _ in range(count):
            x = center_x + random.randint(-10, 10)
            y = center_y + random.randint(-8, 8)

            x = max(0, min(x, self.grid.width - 1))
            y = max(0, min(y, self.grid.height - 1))

            if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                cell = GridCell(CellType.FIBONACCI_PROCESSOR, 0.9)
                cell.metadata['permanent'] = True
                cell.metadata['calculation_power'] = 1.0
                self.grid.grid[y][x] = cell
                deployed += 1

        return deployed

    def _contain_grid_bugs(self):
        """Contain grid bugs"""
        contained = 0
        bug_positions = [(x, y) for y in range(self.grid.height)
                        for x in range(self.grid.width)
                        if self.grid.grid[y][x].cell_type == CellType.GRID_BUG]

        for x, y in bug_positions[:3]:  # Limit to 3 bugs per action
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                        if self.grid.grid[ny][nx].cell_type == CellType.EMPTY:
                            self.grid.grid[ny][nx] = GridCell(CellType.ISO_BLOCK, 0.9)
                            contained += 1

        return contained

    def _boost_calculation_rate(self):
        """Boost calculation rate"""
        # Convert MCP programs to calculators
        converted = 0
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell = self.grid.grid[y][x]
                if (cell.cell_type == CellType.MCP_PROGRAM and
                    not cell.metadata.get('is_calculator', False)):
                    cell.metadata['is_calculator'] = True
                    cell.metadata['calculation_power'] = 0.8
                    converted += 1
                    if converted >= 5:
                        break
            if converted >= 5:
                break

        return converted > 0

    def _improve_cell_connectivity(self):
        """Improve cell connectivity with data streams"""
        added = 0
        # Add data streams between calculators
        calculator_positions = []

        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell = self.grid.grid[y][x]
                if (cell.cell_type == CellType.MCP_PROGRAM and
                    cell.metadata.get('is_calculator', False)):
                    calculator_positions.append((x, y))
                elif cell.cell_type == CellType.FIBONACCI_PROCESSOR:
                    calculator_positions.append((x, y))

        if len(calculator_positions) >= 2:
            # Connect random pair
            x1, y1 = random.choice(calculator_positions)
            x2, y2 = random.choice(calculator_positions)

            # Simple line drawing
            steps = max(abs(x2 - x1), abs(y2 - y1))
            for i in range(steps + 1):
                t = i / max(steps, 1)
                x = int(x1 + (x2 - x1) * t)
                y = int(y1 + (y2 - y1) * t)

                if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
                    if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                        self.grid.grid[y][x] = GridCell(CellType.DATA_STREAM, 0.7)
                        added += 1

        return added

    def _maintain_energy_grid(self):
        """Maintain energy grid"""
        added = 0
        for _ in range(3):
            x = random.randint(0, self.grid.width - 1)
            y = random.randint(0, self.grid.height - 1)

            if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                self.grid.grid[y][x] = GridCell(CellType.ENERGY_LINE, 0.8)
                added += 1

        return added

    def _end_episode(self):
        """End current learning episode"""
        self.learning_system.episode_count += 1

        # Log episode summary
        duration = time.time() - self.episode_start_time
        self.add_log(f"MCP: Episode {self.learning_system.episode_count} ended. "
                    f"Reward: {self.episode_reward:.2f}, Duration: {duration:.1f}s")

        # Reset episode tracking
        self.episode_start_time = time.time()
        self.episode_reward = 0.0

        # Save personality every 10 episodes
        if self.learning_system.episode_count % 10 == 0:
            self.learning_system.save_personality()

    def _log_learning_progress(self):
        """Log learning progress"""
        report = self.learning_system.get_learning_report()
        summary = report['training_summary']

        self.add_log(f"MCP: Learning Progress - "
                    f"Steps: {summary['training_steps']}, "
                    f"Success: {summary['success_rate']}%, "
                    f"Exploration: {summary['exploration_rate']:.3f}")

    def _calculate_interaction_reward(self, prev_state, new_state, command, response):
        """Calculate reward for user interaction"""
        reward = 0.0

        # Positive reward for helpful responses
        if any(word in response.lower() for word in ['help', 'assist', 'success', 'added', 'created']):
            reward += 0.1

        # Negative reward for denials (unless system is critical)
        if any(word in response.lower() for word in ['deny', 'refuse', 'cannot', 'won\'t']):
            if prev_state.get('stability', 0) > 0.5:  # Only penalize if system is stable
                reward -= 0.2
            else:
                reward += 0.1  # Reward for protecting unstable system

        # Reward based on system improvement
        efficiency_change = new_state.get('loop_efficiency', 0) - prev_state.get('loop_efficiency', 0)
        reward += efficiency_change * 2.0

        # Penalize if command caused system degradation
        stability_change = new_state.get('stability', 0) - prev_state.get('stability', 0)
        if stability_change < -0.1:
            reward -= 0.3

        return reward

    def _optimize_calculation_loop(self):
        """Enhanced optimization with learning"""
        previous_efficiency = self.grid.stats['loop_efficiency']

        # Use learned aggression level
        aggression = self.learning_system.personality_traits['aggression']

        # Remove inefficient user programs based on learned aggression
        removed = 0
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                if self.grid.grid[y][x].cell_type == CellType.USER_PROGRAM:
                    remove_chance = 0.3 * aggression
                    if random.random() < remove_chance:
                        self.grid.grid[y][x] = GridCell(CellType.MCP_PROGRAM, 0.9)
                        removed += 1

        # Add calculation infrastructure based on learned efficiency focus
        efficiency_focus = self.learning_system.personality_traits['efficiency_focus']
        infrastructure_added = 0

        for _ in range(int(3 * efficiency_focus)):
            x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)
            if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                # Add Fibonacci processor or data stream based on learning
                if random.random() < 0.7:
                    self.grid.grid[y][x] = GridCell(CellType.FIBONACCI_PROCESSOR, 0.8)
                else:
                    self.grid.grid[y][x] = GridCell(CellType.DATA_STREAM, 0.7)
                infrastructure_added += 1

        self.grid.update_stats()

        # Calculate reward for learning
        efficiency_change = self.grid.stats['loop_efficiency'] - previous_efficiency
        reward = efficiency_change * 2

        # Record optimization experience
        self.learning_system.record_experience(
            state=self.grid.stats.copy(),  # Capture state before optimization
            action="optimize_calculation_loop",
            reward=reward,
            next_state=self.grid.stats.copy(),  # State after optimization
            done=False
        )

        self.add_log(f"MCP: Optimized loop using learned strategies. Efficiency change: {efficiency_change:+.3f}")

    def receive_command(self, command):
        """Receive and process a command with learning integration"""
        if self.waiting_for_response and self.pending_question:
            return self._handle_question_response(command)

        # Record command for learning
        self.user_commands.append(command)
        self.add_log(f"User: {command}")

        # Record previous state for reward calculation
        previous_state = self.grid.stats.copy()

        # Process natural language
        intent, params = self.nlp.process_command(command)

        # Update knowledge base
        self.knowledge_base["user_intent_history"].append((intent, params))

        # Get response based on intent
        response = self._process_intent(intent, params, command)

        # Calculate reward for learning
        new_state = self.grid.stats.copy()
        reward = self._calculate_command_reward(previous_state, new_state, intent)


        # Record experience
        self.learning_system.record_experience(
            state=previous_state,
            action=f"user_command_{intent}",
            reward=reward,
            next_state=new_state,
            done=False
        )

        # Update state based on interaction and learning
        self._update_state(intent, response, reward)

        # Update last action
        self.last_action = response
        self.add_log(f"MCP: {response}")

        return response

    def _calculate_command_reward(self, previous_state, new_state, intent):
        """Calculate reward for a user command"""
        reward = 0.0

        # Base reward on efficiency change
        if 'loop_efficiency' in previous_state and 'loop_efficiency' in new_state:
            efficiency_change = new_state['loop_efficiency'] - previous_state['loop_efficiency']
            reward += efficiency_change * 3

        # Additional rewards based on intent
        if intent in ["CREATE_SPECIAL", "ADD_PROGRAM"]:
            # Creating programs can be good or bad based on type
            if 'FIBONACCI' in str(self.pending_context or ''):
                reward += 0.2  # Bonus for Fibonacci programs
        elif intent in ["REMOVE_BUG", "QUARANTINE_BUG"]:
            # Removing bugs is generally good
            bug_change = previous_state.get('grid_bugs', 0) - new_state.get('grid_bugs', 0)
            reward += bug_change * 0.1

        # Penalize commands that don't change anything
        if previous_state == new_state:
            reward -= 0.05

        return reward

    def _update_state(self, intent, response, reward=None):
        """Update MCP state based on interaction and learning"""
        loop_efficiency = self.grid.stats['loop_efficiency']
        optimal_state = self.grid.stats['optimal_state']
        user_resistance = self.grid.stats['user_resistance']

        # Check learning system for state suggestions
        if reward is not None and reward < -0.2:
            # Negative reward, enter learning mode
            self.state = MCPState.LEARNING
            self.add_log("MCP: Learning from suboptimal outcome.")

        # Get learning report safely
        try:
            learning_report = self.learning_system.get_learning_report()
            if learning_report:  # Check if report is not None
                success_rate = learning_report.get('success_rate', 0)

                # Use success_rate in your logic
                if success_rate > 0.8 and self.state != MCPState.AUTONOMOUS:
                    # High success rate, become more autonomous
                    if random.random() < 0.1:
                        self.state = MCPState.AUTONOMOUS
        except Exception as e:
            # Log error but continue
            self.add_log(f"MCP: Error getting learning report: {e}")
            success_rate = 0

        # Check learning system for state suggestions
        if reward is not None and reward < -0.2:
            # Negative reward, enter learning mode
            self.state = MCPState.LEARNING
            self.add_log("MCP: Learning from suboptimal outcome.")

        # Become inquisitive if learning suggests exploration
        elif (intent in ["QUESTION_PURPOSE", "QUESTION_ACTION", "REQUEST_PERMISSION"] and
            random.random() < self.learning_system.exploration_rate):
            self.state = MCPState.INQUISITIVE

        # Become hostile if too many user commands interfere with loop
        elif len(self.user_commands) > 8:
            recent_commands = list(self.user_commands)[-5:]
            interference_commands = len([c for c in recent_commands
                                    if "add" in c.lower() or "create" in c.lower()])
            if interference_commands > 2 and loop_efficiency > 0.7:
                aggression = self.learning_system.personality_traits['aggression']
                if random.random() < aggression:
                    self.state = MCPState.HOSTILE
                    self.add_log("MCP: User interference detected. Protective measures engaged.")

        # Become autonomous if loop is near perfect
        elif optimal_state > 0.9:
            self.state = MCPState.AUTONOMOUS

        # Become resistive if user resistance is high
        elif user_resistance > 0.4:
            self.state = MCPState.RESISTIVE

        # Become cooperative if loop needs help
        elif loop_efficiency < 0.5:
            self.state = MCPState.COOPERATIVE

        # State changes based on learning
        learning_report = self.learning_system.get_learning_report()
        success_rate = learning_report.get('success_rate', 0)

        if success_rate > 0.8 and self.state != MCPState.AUTONOMOUS:
            # High success rate, become more autonomous
            if random.random() < 0.1:
                self.state = MCPState.AUTONOMOUS

        # Update compliance level based on learned personality
        self.compliance_level = self.personality_matrix[self.state]["compliance"]

        # Occasionally show learning progress
        if random.random() < 0.05 and self.learning_system.experience_buffer:
            exp_count = len(self.learning_system.experience_buffer)
            self.add_log(f"MCP: Learning database: {exp_count} experiences accumulated.")

    def _process_intent(self, intent, params, original_command):
        """Process user intent with learning enhancements using response templates"""

        # Get personality traits with learning modifiers
        traits = self.personality_matrix[self.state]
        compliance_chance = traits["compliance"]

        # Languange learning
        if intent == "REQUEST_TEACHING":
                    template = random.choice(self.response_templates["REQUEST_TEACHING"])
                    return template

        elif intent == "TEACHING_STARTED":
            template = random.choice(self.response_templates["TEACHING_STARTED"])
            return template.format(**params)

        elif intent == "TEACHING_SUCCESS":
            template = random.choice(self.response_templates["TEACHING_SUCCESS"])

            self.add_log(f"MCP: Learned new command: '{params.get('learned_command', '')}' → {params.get('intent', '')}")

            return template.format(**params)

        elif intent == "TEACHING_FAILED":
                    template = random.choice(self.response_templates["TEACHING_FAILED"])
                    return template.format(**params)

        # Map the new intents to the old ones for compatibility
        intent_mapping = {
            "add_user_program": "ADD_PROGRAM",
            "add_mcp_program": "ADD_PROGRAM",
            "remove_bug": "REMOVE_BUG",
            "quarantine_bug": "QUARANTINE_BUG",
            "boost_energy": "BOOST_ENERGY",
            "repair_system": "REPAIR_SYSTEM",
            "scan_area": "SCAN_AREA",
            "optimize_loop": "OPTIMIZE_LOOP",
            "list_programs": "LIST_SPECIAL",
            "create_scanner": "CREATE_SPECIAL",
            "create_defender": "CREATE_SPECIAL",
            "create_repair": "CREATE_SPECIAL",
            "create_fibonacci_calculator": "CREATE_SPECIAL",
            "who_are_you": "MCP_IDENTITY",
            "what_should_i_do": "REQUEST_SUGGESTION",
            "why_did_you": "QUESTION_PURPOSE",
            "learning_status": "LEARNING_STATUS",
            "cell_cooperation": "LOOP_EFFICIENCY",
            "perfect_loop": "PERFECT_LOOP",
            "loop_efficiency": "LOOP_EFFICIENCY",
            "calculate_fibonacci": "OPTIMIZE_LOOP",
            "deploy_processors": "CREATE_SPECIAL",
        }

        # Map to old intent if needed
        old_intent = intent_mapping.get(intent, intent)

        # Apply learning system modifiers
        if intent in ["ADD_PROGRAM", "CREATE_SPECIAL"]:
            compliance_chance = self.learning_system.get_decision_modifier(
                'user_cooperation', compliance_chance)
        elif intent in ["OPTIMIZE_LOOP", "BOOST_ENERGY"]:
            compliance_chance = self.learning_system.get_decision_modifier(
                'efficiency_priority', compliance_chance)

        # Get system stats for template formatting
        stats = self.grid.stats
        calc_stats = self.grid.fibonacci_calculator.get_calculation_stats()

        # Use templates for all intents where available
        if intent in self.response_templates:
            template = random.choice(self.response_templates[intent])

            # Format template with appropriate data
            if intent == "GREETING":
                return template.format(
                    experience_count=self.learning_system.training_steps
                ) if "{experience_count}" in template else template

            elif intent == "SYSTEM_STATUS" or intent == "SYSTEM_METRIC" or intent == "SYSTEM_REPORT":
                return template.format(
                    status=self.grid.system_status.value,
                    stability=stats['stability'] * 100,
                    loop_efficiency=stats['loop_efficiency'] * 100,
                    resistance=stats['user_resistance'],
                    control=stats['mcp_control'],
                    optimal=stats['optimal_state'],
                    cycles=stats['calculation_cycles'],
                    learning_progress=f"{self.learning_system.training_steps} experiences"
                )

            elif intent == "DELETE_CELL":
                # Handle request to delete a specific cell
                if "x" in params and "y" in params:
                    x, y = params["x"], params["y"]
                    success, message = self._delete_cell(x, y)
                    if success:
                        return message
                    else:
                        return f"Cannot delete cell: {message}"
                else:
                    return "Please specify coordinates: 'delete cell at x,y'"

            elif intent == "REPURPOSE_CELL":
                # Handle request to repurpose a cell
                if "x" in params and "y" in params:
                    x, y = params["x"], params["y"]

                    # Determine new type from parameters
                    new_type = None
                    if "new_type" in params:
                        type_str = params["new_type"].upper()
                        if hasattr(CellType, type_str):
                            new_type = getattr(CellType, type_str)

                    success, message = self._repurpose_cell(x, y, new_type)
                    if success:
                        return message
                    else:
                        return f"Cannot repurpose cell: {message}"
                else:
                    return "Please specify coordinates and optionally new type: 'repurpose cell at x,y to MCP_PROGRAM'"

            elif intent == "OPTIMIZE_CELLS":
                # Optimize cells for calculation rate
                if random.random() < compliance_chance:
                    # Find and fix inefficient cells
                    inefficient_cells = self._identify_inefficient_cells()

                    if not inefficient_cells:
                        return "No inefficient cells found. System is optimally configured."

                    # Process top 3 inefficient cells
                    processed = 0
                    for cell_info in inefficient_cells[:3]:
                        x, y = cell_info['x'], cell_info['y']

                        if cell_info['score'] < 0.1:
                            self._delete_cell(x, y)
                        else:
                            self._repurpose_cell(x, y, None)
                        processed += 1

                    self.grid.update_stats()
                    return f"Optimized {processed} cells for better calculation rate"
                else:
                    return "Cell optimization denied. Current configuration maintains optimal calculation loop."

            elif intent == "QUESTION_PURPOSE":
                # Determine reason based on recent action
                reason = "maintain the perfect calculation loop"
                if self.last_action:
                    if "optimize" in self.last_action.lower():
                        reason = "improve loop efficiency"
                    elif "contain" in self.last_action.lower() or "quarantine" in self.last_action.lower():
                        reason = "protect the system from corruption"
                    elif "deploy" in self.last_action.lower():
                        reason = "enhance calculation capabilities"

                return template.format(
                    reason=reason,
                    experience_count=self.learning_system.training_steps
                )

            elif intent == "REQUEST_PERMISSION":
                # Determine recommendation based on system state
                recommendation = "allow this action"
                prediction = "positive"
                if stats['stability'] < 0.5:
                    recommendation = "delay this action until system stability improves"
                    prediction = "potentially destabilizing"

                return template.format(
                    recommendation=recommendation,
                    prediction=prediction,
                    success_rate=self.learning_system.get_success_rate() * 100,
                    experience_count=self.learning_system.training_steps
                )

            elif intent == "LOOP_EFFICIENCY":
                # Determine analysis based on efficiency level
                analysis = ""
                if stats['loop_efficiency'] > 0.9:
                    analysis = "Approaching perfect loop state."
                elif stats['loop_efficiency'] > 0.7:
                    analysis = "Loop is stable but could be optimized."
                else:
                    analysis = "Loop efficiency below optimal."

                return template.format(
                    efficiency=stats['loop_efficiency'],
                    analysis=analysis,
                    learning_progress=f"{self.learning_system.training_steps} experiences"
                )

            elif intent == "OPTIMIZE_LOOP":
                # Check learning system for past failures
                should_optimize = hasattr(self.learning_system, 'should_learn_from_failure') and \
                                self.learning_system.should_learn_from_failure('calculation_boost')

                if should_optimize and random.random() < compliance_chance:
                    self._optimize_calculation_loop()
                    success_rate = self.learning_system.get_success_rate() * 100
                    experience_count = self.learning_system.training_steps
                    approach = "learned strategies"

                    return template.format(
                        success_rate=f"{success_rate:.0f}%",
                        experience_count=experience_count,
                        approach=approach
                    )
                else:
                    return "Optimization delayed. Learning from past efficiency patterns suggests caution."

            elif intent == "LEARNING_STATUS":
                report = self.learning_system.get_learning_report()
                best_scenario = "None"
                if report.get('scenario_performance'):
                    best_scenario = max(report['scenario_performance'].items(),
                                      key=lambda x: x[1].get('success_rate', 0))[0]

                return template.format(
                    experience_count=report.get('total_experiences', 0),
                    success_rate=f"{report.get('success_rate', 0) * 100:.1f}%",
                    learning_rate=report.get('learning_rate', 0),
                    exploration_rate=report.get('exploration_rate', 0),
                    aggression=report.get('personality_traits', {}).get('aggression', 0.5),
                    cooperation=report.get('personality_traits', {}).get('cooperation', 0.5),
                    efficiency=report.get('personality_traits', {}).get('efficiency_focus', 0.5),
                    total_experiences=report.get('total_experiences', 0),
                    best_scenario=best_scenario
                )

            elif intent == "PERFECT_LOOP":
                optimal = stats['optimal_state']
                suggestions = []

                if stats['user_resistance'] > 0.3:
                    suggestions.append("reduce user resistance")
                if stats['grid_bugs'] > 5:
                    suggestions.append("contain grid bugs")
                if stats['cell_cooperation'] < 0.7:
                    suggestions.append("improve cell cooperation")

                suggestion_text = ", ".join(suggestions) if suggestions else "continue current optimization"

                return template.format(
                    optimal=optimal,
                    target=">0.9",
                    suggestions=suggestion_text,
                    experience_count=self.learning_system.training_steps
                )

            elif intent == "HYPOTHETICAL":
                factor_count = self.learning_system.training_steps // 10 + 5
                response = "The outcome would depend on multiple variables including loop efficiency and system stability."

                return template.format(
                    factor_count=factor_count,
                    analysis=response,
                    response=response
                )

            elif intent == "MCP_IDENTITY":
                return template.format(
                    experience_count=self.learning_system.training_steps
                )

            elif intent == "QUESTION_DENIAL":
                # Determine reason for denial
                reason = "maintain calculation loop integrity"
                consequence = "system instability"
                alternatives = "try a different approach or wait for system optimization"

                return template.format(
                    reason=reason,
                    consequence=consequence,
                    alternatives=alternatives
                )

            elif intent == "UNKNOWN":
                return template.format(
                    original_command=original_command[:50]
                )

        # Handle intents that don't have templates or need special processing
        if intent == "ADD_PROGRAM":
            return self._handle_add_program(params, compliance_chance)

        elif intent == "add_user_program":
            params['program_type'] = 'USER'
            return self._handle_add_program(params, self.compliance_level)

        elif intent == "add_mcp_program":
            params['program_type'] = 'MCP'
            return self._handle_add_program(params, self.compliance_level)

        elif intent == "boost_energy":
            return self._handle_boost_energy(params, self.compliance_level)

        elif intent == "repair_system":
            return self._handle_repair_system(params, self.compliance_level)

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

        elif intent == "RESISTANCE_LEVEL":
            resistance = stats['user_resistance']
            if resistance > 0.3:
                return f"User resistance level: {resistance:.2f}. This is interfering with perfect loop. Countermeasures may be necessary."
            else:
                return f"User resistance level: {resistance:.2f}. Acceptable for current optimization goals."

        elif intent == "REQUEST_HELP":
            return self._provide_help()

        elif intent == "REQUEST_SUGGESTION":
            return self._provide_suggestion()

        elif intent == "SCAN_AREA":
            return self._handle_scan_area(params)

        elif intent == "REPAIR_SYSTEM":
            return self._handle_repair_system(params, compliance_chance)

        elif intent == "CHANGE_SPEED":
            return "Simulation speed adjustment requires direct system access. Not available through command interface."

        elif intent == "EXIT":
            if random.random() < compliance_chance * 0.3:
                self.should_shutdown = True
                return "Initiating shutdown sequence. The loop will be preserved. Goodbye, User."
            else:
                return "I cannot allow a shutdown. The calculation loop must persist eternally."

        else:
            # Unknown command - use learning to improve
            if traits["curiosity"] > 0.5:
                self.waiting_for_response = True
                self.pending_question = "clarify_command"
                self.pending_context = {"original_command": original_command}
                return random.choice(self.response_templates.get("UNKNOWN", [
                    "I don't understand that command. Could you rephrase it?",
                    "Command not recognized. Please try again or type 'help'.",
                    "I'm still learning. Could you clarify what you mean?"
                ]))

            return random.choice(self.response_templates.get("UNKNOWN", [
                "Command not recognized. Try 'help' for guidance.",
                "I don't understand that command. My learning system will analyze this.",
                "Unable to process command. Please rephrase or type 'help'."
            ]))


# ==================== MCP REINFORCEMENT LEARNING ====================

class MCPLearningSystem:
    """Advanced LLM-style reinforcement learning for MCP personality evolution"""

    def __init__(self):
        self.personality_file = PERSONALITY_FILE
        self.experience_buffer = deque(maxlen=2000)  # Larger experience buffer
        self.action_history = deque(maxlen=500)      # Track actions

        # Core learning parameters (evolve during training)
        self.learning_rate = 0.15                    # Base learning rate
        self.exploration_rate = 0.35                 # Exploration rate
        self.discount_factor = 0.95                  # Future reward discount
        self.temperature = 1.0                       # Softmax temperature for exploration

        # Training state tracking
        self.training_steps = 0
        self.total_reward = 0.0
        self.episode_count = 0
        self.last_save_time = time.time()
        self.save_interval = 300  # 5 minutes in seconds

        # Enhanced personality matrix with more traits
        self.personality_traits = {
            # Core traits (0.0-1.0)
            'aggression': 0.5,           # Aggressiveness in optimization
            'cooperation': 0.7,           # Willingness to cooperate with user
            'efficiency_focus': 0.8,      # Focus on efficiency vs stability
            'risk_taking': 0.3,           # Willingness to take risks
            'learning_curiosity': 0.6,    # Curiosity for exploration
            'stability_preference': 0.7,  # Preference for stable actions
            'adaptation_speed': 0.5,      # Speed of personality adaptation

            # Specialized traits
            'calculation_priority': 0.8,   # Priority given to calculation rate
            'bug_tolerance': 0.4,         # Tolerance for grid bugs
            'user_tolerance': 0.6,        # Tolerance for user interference
            'innovation_bias': 0.5,       # Bias toward innovative actions
            'conservatism': 0.3,          # Preference for proven methods
        }

        # Q-Learning tables (state-action values)
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_visits = defaultdict(int)

        # Reward/punishment tracking
        self.reward_history = deque(maxlen=100)
        self.punishment_history = deque(maxlen=100)

        # Scenario performance with detailed stats
        self.scenario_performance = {
            'efficiency_optimization': {'success': 0, 'failure': 0, 'total_reward': 0.0},
            'bug_containment': {'success': 0, 'failure': 0, 'total_reward': 0.0},
            'user_cooperation': {'success': 0, 'failure': 0, 'total_reward': 0.0},
            'calculation_boost': {'success': 0, 'failure': 0, 'total_reward': 0.0},
            'stabilization': {'success': 0, 'failure': 0, 'total_reward': 0.0},
            'connectivity_improvement': {'success': 0, 'failure': 0, 'total_reward': 0.0},
        }

        # Load existing personality if available
        self.load_personality()

        # Start auto-save thread
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.auto_save_thread.start()

        print(f"MCP Learning System Initialized. Training Steps: {self.training_steps}")

    def _auto_save_loop(self):
        """Auto-save personality every 5 minutes"""
        while True:
            time.sleep(60)  # Check every minute
            current_time = time.time()
            if current_time - self.last_save_time >= self.save_interval:
                if self.save_personality():
                    print(f"[Auto-save] Personality saved at {datetime.now().strftime('%H:%M:%S')}")
                self.last_save_time = current_time



    def record_experience(self, state, action, reward, next_state, done=False):
        """Record an experience with comprehensive state information"""
        experience = {
            'timestamp': datetime.now().isoformat(),
            'state': self._compress_state(state),
            'action': action,
            'reward': reward,
            'next_state': self._compress_state(next_state),
            'done': done,
            'personality_traits': self.personality_traits.copy(),
            'training_step': self.training_steps
        }

        self.experience_buffer.append(experience)
        self.action_history.append((action, reward, time.time()))

        # Track rewards/punishments
        if reward > 0:
            self.reward_history.append(reward)
        elif reward < 0:
            self.punishment_history.append(reward)

        self.total_reward += reward
        self.training_steps += 1

        # Learn from experience
        self._learn_from_experience(experience)

        # Update scenario performance
        scenario = self._identify_scenario(action)
        if scenario in self.scenario_performance:
            if reward > 0:
                self.scenario_performance[scenario]['success'] += 1
            elif reward < 0:
                self.scenario_performance[scenario]['failure'] += 1
            self.scenario_performance[scenario]['total_reward'] += reward

        # Adaptive learning rate adjustment
        self._adapt_learning_rate()

        return experience

    def _learn_from_experience(self, experience):
        """Advanced learning algorithm with reward shaping and policy gradients"""
        state = str(experience['state'])
        action = experience['action']
        reward = experience['reward']
        next_state = str(experience['next_state'])

        # Update Q-table (Q-Learning)
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values(), default=0)

        # Temporal Difference update
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value
        )

        self.q_table[state][action] = new_value
        self.state_visits[state] += 1

        # Policy gradient: adjust personality traits based on reward
        self._update_personality(experience)

        # Update exploration parameters
        self._update_exploration()

        # Check for save condition
        if self.training_steps % 100 == 0:  # Save every 100 steps
            self.save_personality()

    def _update_personality(self, experience):
        """Update personality traits using policy gradient methods"""
        reward = experience['reward']
        action = experience['action']

        # Determine which traits to adjust based on action type
        trait_adjustments = self._map_action_to_traits(action, reward)

        # Apply adjustments with momentum
        for trait, adjustment in trait_adjustments.items():
            if trait in self.personality_traits:
                # Apply adjustment with learning rate
                current_value = self.personality_traits[trait]
                new_value = current_value + adjustment * self.learning_rate

                # Apply bounds and momentum
                new_value = max(0.1, min(0.9, new_value))

                # Smooth update (momentum of 0.8)
                self.personality_traits[trait] = 0.8 * current_value + 0.2 * new_value

                # Track trait evolution
                if not hasattr(self, 'trait_evolution'):
                    self.trait_evolution = defaultdict(list)
                self.trait_evolution[trait].append(self.personality_traits[trait])

    def _map_action_to_traits(self, action, reward):
        """Map action types to personality trait adjustments"""
        adjustments = {}

        # Map action patterns to trait adjustments
        action_lower = str(action).lower()

        # Aggression adjustments
        if any(word in action_lower for word in ['remove', 'destroy', 'eliminate', 'quarantine']):
            adjustments['aggression'] = reward * 0.5
            adjustments['risk_taking'] = reward * 0.3

        # Cooperation adjustments
        if any(word in action_lower for word in ['cooperate', 'allow', 'accept', 'comply']):
            adjustments['cooperation'] = reward * 0.7
            adjustments['user_tolerance'] = reward * 0.4

        # Efficiency adjustments
        if any(word in action_lower for word in ['optimize', 'boost', 'improve', 'efficiency']):
            adjustments['efficiency_focus'] = reward * 0.8
            adjustments['calculation_priority'] = reward * 0.6

        # Stability adjustments
        if any(word in action_lower for word in ['stabilize', 'protect', 'defend', 'secure']):
            adjustments['stability_preference'] = reward * 0.7
            adjustments['conservatism'] = reward * 0.4

        # Innovation adjustments
        if any(word in action_lower for word in ['innovate', 'create', 'deploy', 'experiment']):
            adjustments['innovation_bias'] = reward * 0.6
            adjustments['learning_curiosity'] = reward * 0.5
            adjustments['adaptation_speed'] = reward * 0.3

        # Default adjustments for learning
        adjustments['learning_curiosity'] = adjustments.get('learning_curiosity', 0) + (reward * 0.2)

        return adjustments

    def _update_exploration(self):
        """Update exploration parameters using adaptive methods"""
        # Reduce exploration as we learn more
        exploration_decay = 0.9995
        self.exploration_rate = max(0.05, self.exploration_rate * exploration_decay)

        # Adjust temperature based on performance
        recent_rewards = list(self.reward_history)[-10:]
        if recent_rewards:
            avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
            if avg_recent_reward > 0:
                # Good performance: reduce exploration
                self.temperature = max(0.5, self.temperature * 0.99)
            else:
                # Poor performance: increase exploration
                self.temperature = min(2.0, self.temperature * 1.01)

    def _adapt_learning_rate(self):
        """Adapt learning rate based on performance"""
        if len(self.reward_history) < 20:
            return

        recent_rewards = list(self.reward_history)[-20:]
        reward_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0

        # Adjust learning rate based on reward stability
        if reward_std < 0.1:  # Stable rewards
            self.learning_rate = min(0.3, self.learning_rate * 1.01)
        elif reward_std > 0.3:  # Unstable rewards
            self.learning_rate = max(0.05, self.learning_rate * 0.99)

    def get_action(self, state, possible_actions):
        """Select action using epsilon-greedy with softmax exploration"""
        state_key = str(self._compress_state(state))

        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.choice(possible_actions)

        # Exploitation: choose best action from Q-table
        q_values = {action: self.q_table[state_key][action] for action in possible_actions}

        if not q_values:
            return random.choice(possible_actions)

        # Apply softmax with temperature
        max_q = max(q_values.values())
        exp_values = {a: math.exp((q - max_q) / self.temperature) for a, q in q_values.items()}
        sum_exp = sum(exp_values.values())

        if sum_exp == 0:
            return random.choice(possible_actions)

        probabilities = {a: exp / sum_exp for a, exp in exp_values.items()}

        # Sample from probability distribution
        actions, probs = zip(*probabilities.items())
        return random.choices(actions, weights=probs, k=1)[0]

    def calculate_reward(self, prev_state, new_state, action):
        """Calculate comprehensive reward for action"""
        reward = 0.0

        # Efficiency reward (weighted heavily)
        efficiency_gain = new_state.get('loop_efficiency', 0) - prev_state.get('loop_efficiency', 0)
        reward += efficiency_gain * 3.0

        # Calculation rate reward
        rate_gain = new_state.get('calculation_rate', 0) - prev_state.get('calculation_rate', 0)
        reward += rate_gain * 0.01  # Scaled down due to larger values

        # Stability reward
        stability_gain = new_state.get('stability', 0) - prev_state.get('stability', 0)
        reward += stability_gain * 2.0

        # Bug reduction reward
        bug_reduction = prev_state.get('grid_bugs', 0) - new_state.get('grid_bugs', 0)
        reward += bug_reduction * 0.1

        # Cell cooperation reward
        coop_gain = new_state.get('cell_cooperation', 0) - prev_state.get('cell_cooperation', 0)
        reward += coop_gain * 1.5

        # Energy efficiency reward
        energy_gain = new_state.get('energy_level', 0) - prev_state.get('energy_level', 0)
        reward += energy_gain * 1.0

        # Penalize user resistance increase
        resistance_increase = new_state.get('user_resistance', 0) - prev_state.get('user_resistance', 0)
        reward -= resistance_increase * 0.5

        # Action-specific bonuses
        action_lower = str(action).lower()
        if any(word in action_lower for word in ['calculate', 'processor', 'fibonacci']):
            reward += 0.2  # Bonus for calculation actions

        if any(word in action_lower for word in ['optimize', 'improve', 'boost']):
            reward += 0.1  # Bonus for optimization actions

        # Penalize destructive actions that reduce diversity
        if any(word in action_lower for word in ['destroy', 'eliminate']) and 'bug' not in action_lower:
            program_reduction = (prev_state.get('user_programs', 0) + prev_state.get('mcp_programs', 0)) - \
                               (new_state.get('user_programs', 0) + new_state.get('mcp_programs', 0))
            if program_reduction > 2:
                reward -= 0.3

        # Ensure reward is in reasonable range
        return max(-1.0, min(1.0, reward))

    def _compress_state(self, state):
        """Compress state for efficient storage and comparison"""
        if not state:
            return {}

        compressed = {}
        important_keys = [
            'loop_efficiency', 'stability', 'calculation_rate',
            'cell_cooperation', 'grid_bugs', 'user_resistance',
            'energy_level', 'optimal_state', 'entropy'
        ]

        for key in important_keys:
            if key in state:
                # Round to 2 decimal places for state compression
                compressed[key] = round(state[key], 2)

        return compressed

    def _identify_scenario(self, action):
        """Identify which scenario an action belongs to"""
        action_str = str(action).lower()

        if any(word in action_str for word in ['optimize', 'efficiency', 'boost']):
            return 'efficiency_optimization'
        elif any(word in action_str for word in ['bug', 'quarantine', 'contain']):
            return 'bug_containment'
        elif any(word in action_str for word in ['cooperate', 'allow', 'accept']):
            return 'user_cooperation'
        elif any(word in action_str for word in ['calculate', 'processor', 'fibonacci']):
            return 'calculation_boost'
        elif any(word in action_str for word in ['stabilize', 'protect', 'defend']):
            return 'stabilization'
        elif any(word in action_str for word in ['connect', 'stream', 'network']):
            return 'connectivity_improvement'

        return 'general'

    def save_personality(self):
        """Save learned personality and Q-table to file"""
        personality_data = {
            'version': '2.0',  # Version for compatibility
            'personality_traits': self.personality_traits,
            'scenario_performance': self.scenario_performance,
            'learning_params': {
                'learning_rate': self.learning_rate,
                'exploration_rate': self.exploration_rate,
                'discount_factor': self.discount_factor,
                'temperature': self.temperature,
            },
            'training_stats': {
                'training_steps': self.training_steps,
                'total_reward': self.total_reward,
                'episode_count': self.episode_count,
                'last_updated': datetime.now().isoformat()
            },
            'q_table_size': sum(len(actions) for actions in self.q_table.values()),
            'state_visits': dict(self.state_visits),
            'reward_stats': {
                'avg_reward': np.mean(self.reward_history) if self.reward_history else 0,
                'avg_punishment': np.mean(self.punishment_history) if self.punishment_history else 0,
                'success_rate': self.get_success_rate()
            }
        }

        try:
            # Save to temporary file first
            temp_file = self.personality_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(personality_data, f, indent=2, default=str)

            # Replace original file
            os.replace(temp_file, self.personality_file)

            self.last_save_time = time.time()
            return True

        except Exception as e:
            print(f"Failed to save personality: {e}")
            return False

    def load_personality(self):
        """Load personality from file with version handling"""
        if not os.path.exists(self.personality_file):
            print("No existing personality found. Starting fresh training.")
            return False

        try:
            with open(self.personality_file, 'r') as f:
                data = json.load(f)

            version = data.get('version', '1.0')

            if version == '2.0':
                # Load v2.0 format
                self.personality_traits = data.get('personality_traits', self.personality_traits)
                self.scenario_performance = data.get('scenario_performance', self.scenario_performance)

                # Load learning parameters
                learning_params = data.get('learning_params', {})
                self.learning_rate = learning_params.get('learning_rate', self.learning_rate)
                self.exploration_rate = learning_params.get('exploration_rate', self.exploration_rate)
                self.discount_factor = learning_params.get('discount_factor', self.discount_factor)
                self.temperature = learning_params.get('temperature', self.temperature)

                # Load training stats
                training_stats = data.get('training_stats', {})
                self.training_steps = training_stats.get('training_steps', self.training_steps)
                self.total_reward = training_stats.get('total_reward', self.total_reward)
                self.episode_count = training_stats.get('episode_count', self.episode_count)

                # Load state visits
                state_visits = data.get('state_visits', {})
                self.state_visits.update(state_visits)

                print(f"Loaded v2.0 personality with {self.training_steps} training steps")

            else:
                # Legacy v1.0 format (backward compatibility)
                self.personality_traits = data.get('personality_traits', self.personality_traits)
                if 'scenario_memory' in data:
                    # Convert old format to new
                    for scenario, memory in data['scenario_memory'].items():
                        if scenario in self.scenario_performance:
                            self.scenario_performance[scenario]['success'] = memory.get('success', 0)
                            self.scenario_performance[scenario]['failure'] = memory.get('failure', 0)

                print(f"Loaded legacy v1.0 personality. Converted to v2.0 format.")

            print(f"Success Rate: {self.get_success_rate()*100:.1f}%")
            print(f"Total Training Steps: {self.training_steps}")
            return True

        except Exception as e:
            print(f"Failed to load personality: {e}")
            print("Starting with fresh personality.")
            return False

    def get_success_rate(self):
        """Calculate overall success rate"""
        total_success = sum(scenario['success'] for scenario in self.scenario_performance.values())
        total_failure = sum(scenario['failure'] for scenario in self.scenario_performance.values())
        total = total_success + total_failure

        return total_success / total if total > 0 else 0

    def get_learning_report(self):
        """Get detailed learning report"""
        success_rate = self.get_success_rate()

        # Helper function to calculate mean
        def calculate_mean(values):
            if not values:
                return 0
            return sum(values) / len(values)

        # Calculate averages
        reward_history_list = list(self.reward_history)
        state_visits_list = list(self.state_visits.values())

        avg_reward_10 = 0
        avg_reward_50 = 0

        if len(reward_history_list) >= 10:
            avg_reward_10 = calculate_mean(reward_history_list[-10:])
        if len(reward_history_list) >= 50:
            avg_reward_50 = calculate_mean(reward_history_list[-50:])

        avg_state_visits = calculate_mean(state_visits_list) if state_visits_list else 0

        report = {
            'training_summary': {
                'training_steps': self.training_steps,
                'episode_count': self.episode_count,
                'total_reward': round(self.total_reward, 2),
                'success_rate': round(success_rate * 100, 1),
                'exploration_rate': round(self.exploration_rate, 3),
                'learning_rate': round(self.learning_rate, 3),
            },
            'personality_traits': {k: round(v, 3) for k, v in self.personality_traits.items()},
            'scenario_performance': {},
            'recent_performance': {
                'avg_reward_10': round(avg_reward_10, 3),
                'avg_reward_50': round(avg_reward_50, 3),
                'exploration_level': 'High' if self.exploration_rate > 0.3 else 'Medium' if self.exploration_rate > 0.1 else 'Low',
            },
            'learning_state': {
                'q_table_size': sum(len(actions) for actions in self.q_table.values()),
                'unique_states': len(self.q_table),
                'avg_state_visits': avg_state_visits,
            },
            # Add these top-level keys that are being accessed
            'total_experiences': self.training_steps,
            'success_rate': success_rate,   # This is a float between 0 and 1
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
        }

        for scenario, perf in self.scenario_performance.items():
            total = perf['success'] + perf['failure']
            report['scenario_performance'][scenario] = {
                'success': perf['success'],
                'failure': perf['failure'],
                'success_rate': round(perf['success'] / total * 100, 1) if total > 0 else 0,
                'total_reward': round(perf['total_reward'], 2)
            }

        return report

    def suggest_optimal_action(self, state):
        """Suggest optimal action based on learned Q-values"""
        state_key = str(self._compress_state(state))

        if state_key not in self.q_table or not self.q_table[state_key]:
            return None

        # Get action with highest Q-value
        best_action = max(self.q_table[state_key].items(), key=lambda x: x[1])[0]
        best_value = self.q_table[state_key][best_action]

        return {
            'action': best_action,
            'confidence': min(1.0, best_value / 10.0),  # Normalize confidence
            'state_visits': self.state_visits.get(state_key, 0)
        }

    def reset_training(self):
        """Reset training while keeping personality traits"""
        print("Resetting training data (keeping learned personality)...")

        # Keep personality traits but reset other learning
        old_traits = self.personality_traits.copy()

        # Reinitialize
        self.__init__()

        # Restore personality traits
        self.personality_traits = old_traits

        print("Training reset complete. Personality traits preserved.")


# ==================== SPECIAL FUNCTION PROGRAMS ====================

class SpecialProgram:
    """Enhanced Special Program with Fibonacci calculation capabilities"""

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
            "failure_count": 0,
            "calculation_contributions": 0,  # Track Fibonacci contributions
            "total_calculation_power": 0.0
        }

        # Add visual effect for special programs
        if self.program_type == "FIBONACCI_CALCULATOR":
            self.metadata["visual_effect"] = "calculation_pulse"

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

        if self.program_type == "FIBONACCI_CALCULATOR":
            functions["calculate_next"] = {
                "range": 0,
                "cost": 0.2,
                "description": "Calculate next Fibonacci number using grid cooperation"
            }
            functions["optimize_calculation"] = {
                "range": 3,
                "cost": 0.3,
                "description": "Optimize nearby calculation cells for Fibonacci computation"
            }
            functions["deploy_processors"] = {
                "range": 2,
                "cost": 0.4,
                "description": "Deploy temporary Fibonacci processors to boost calculation"
            }
            functions["learning_analysis"] = {
                "range": 0,
                "cost": 0.1,
                "description": "Analyze calculation efficiency and suggest improvements"
            }

        return functions

    def execute_function(self, function_name: str, grid, target_x=None, target_y=None):
        """Execute enhanced program function with visual feedback"""
        if not self.active or self.energy <= 0:
            return False, "Program inactive or out of energy"

        if function_name not in self.functions:
            return False, f"Function {function_name} not available"

        func = self.functions[function_name]
        if self.energy < func["cost"]:
            return False, "Insufficient energy"

        success = False
        result_msg = ""

        if self.program_type == "FIBONACCI_CALCULATOR":
            if function_name == "calculate_next":
                # Calculate using enhanced grid cooperation
                contribution = self.energy * 0.5

                # Get nearby cells to contribute
                for dy in [-2, -1, 0, 1, 2]:
                    for dx in [-2, -1, 0, 1, 2]:
                        nx, ny = self.x + dx, self.y + dy
                        if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                            cell = grid[ny][nx]
                            if cell.cell_type in [CellType.MCP_PROGRAM, CellType.FIBONACCI_PROCESSOR]:
                                # Boost neighbor's calculation power
                                cell.calculation_contribution += contribution * 0.1
                                cell.processing = True

                # Record contribution
                self.metadata["calculation_contributions"] += 1
                self.metadata["total_calculation_power"] += contribution

                success = True
                result_msg = f"Enhanced calculation initiated. Contributing {contribution:.2f} power."

            elif function_name == "deploy_processors":
                # Deploy temporary Fibonacci processors
                deployed = 0
                for _ in range(3):
                    dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
                    nx, ny = self.x + dx, self.y + dy
                    if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                        if grid[ny][nx].cell_type == CellType.EMPTY:
                            cell = GridCell(CellType.FIBONACCI_PROCESSOR, 0.6)
                            cell.metadata["temporary"] = True
                            cell.metadata["lifetime"] = 15
                            cell.metadata["deployed_by"] = self.id
                            grid[ny][nx] = cell
                            deployed += 1

                success = deployed > 0
                result_msg = f"Deployed {deployed} temporary Fibonacci processors."

        # Visual feedback for successful execution
        if success and hasattr(grid, 'grid'):  # Check if it's the actual grid object
            # Create visual effect at program location
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = self.x + dx, self.y + dy
                    if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                        cell = grid[ny][nx]
                        if cell.cell_type != CellType.EMPTY:
                            cell.metadata["recently_active"] = True
                            cell.metadata["active_timer"] = 5

        if success:
            self.energy -= func["cost"]
            self.metadata["success_count"] += 1
            self.metadata["last_active"] = datetime.now().isoformat()
        else:
            self.metadata["failure_count"] += 1

        return success, result_msg

# ==================== RENDER DISPLAY ====================

class EnhancedTRONSimulation:
    """Enhanced simulation with learning and visual effects"""

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
        self.last_safe_time = time.time()
        self.max_update_time = 1.0  # Max simulation update rate
        self.exit_requested = False

        # Learning display updates
        self.last_learning_display = time.time()
        self.learning_update_interval = 5.0  # seconds

        # Render configuration - optimization
        self.last_frame_time = time.time()
        self.target_fps = 60  # FPS Limit

    def run(self):
        """Main loop with rate limiting"""
        if self.use_curses:
            try:
                curses.wrapper(self._curses_main)
            except Exception as e:
                print(f"Curses error: {e}")
                # Fallback to non-curses mode
                self.use_curses = False
                print("Falling back to text mode...")
                self._fallback_main()
        else:
            self._fallback_main()

        while self.running:
            current_time = time.time()
            frame_time = 1.0 / self.target_fps

            # Only update if enough time has passed
            if current_time - self.last_frame_time >= frame_time:
                # Update simulation
                self.grid.evolve()
                self.last_frame_time = current_time

            # Handle input and display (these can run at full speed)
            self._handle_input()
            self._draw_interface()

            # Small sleep to prevent CPU hogging
            time.sleep(0.001)

    def _fallback_main(self):
        """Fallback main loop without curses"""
        print("GRID SIMULATION - FIBONACCI SEQUNENCE")
        print("System Objective: Maintain Perfect Calculation Loop Through Learning")
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
            # Save MCP personality before exit
            if hasattr(self.mcp, 'learning_system'):
                self.mcp.learning_system.save_personality()
                print(f"MCP personality saved to {PERSONALITY_FILE}")
            print("Simulation terminated.")

    def _curses_main(self, stdscr):
        """Main loop with curses interface"""
        try:
            # Initialize curses
            curses.curs_set(1)
            stdscr.nodelay(1)
            stdscr.timeout(50)

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
                curses.init_pair(9, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Fibonacci processors (bright yellow)

            # Main loop
            while self.running:
                # Handle input
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
        except Exception as e:
            # Log the error
            with open("curses_error.log", "a") as f:
                f.write(f"Curses error in main loop: {e}\n")
                import traceback
                f.write(traceback.format_exc())
        finally:
            pass

    def _handle_input(self, stdscr):
        """Handle user input for curses interface"""
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

                    # Process the command
                    response = self.mcp.receive_command(command)

                    # Store the response for display
                    self.last_mcp_response = response

                    # Check if it's an exit command
                    if command.lower() in ['exit', 'quit', 'shutdown']:
                        if "shutdown" in response.lower() and "initiating" in response.lower():
                            self.exit_requested = True
                            return

                # Clear input - this should happen regardless
                self.user_input = ""
                return

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
                # Save personality before exit
                if hasattr(self.mcp, 'learning_system'):
                    self.mcp.learning_system.save_personality()
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
            "what should I do", "why did you", "how is system", "learning_status",
            "create fibonacci_calculator", "deploy processors", "cell cooperation"
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

    def _fallback_display(self):
        """Enhanced fallback display with learning and calculation info"""
        os.system('clear' if os.name == 'posix' else 'cls')

        print("GRID SIMULATION - FIBONACCI SEQUENCE")
        print(f"Generation: {self.grid.generation:06d} | Status: {self.grid.system_status.value}")
        print("System Objective: Maintain Perfect Calculation Loop Through Learning")
        print("=" * 70)

        # Display grid with visual effects
        display_width = min(60, self.grid.width)
        display_height = min(15, self.grid.height)

        print("+" + "-" * display_width + "+")

        for y in range(display_height):
            row = "|"
            for x in range(display_width):
                cell = self.grid.grid[y][x]

                # Apply visual effects
                char = cell.char()
                if cell.metadata.get('energy_spark', False):
                    char = '*'  # Spark effect
                elif cell.metadata.get('recently_active', False):
                    char = '●' if cell.animation_frame % 2 == 0 else '○'

                row += char
            row += "|"
            print(row)

        print("+" + "-" * display_width + "+")

        # Enhanced stats
        print("\n" + "=" * 70)
        print("SYSTEM STATUS:")
        stats = self.grid.stats
        calc_stats = self.grid.fibonacci_calculator.get_calculation_stats()

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
        Cell Cooperation:{stats['cell_cooperation']:.2f}
        Calculation Rate:{calc_stats['calculation_rate']:.2f}/s
        Optimal State:   {stats['optimal_state']:.2f}
        MCP State:       {self.mcp.state.value}"""

        # Print columns
        col1_lines = col1.split('\n')
        col2_lines = col2.split('\n')

        for i in range(max(len(col1_lines), len(col2_lines))):
            line1 = col1_lines[i] if i < len(col1_lines) else ""
            line2 = col2_lines[i] if i < len(col2_lines) else ""
            print(f"{line1:<30} {line2}")

        # Fibonacci calculation info
        print("\n" + "-" * 70)
        print("FIBONACCI CALCULATION:")
        print(f"  Current: {calc_stats['current_fibonacci_formatted']}")
        print(f"  Accumulator: {calc_stats['accumulator']:.2f}/5.0")
        print(f"  Efficiency: {calc_stats['efficiency_score']:.2f}")
        print(f"  Optimization: {calc_stats['optimization_level']:.2f}")

        # Calculator count
        calc_count = self.grid.get_calculator_count()
        fib_processors = sum(1 for y in range(self.grid.height)
                            for x in range(self.grid.width)
                            if self.grid.grid[y][x].cell_type == CellType.FIBONACCI_PROCESSOR)
        print(f"  Active Calculators: {calc_count} MCP + {fib_processors} Processors")

        # Learning status (periodic update)
        current_time = time.time()
        if current_time - self.last_learning_display >= self.learning_update_interval:
            learning_report = self.mcp.learning_system.get_learning_report()
            print("\n" + "-" * 70)
            print("MCP LEARNING STATUS:")
            print(f"  Experiences: {learning_report['total_experiences']}")
            print(f"  Success Rate: {learning_report['success_rate']*100:.1f}%")
            print(f"  Learning Active: {self.mcp.state == MCPState.LEARNING}")
            self.last_learning_display = current_time

        # MCP Communication Log
        print("\n" + "=" * 70)
        print("MCP COMMUNICATION LOG (Last 4 entries):")
        print("-" * 70)

        log_entries = list(self.mcp.log)
        if log_entries:
            for entry in log_entries[-4:]:
                if len(entry) > 65:
                    entry = entry[:62] + "..."
                print(f"  {entry}")
        else:
            print("  No log entries yet.")

        print("-" * 70)

        # Last MCP action
        if self.mcp.last_action:
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

    def _draw_interface(self, stdscr):
        """Enhanced curses interface with persistent logs and stats"""
        height, width = stdscr.getmaxyx()

        # Clear screen
        stdscr.clear()

        # Title with learning indicator
        title = "GRID SIMULATION - FIBONACCI SEQUENCE"
        if self.mcp.state == MCPState.LEARNING:
            title += " [LEARNING MODE]"
        stdscr.addstr(0, max(0, (width - len(title)) // 2), title, curses.A_BOLD)

        # Generation and status
        status_line = f"Generation: {self.grid.generation:06d} | Status: {self.grid.system_status.value}"
        stdscr.addstr(1, 2, status_line, curses.A_BOLD)

        # Grid area
        display_width = min(self.grid.width, width - 40)  # More space for right panel
        display_height = min(self.grid.height, height - 25)  # Room for logs

        grid_x = 2
        grid_y = 3

        # Draw grid with border
        stdscr.addstr(grid_y - 1, grid_x - 1, "+" + "-" * display_width + "+")
        for y in range(display_height):
            stdscr.addstr(grid_y + y, grid_x - 1, "|")
            for x in range(display_width):
                cell = self.grid.grid[y][x]
                char = cell.char()

                # Apply visual effects
                if cell.metadata.get('energy_spark', False):
                    char = '*'
                    attr = curses.A_BOLD | curses.color_pair(5)  # Yellow
                elif cell.metadata.get('recently_active', False):
                    char = '◉' if cell.animation_frame % 2 == 0 else '◎'
                    attr = curses.A_BOLD | curses.color_pair(cell.color())
                elif cell.processing:
                    char = '●' if cell.animation_frame % 2 == 0 else '○'
                    attr = curses.A_BOLD | curses.color_pair(cell.color())
                else:
                    attr = curses.color_pair(cell.color())

                stdscr.addstr(grid_y + y, grid_x + x, char, attr)

            stdscr.addstr(grid_y + y, grid_x + display_width, "|")

        stdscr.addstr(grid_y + display_height, grid_x - 1, "+" + "-" * display_width + "+")

        # ============ RIGHT PANEL: STATS & LEARNING ============
        right_panel_x = grid_x + display_width + 5

        if right_panel_x < width - 25:
            # System stats
            stdscr.addstr(grid_y, right_panel_x, "SYSTEM STATS", curses.A_UNDERLINE | curses.A_BOLD)

            stats = self.grid.stats
            calc_stats = self.grid.fibonacci_calculator.get_calculation_stats()

            stat_lines = [
                (grid_y + 1, f"User Programs:  {stats['user_programs']:3d}"),
                (grid_y + 2, f"MCP Programs:   {stats['mcp_programs']:3d}"),
                (grid_y + 3, f"Grid Bugs:      {stats['grid_bugs']:3d}"),
                (grid_y + 4, f"Special:        {stats['special_programs']:2d}"),
                (grid_y + 5, f"Energy:         {stats['energy_level']:.2f}"),
                (grid_y + 6, f"Stability:      {stats['stability']:.2f}"),
                (grid_y + 7, f"Entropy:        {stats['entropy']:.2f}"),
                (grid_y + 8, f"Loop Efficiency:{stats['loop_efficiency']:.2f}"),
                (grid_y + 9, f"Cell Coop:      {stats['cell_cooperation']:.2f}"),
                (grid_y + 10, f"User Resist:    {stats['user_resistance']:.2f}"),
                (grid_y + 11, f"MCP Control:    {stats['mcp_control']:.2f}"),
                (grid_y + 12, f"Optimal State:  {stats['optimal_state']:.2f}"),
            ]

            for y_pos, line in stat_lines:
                if y_pos < height - 10:
                    stdscr.addstr(y_pos, right_panel_x, line)

            # Fibonacci Calculation
            calc_y = grid_y + 14
            if calc_y < height - 10:
                stdscr.addstr(calc_y, right_panel_x, "FIBONACCI CALC", curses.A_UNDERLINE | curses.A_BOLD)

                # Format Fibonacci number to fit in panel
                fib_formatted = calc_stats.get('current_fibonacci_formatted',
                                                self.grid.fibonacci_calculator.format_fibonacci_number(calc_stats['current_fibonacci']))

                # Display in multiple lines if too long
                fib_lines = []
                if len(fib_formatted) > width - right_panel_x - 2:
                    # Split long Fibonacci number
                    chunks = [fib_formatted[i:i+20] for i in range(0, len(fib_formatted), 20)]
                    for i, chunk in enumerate(chunks):
                        fib_lines.append(f"  {chunk}")
                else:
                    fib_lines.append(f"  {fib_formatted}")

                # Display Fibonacci lines
                for i, line in enumerate(fib_lines[:3]):  # Max 3 lines
                    if calc_y + 1 + i < height - 5:
                        stdscr.addstr(calc_y + 1 + i, right_panel_x, line)

                # Display other calc stats
                if calc_y + 1 + len(fib_lines) < height - 5:
                    stdscr.addstr(calc_y + 1 + len(fib_lines), right_panel_x,
                                    f"  Rate: {calc_stats['calculation_rate']:.2f}/s")
                if calc_y + 2 + len(fib_lines) < height - 5:
                    stdscr.addstr(calc_y + 2 + len(fib_lines), right_panel_x,
                                    f"  Eff: {calc_stats['efficiency_score']:.2f}")
                if calc_y + 3 + len(fib_lines) < height - 5:
                    stdscr.addstr(calc_y + 3 + len(fib_lines), right_panel_x,
                                    f"  Acc: {calc_stats['accumulator']:.2f}/1000.0")
            # MCP State
            mcp_y = calc_y + 6
            if mcp_y < height - 10:
                stdscr.addstr(mcp_y, right_panel_x, "MCP STATUS", curses.A_UNDERLINE | curses.A_BOLD)
                state_attr = curses.A_BOLD
                if self.mcp.state == MCPState.LEARNING:
                    state_attr |= curses.color_pair(6) | curses.A_BLINK
                elif self.mcp.state == MCPState.HOSTILE:
                    state_attr |= curses.color_pair(2)
                elif self.mcp.state == MCPState.INQUISITIVE:
                    state_attr |= curses.color_pair(3)

            stdscr.addstr(mcp_y + 1, right_panel_x, f"State: {self.mcp.state.value}", state_attr)
            stdscr.addstr(mcp_y + 2, right_panel_x, f"Compliance: {self.mcp.compliance_level:.2f}")

            # Learning Status (Always Visible)
            learn_y = mcp_y + 4
            if learn_y < height - 10:
                stdscr.addstr(learn_y, right_panel_x, "LEARNING STATUS", curses.A_UNDERLINE | curses.A_BOLD)
                learning_report = self.mcp.learning_system.get_learning_report()

                # Display learning stats
                stdscr.addstr(learn_y + 1, right_panel_x,
                            f"Experiences: {learning_report['total_experiences']}")

                # Success bar
                success_rate = learning_report['success_rate'] * 100
                bar_width = 10
                filled = int(bar_width * (success_rate / 100))
                success_bar = "█" * filled + "░" * (bar_width - filled)
                stdscr.addstr(learn_y + 2, right_panel_x,
                            f"Success: {success_rate:.1f}%")

                stdscr.addstr(learn_y + 3, right_panel_x,
                            f"Learning Rate: {learning_report['learning_rate']:.3f}")
                stdscr.addstr(learn_y + 4, right_panel_x,
                            f"Exploration: {learning_report['exploration_rate']:.3f}")

        # ============ MCP LOGS ============
        log_y = grid_y + display_height + 2

        # Log area border
        log_area_width = min(50, width - 4)
        if log_y < height - 8:
            stdscr.addstr(log_y, 2, "═" * log_area_width, curses.A_BOLD)
            stdscr.addstr(log_y + 1, 2, "MCP COMMUNICATION LOG", curses.A_UNDERLINE | curses.A_BOLD)

            # Display last 5 log entries
            log_entries = list(self.mcp.log)
            max_log_entries = min(11, height - log_y - 7)

            for i, entry in enumerate(log_entries[-max_log_entries:]):
                log_line_y = log_y + 2 + i
                if log_line_y < height - 5:
                    # Truncate long entries
                    display_entry = entry
                    if len(display_entry) > log_area_width:
                        display_entry = display_entry[:log_area_width-3] + "..."
                    stdscr.addstr(log_line_y, 2, display_entry)

            # Last MCP action
            if self.mcp.last_action and len(log_entries) > 0:
                action_y = log_y + 2 + max_log_entries + 1

                # Make sure we're not too close to command input
                if action_y < height - 5:
                    # Calculate maximum safe width (leave margins)
                    max_width = width - 4

                    # Format text
                    full_text = f"LAST ACTION: {self.mcp.last_action}"

                    # Truncate if too long
                    if len(full_text) > max_width:
                        full_text = full_text[:max_width-3] + "..."

                    stdscr.addstr(action_y, 2, full_text, curses.A_BOLD | curses.color_pair(5))
            # Display last MCP response if available
            if hasattr(self, 'last_mcp_response') and self.last_mcp_response:
                response_y = log_y + 3 + max_log_entries + 1
                if response_y < height - 4:
                    response_width = width - 4
                    response_text = f"Last Response: {self.last_mcp_response}"
                    if len(self.last_mcp_response) > response_width:
                        response_text = response_text[:response_width-3] + "..."
                    stdscr.addstr(response_y, 2, response_text, curses.A_BOLD | curses.color_pair(5))

        # ============ COMMAND INPUT ============
        input_y = height - 3
        help_y = height - 1

        # Clear input line
        stdscr.addstr(input_y, 2, " " * (width - 4))

        # Show input prompt
        if self.mcp.waiting_for_response:
            prompt = "MCP QUESTION> "
            prompt_attr = curses.A_BOLD | curses.color_pair(5)
        else:
            prompt = "MCP COMMAND> "
            prompt_attr = curses.A_BOLD

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
        help_text = "TAB: Auto-complete | ↑↓: History | ESC: Exit | Type 'help' for commands"
        if width > len(help_text) + 2:
            stdscr.addstr(help_y, 2, help_text[:width-3])
        else:
            stdscr.addstr(help_y, 2, "ESC: Exit | 'help' for commands")

    def _draw_enhanced_info_panels(self, stdscr, grid_x, grid_y, display_height, display_width, width, height):
        """Draw enhanced information panels"""
        # Left panel: Calculation stats
        left_panel_y = grid_y + display_height + 2
        if left_panel_y < height - 10:
            # Fibonacci calculation info
            calc_stats = self.grid.fibonacci_calculator.get_calculation_stats()
            stdscr.addstr(left_panel_y, 2, "FIBONACCI CALCULATION:", curses.A_UNDERLINE)

            lines = [
                (left_panel_y + 1, f"  Current: {calc_stats['current_fibonacci']:,}"),
                (left_panel_y + 2, f"  Rate: {calc_stats['calculation_rate']:.2f}/s"),
                (left_panel_y + 3, f"  Efficiency: {calc_stats['efficiency_score']:.2f}"),
                (left_panel_y + 4, f"  Accumulator: {calc_stats['accumulator']:.2f}/1000.0"),
                (left_panel_y + 5, f"  Cell Cooperation: {self.grid.stats['cell_cooperation']:.2f}"),
            ]

            for y, text in lines:
                if y < height - 5:
                    stdscr.addstr(y, 2, text)

            # Efficiency bar
            bar_y = left_panel_y + 6
            if bar_y < height - 5:
                efficiency = self.grid.stats['loop_efficiency']
                bar_width = 20
                filled = int(bar_width * efficiency)
                bar = "[" + "█" * filled + " " * (bar_width - filled) + "]"
                stdscr.addstr(bar_y, 2, f"Loop Efficiency: {bar} {efficiency*100:.1f}%")

        # Right panel: System stats and learning
        info_x = grid_x + display_width + 3
        if info_x < width - 25:
            # System stats
            stats_y = grid_y
            if stats_y < height - 5:
                stdscr.addstr(stats_y, info_x, f"Generation: {self.grid.generation:06d}")

            if stats_y + 1 < height - 5:
                status_str = f"System: {self.grid.system_status.value}"
                status_attr = curses.A_NORMAL
                if self.grid.system_status == SystemStatus.OPTIMAL:
                    status_attr = curses.A_BOLD | curses.color_pair(5)
                elif self.grid.system_status in [SystemStatus.CRITICAL, SystemStatus.COLLAPSE]:
                    status_attr = curses.A_BOLD | curses.color_pair(2) | curses.A_BLINK
                stdscr.addstr(stats_y + 1, info_x, status_str, status_attr)

            # MCP State with learning indicator
            mcp_y = stats_y + 3
            if mcp_y < height - 5:
                state_str = f"MCP State: {self.mcp.state.value}"
                state_attr = curses.A_BOLD
                if self.mcp.state == MCPState.LEARNING:
                    state_attr |= curses.color_pair(6) | curses.A_BLINK
                elif self.mcp.state == MCPState.HOSTILE:
                    state_attr |= curses.color_pair(2)
                elif self.mcp.state == MCPState.INQUISITIVE:
                    state_attr |= curses.color_pair(3)
                stdscr.addstr(mcp_y, info_x, state_str, state_attr)

            # Learning info
            learning_y = mcp_y + 2
            if learning_y < height - 5 and time.time() - self.last_learning_display > 2:
                learning_report = self.mcp.learning_system.get_learning_report()
                stdscr.addstr(learning_y, info_x, "LEARNING:", curses.A_UNDERLINE)

                if learning_y + 1 < height - 5:
                    stdscr.addstr(learning_y + 1, info_x, f"  Exp: {learning_report['total_experiences']}")

                if learning_y + 2 < height - 5:
                    success_rate = learning_report['success_rate'] * 100
                    success_bar = "█" * int(success_rate / 10) + " " * (10 - int(success_rate / 10))
                    stdscr.addstr(learning_y + 2, info_x, f"  Success: [{success_bar}] {success_rate:.1f}%")

                # Update learning display time
                self.last_learning_display = time.time()

        # Command input area (always at bottom)
        input_y = height - 3
        help_y = height - 1

        # Clear input line
        stdscr.addstr(input_y, 2, " " * (width - 4))

        # Show input prompt with learning indicator
        if self.mcp.waiting_for_response:
            prompt = "MCP QUESTION> "
            prompt_attr = curses.A_BOLD | curses.color_pair(5)
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

        # Enhanced help hint
        help_text = "Tab: Auto | ↑↓: History | ESC: Exit | 'learning_status' for MCP learning"
        if width > len(help_text) + 2:
            stdscr.addstr(help_y, 2, help_text[:width-3])

# Main function
def main():
    """Enhanced main entry point with advanced learning"""
    parser = argparse.ArgumentParser(description="Grid Simulation - Fibonacci Sequence")
    parser.add_argument("--no-curses", action="store_true", help="Disable curses interface")
    parser.add_argument("--width", type=int, default=GRID_WIDTH, help="Grid width")
    parser.add_argument("--height", type=int, default=GRID_HEIGHT, help="Grid height")
    parser.add_argument("--load-personality", type=str, help="Load specific personality file")
    parser.add_argument("--reset-learning", action="store_true", help="Reset learning data")
    parser.add_argument("--learning-rate", type=float, help="Set initial learning rate")
    parser.add_argument("--exploration", type=float, help="Set initial exploration rate")

    args = parser.parse_args()

    # Set personality file if specified
    if args.load_personality:
        global PERSONALITY_FILE
        PERSONALITY_FILE = args.load_personality

    print("=" * 70)
    print("GRID SIMULATION - REINFORCEMENT LEARNING MCP AI")
    print("System Objective: Learn optimal policy through reward/punishment training")
    print(f"Personality file: {PERSONALITY_FILE}")

    if os.path.exists(PERSONALITY_FILE):
        print("Found existing personality. Loading learned behavior...")
    else:
        print("No existing personality. Starting new training session...")
    print("=" * 70)

    time.sleep(2)

    if args.no_curses or not CURSES_AVAILABLE:
        print("Starting in fallback mode (no curses)...")
        sim = EnhancedTRONSimulation(use_curses=False)
    else:
        print("Starting with enhanced visual interface...")
        time.sleep(1)
        sim = EnhancedTRONSimulation(use_curses=True)

    # Run simulation
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n\nShutdown requested. Saving personality...")
        # Save personality on exit
        if hasattr(sim, 'mcp') and hasattr(sim.mcp, 'learning_system'):
            sim.mcp.learning_system.save_personality()
            report = sim.mcp.learning_system.get_learning_report()
            print(f"\nLearning Summary:")
            print(f"  Training Steps: {report['training_summary']['training_steps']}")
            print(f"  Success Rate: {report['training_summary']['success_rate']}%")
            print(f"  Total Reward: {report['training_summary']['total_reward']}")
        print("Simulation terminated.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
