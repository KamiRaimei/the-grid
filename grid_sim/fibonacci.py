"""Fibonacci calculation engine for grid cell cooperation."""

import time
import random
import threading
from queue import Queue
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Callable
from datetime import datetime
import math

from .cells import CellType, GridCell


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
        for x, y, contribution in contributions:
            cell = self.grid.grid[y][x]
            if cell.cell_type == CellType.FIBONACCI_PROCESSOR and contribution > 0.3:
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

