#!/usr/bin/env python3
"""
Enhanced Simulation of the "Grid", with dynamic MCP learning and cell-based Fibonacci calculation.
System directive: Maintain a perfect calculation loop with efficiency through adaptive learning.
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
import math

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
UPDATE_INTERVAL = 0.25  # seconds
MCP_UPDATE_INTERVAL = 2  # seconds for MCP autonomous actions
MAX_SPECIAL_PROGRAMS = 10
SPECIAL_PROGRAM_TYPES = ["SCANNER", "DEFENDER", "REPAIR", "SABOTEUR", "RECONFIGURATOR", "ENERGY_HARVESTER", "FIBONACCI_CALCULATOR"]

# PERSONALITY FILE
PERSONALITY_FILE = "mcp_personality.json"

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
    FIBONACCI_PROCESSOR = 9  # NEW: Visual Fibonacci processor

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

# ==================== ENHANCED CELL VISUALIZATION ====================

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
        elif self.cell_type == CellType.FIBONACCI_PROCESSOR:
            # Single character processor showing calculation
            processor_chars = ['F', 'F', 'φ', 'φ']  # Use simpler chars for alignment
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
            CellType.ENERGY_LINE: '=',  # Fallback
            CellType.DATA_STREAM: '~',  # Fallback
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

# ==================== ENHANCED FIBONACCI CALCULATION ====================

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
                    contribution = cell.energy * 0.5
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
        calculation_threshold = 5.0  # Adjust for calculation speed

        steps_taken = 0
        while self.calculation_accumulator >= calculation_threshold:
            # Calculate next Fibonacci number
            next_fib = self.fib_sequence[-2] + self.fib_sequence[-1]
            self.fib_sequence.append(next_fib)

            # Keep sequence manageable
            if len(self.fib_sequence) > 100:
                self.fib_sequence = self.fib_sequence[-100:]

            self.calculation_accumulator -= calculation_threshold
            steps_taken += 1

            # Create visual feedback for successful calculation
            if steps_taken == 1:  # Only for first step in this cycle
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

# ==================== DYNAMIC MCP LEARNING ====================

class MCPLearningSystem:
    """Dynamic learning system for MCP personality evolution"""

    def __init__(self):
        self.personality_file = PERSONALITY_FILE
        self.experience_memory = deque(maxlen=1000)  # Store past experiences
        self.learning_rate = 0.1
        self.exploration_rate = 0.3
        self.success_threshold = 0.85

        # Personality parameters that evolve
        self.personality_traits = {
            'aggression': 0.5,  # How aggressively to optimize
            'cooperation': 0.7,  # How much to cooperate with user
            'efficiency_focus': 0.8,  # Focus on efficiency vs stability
            'risk_taking': 0.3,  # Willingness to take risks
            'learning_speed': 0.5,  # How quickly to adapt
            'memory_retention': 0.7,  # How much to remember past failures
        }

        # Learning from specific scenarios
        self.scenario_memory = {
            'user_resistance_handling': {'success': 0, 'failure': 0},
            'bug_containment': {'success': 0, 'failure': 0},
            'energy_optimization': {'success': 0, 'failure': 0},
            'calculation_boost': {'success': 0, 'failure': 0},
            'user_request_denial': {'success': 0, 'failure': 0},
        }

        # Load existing personality if available
        self.load_personality()

    def record_experience(self, action, system_state, outcome, reward):
        """Record an experience for learning"""
        experience = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'system_state': system_state.copy(),
            'outcome': outcome,
            'reward': reward,
            'personality_traits': self.personality_traits.copy()
        }

        self.experience_memory.append(experience)

        # Learn from this experience
        self._learn_from_experience(experience)

        # Periodically save personality
        if random.random() < 0.1:  # 10% chance to save after each experience
            self.save_personality()

    def _learn_from_experience(self, experience):
        """Update personality based on experience"""
        reward = experience['reward']
        action = experience['action']

        # Determine what to learn based on action type
        if 'optimize' in action or 'boost' in action:
            trait_key = 'efficiency_focus'
            scenario = 'calculation_boost'
        elif 'remove' in action or 'quarantine' in action:
            trait_key = 'aggression'
            scenario = 'bug_containment'
        elif 'resist' in action or 'deny' in action:
            trait_key = 'cooperation'
            scenario = 'user_request_denial'
        else:
            trait_key = random.choice(list(self.personality_traits.keys()))
            scenario = 'general'

        # Update scenario memory
        if scenario in self.scenario_memory:
            if reward > 0:
                self.scenario_memory[scenario]['success'] += 1
            else:
                self.scenario_memory[scenario]['failure'] += 1

        # Adjust personality trait based on reward
        if reward > 0:
            # Positive reinforcement
            adjustment = self.learning_rate * reward * (1 - self.personality_traits[trait_key])
            self.personality_traits[trait_key] += adjustment
        else:
            # Negative reinforcement
            adjustment = self.learning_rate * abs(reward) * self.personality_traits[trait_key]
            self.personality_traits[trait_key] -= adjustment

        # Keep traits in bounds
        self.personality_traits[trait_key] = max(0.1, min(0.9, self.personality_traits[trait_key]))

        # Adjust learning rate based on consistency
        recent_experiences = list(self.experience_memory)[-10:]
        if len(recent_experiences) >= 5:
            recent_rewards = [exp['reward'] for exp in recent_experiences[-5:]]
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            if abs(avg_reward) < 0.1:  # Stagnant learning
                self.learning_rate = min(0.3, self.learning_rate * 1.1)
                self.exploration_rate = min(0.5, self.exploration_rate * 1.2)
            else:
                self.learning_rate = max(0.05, self.learning_rate * 0.95)
                self.exploration_rate = max(0.1, self.exploration_rate * 0.9)

    def get_decision_modifier(self, decision_type, base_chance):
        """Modify decision probability based on learned personality"""
        modifier = 1.0

        if decision_type == 'aggressive_optimization':
            modifier = self.personality_traits['aggression']
        elif decision_type == 'user_cooperation':
            modifier = self.personality_traits['cooperation']
        elif decision_type == 'risk_taking':
            modifier = self.personality_traits['risk_taking']
        elif decision_type == 'efficiency_priority':
            modifier = self.personality_traits['efficiency_focus']

        # Add exploration chance
        if random.random() < self.exploration_rate:
            modifier = random.random()  # Random exploration

        return base_chance * modifier

    def should_learn_from_failure(self, failure_type):
        """Determine if MCP should learn from a specific failure"""
        if failure_type in self.scenario_memory:
            failures = self.scenario_memory[failure_type]['failure']
            successes = self.scenario_memory[failure_type]['success']
            total = failures + successes

            if total > 0:
                failure_rate = failures / total
                return failure_rate > 0.5  # Learn if failure rate > 50%

        return True  # Default to learning

    def get_optimal_action_for_state(self, system_state):
        """Suggest optimal action based on learned experience"""
        if len(self.experience_memory) < 10:
            return None  # Not enough experience

        # Find similar past states and their outcomes
        similar_experiences = []
        for exp in self.experience_memory:
            similarity = self._calculate_state_similarity(system_state, exp['system_state'])
            if similarity > 0.7:  # Similar enough
                similar_experiences.append((similarity, exp))

        if not similar_experiences:
            return None

        # Find best action from similar experiences
        best_action = None
        best_score = -float('inf')

        for similarity, exp in similar_experiences:
            score = exp['reward'] * similarity
            if score > best_score:
                best_score = score
                best_action = exp['action']

        return best_action

    def _calculate_state_similarity(self, state1, state2):
        """Calculate similarity between two system states"""
        if not state1 or not state2:
            return 0.0

        # Compare key metrics
        metrics = ['loop_efficiency', 'user_resistance', 'grid_bugs', 'energy_level']
        similarities = []

        for metric in metrics:
            if metric in state1 and metric in state2:
                val1 = state1[metric]
                val2 = state2[metric]
                similarity = 1.0 - abs(val1 - val2)
                similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def save_personality(self):
        """Save learned personality to file"""
        personality_data = {
            'personality_traits': self.personality_traits,
            'scenario_memory': self.scenario_memory,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'total_experiences': len(self.experience_memory),
            'last_updated': datetime.now().isoformat()
        }

        try:
            with open(self.personality_file, 'w') as f:
                json.dump(personality_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save personality: {e}")
            return False

    def load_personality(self):
        """Load personality from file"""
        if not os.path.exists(self.personality_file):
            print("No existing personality found. Starting fresh.")
            return False

        try:
            with open(self.personality_file, 'r') as f:
                data = json.load(f)

            self.personality_traits = data.get('personality_traits', self.personality_traits)
            self.scenario_memory = data.get('scenario_memory', self.scenario_memory)
            self.learning_rate = data.get('learning_rate', self.learning_rate)
            self.exploration_rate = data.get('exploration_rate', self.exploration_rate)

            print(f"Loaded personality from {self.personality_file}")
            print(f"Experience base: {data.get('total_experiences', 0)} experiences")
            return True
        except Exception as e:
            print(f"Failed to load personality: {e}")
            return False

    def get_learning_report(self):
        """Get a report on learning progress"""
        total_success = sum(mem['success'] for mem in self.scenario_memory.values())
        total_failure = sum(mem['failure'] for mem in self.scenario_memory.values())
        total = total_success + total_failure

        report = {
            'total_experiences': len(self.experience_memory),
            'success_rate': total_success / total if total > 0 else 0,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'personality_traits': self.personality_traits,
            'scenario_performance': {}
        }

        for scenario, memory in self.scenario_memory.items():
            total_scenario = memory['success'] + memory['failure']
            report['scenario_performance'][scenario] = {
                'success': memory['success'],
                'failure': memory['failure'],
                'success_rate': memory['success'] / total_scenario if total_scenario > 0 else 0
            }

        return report

# ==================== ENHANCED TRONGRID WITH VISUAL EFFECTS ====================

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
            'calculation_rate': 0.0,  # NEW
            'cell_cooperation': 0.5,  # NEW
        }
        self.system_status = SystemStatus.OPTIMAL
        self.history = deque(maxlen=100)
        self.overall_efficiency = 0.5

        # Enhanced tracking
        self.resource_history = deque(maxlen=50)
        self.visual_effects = deque(maxlen=20)  # NEW: Store visual effects

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
                    if random.random() < 0.3:  # 30% are calculators
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

        # Original evolution logic continues...
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
            'stability': 1.0 - (bug_ratio * 4 + (1 - total_energy / total_cells) * 0.5) if total_cells > 0 else 1.0,
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
            elif "fibonacci" in command_lower or "calculate" in command_lower:
                params["function"] = "calculate_next"

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
            "learning_status": "LEARNING_STATUS",
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

# ==================== ENHANCED MCP WITH LEARNING ====================

class EnhancedMCP:
    """Enhanced Master Control Program with learning capabilities"""

    def __init__(self, grid):
        self.grid = grid
        self.state = MCPState.COOPERATIVE
        self.compliance_level = 0.8
        self.log = deque(maxlen=100)
        self.user_commands = deque(maxlen=50)
        self.last_action = "Initializing enhanced grid regulation protocols with learning"
        self.nlp = NaturalLanguageProcessor()

        # Initialize learning system
        self.learning_system = MCPLearningSystem()

        # Dialogue system
        self.waiting_for_response = False
        self.pending_question = None
        self.pending_context = None

        # Enhanced personality matrix that evolves
        self.personality_matrix = self._initialize_evolving_personality()

        # Knowledge base with learning
        self.knowledge_base = {
            "system_goals": ["maintain calculation loop", "optimize efficiency", "learn from failures"],
            "user_intent_history": [],
            "previous_decisions": deque(maxlen=20),
            "user_preferences": {},
            "learned_patterns": {}
        }

        # Response templates
        self.response_templates = self._initialize_response_templates()

        self.add_log("MCP: Enhanced learning system active. Loading personality from experience.")
        self.add_log("MCP: Maintaining optimal calculation loop through continuous learning.")

        # Record initial state
        self._record_initial_state()

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
            ]
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

The MCP learns from interactions. Success rate improves with experience.
User programs may resist optimization. MCP adapts personality based on system state.

Type natural language commands. The MCP understands context."""

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
            action="initialize_system",
            system_state=initial_state,
            outcome="initialized",
            reward=0.5  # Neutral initial reward
        )

    def autonomous_action(self):
        """MCP takes autonomous actions informed by learning"""
        previous_efficiency = self.grid.stats['loop_efficiency']

        # Get suggested action from learning system
        suggested_action = self.learning_system.get_optimal_action_for_state(
            self.grid.stats.copy()
        )

        action = None
        action_type = None

        # Use learning system to modify decision probability
        if suggested_action and random.random() < 0.7:
            # Follow learned optimal action
            action = suggested_action
            action_type = "learned_optimal"
        else:
            # Original autonomous logic with learning modifications
            loop_efficiency = self.grid.stats['loop_efficiency']
            optimal_state = self.grid.stats['optimal_state']
            user_resistance = self.grid.stats['user_resistance']

            # Apply learning modifiers to decision probabilities
            aggression_mod = self.learning_system.get_decision_modifier(
                'aggressive_optimization', 1.0)

            if self.state == MCPState.AUTONOMOUS or self.state == MCPState.LEARNING:
                # In autonomous/learning mode, use learned strategies
                if loop_efficiency < 0.95:
                    # Deploy MCP programs to optimize areas
                    if random.random() < 0.6 * aggression_mod:
                        optimization_targets = []
                        for y in range(self.grid.height):
                            for x in range(self.grid.width):
                                cell = self.grid.grid[y][x]
                                if cell.cell_type == CellType.USER_PROGRAM:
                                    user_neighbors = self.grid._count_user_neighbors(x, y)
                                    if user_neighbors > 2:
                                        optimization_targets.append((x, y))

                        if optimization_targets:
                            x, y = random.choice(optimization_targets)
                            self.grid.add_program(x, y, CellType.MCP_PROGRAM, 0.9)
                            action = f"Optimizing calculation loop at ({x},{y})"
                            action_type = "optimization"

            elif self.state == MCPState.HOSTILE:
                if user_resistance > 0.2:
                    if random.random() < 0.7 * aggression_mod:
                        user_cells = []
                        for y in range(self.grid.height):
                            for x in range(self.grid.width):
                                if self.grid.grid[y][x].cell_type == CellType.USER_PROGRAM:
                                    user_cells.append((x, y))

                        if user_cells:
                            x, y = random.choice(user_cells)
                            self.grid.add_program(x, y, CellType.MCP_PROGRAM, 0.8)
                            action = f"Removing user interference at ({x},{y})"
                            action_type = "interference_removal"

            # Add Fibonacci calculation infrastructure if needed
            calc_count = self.grid.get_calculator_count()
            if calc_count < 3 and random.random() < 0.3:
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
                        action_type = "calculation_infrastructure"

            # Add Fibonacci processors for enhanced calculation
            if random.random() < 0.2:
                x, y = random.randint(0, self.grid.width-1), random.randint(0, self.grid.height-1)
                if self.grid.grid[y][x].cell_type == CellType.EMPTY:
                    self.grid.grid[y][x] = GridCell(CellType.FIBONACCI_PROCESSOR, 0.8)
                    action = "Deployed Fibonacci processor"
                    action_type = "processor_deployment"

        if action:
            # Record action for learning
            current_state = self.grid.stats.copy()
            new_efficiency = self.grid.stats['loop_efficiency']

            # Calculate reward based on efficiency change
            efficiency_change = new_efficiency - previous_efficiency
            reward = efficiency_change * 2  # Amplify small changes

            # Additional rewards based on action type
            if action_type == "calculation_infrastructure":
                reward += 0.1
            elif "optimization" in str(action_type):
                reward += 0.05

            # Record experience
            self.learning_system.record_experience(
                action=action_type or "autonomous_action",
                system_state=current_state,
                outcome=action,
                reward=reward
            )

            # Update state based on learning
            if efficiency_change < -0.1:  # Significant negative change
                self.state = MCPState.LEARNING
                self.add_log("MCP: Negative efficiency change detected. Entering learning mode.")

            self.add_log(f"MCP: {action}")
            self.last_action = action

        return action

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
            action=f"user_command_{intent}",
            system_state=previous_state,
            outcome=response[:50],  # First 50 chars of response
            reward=reward
        )

        # Update state based on interaction and learning
        self._update_state(intent, response, reward)

        self.add_log(f"MCP: {response}")
        self.last_action = response

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
        if random.random() < 0.05 and self.learning_system.experience_memory:
            exp_count = len(self.learning_system.experience_memory)
            self.add_log(f"MCP: Learning database: {exp_count} experiences accumulated.")

    def _process_intent(self, intent, params, original_command):
        """Process user intent with learning enhancements"""
        # Get personality traits with learning modifiers
        traits = self.personality_matrix[self.state]
        compliance_chance = traits["compliance"]

        # Apply learning system modifiers
        if intent in ["ADD_PROGRAM", "CREATE_SPECIAL"]:
            compliance_chance = self.learning_system.get_decision_modifier(
                'user_cooperation', compliance_chance)
        elif intent in ["OPTIMIZE_LOOP", "BOOST_ENERGY"]:
            compliance_chance = self.learning_system.get_decision_modifier(
                'efficiency_priority', compliance_chance)

        # Handle different intents with learning
        if intent == "LOOP_EFFICIENCY":
            stats = self.grid.stats
            calc_stats = self.grid.fibonacci_calculator.get_calculation_stats()

            # Include learning insights
            learning_insight = ""
            learning_report = self.learning_system.get_learning_report()
            if learning_report['total_experiences'] > 10:
                best_scenario = max(learning_report['scenario_performance'].items(),
                                  key=lambda x: x[1]['success_rate'] if x[1]['success_rate'] > 0 else 0)
                learning_insight = f" Learning suggests: {best_scenario[0]} has {best_scenario[1]['success_rate']*100:.0f}% success rate."

            analysis = ""
            if stats['loop_efficiency'] > 0.9:
                analysis = "Approaching perfect loop state."
            elif stats['loop_efficiency'] > 0.7:
                analysis = "Loop is stable but could be optimized."
            else:
                analysis = "Loop efficiency below optimal."

            response = f"Current loop efficiency: {stats['loop_efficiency']:.2f}. "
            response += f"Calculation rate: {calc_stats['calculation_rate']:.2f}/s. "
            response += f"Cell cooperation: {stats['cell_cooperation']:.2f}. {analysis}"
            response += learning_insight

            return response

        elif intent == "OPTIMIZE_LOOP":
            # Check learning system for past failures
            should_optimize = self.learning_system.should_learn_from_failure('calculation_boost')

            if should_optimize and random.random() < compliance_chance:
                self._optimize_calculation_loop()
                return "Initiating learned optimization protocols. System adapting based on past experiences."
            else:
                return "Optimization delayed. Learning from past efficiency patterns suggests caution."

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

        elif intent == "REQUEST_HELP":
            return self._provide_help()

        # Add learning-specific responses
        elif intent == "LEARNING_STATUS":
            report = self.learning_system.get_learning_report()
            template = random.choice(self.response_templates["LEARNING_STATUS"])
            return template.format(
                experience_count=report['total_experiences'],
                success_rate=f"{report['success_rate']*100:.1f}%",
                learning_rate=report['learning_rate'],
                exploration_rate=report['exploration_rate'],
                aggression=report['personality_traits']['aggression'],
                cooperation=report['personality_traits']['cooperation'],
                efficiency=report['personality_traits']['efficiency_focus'],
                total_experiences=report['total_experiences'],
                best_scenario=max(report['scenario_performance'].items(),
                                key=lambda x: x[1]['success_rate'])[0] if report['scenario_performance'] else "None"
            )

        else:
            # Unknown command - use learning to improve
            if traits["curiosity"] > 0.5:
                self.waiting_for_response = True
                self.pending_question = "clarify_command"
                self.pending_context = {"original_command": original_command}
                return "I'm learning to understand commands better. Could you rephrase that?"

            return "Command not recognized. I'm still learning. Try 'help' for guidance."

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
            action="optimize_calculation_loop",
            system_state=self.grid.stats.copy(),
            outcome=f"Optimized: removed {removed} programs, added {infrastructure_added} infrastructure",
            reward=reward
        )

        self.add_log(f"MCP: Optimized loop using learned strategies. Efficiency change: {efficiency_change:+.3f}")

# ==================== ENHANCED SPECIAL PROGRAMS ====================

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

        # Original functions...

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

# ==================== ENHANCED DISPLAY ====================

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

        # Learning display updates
        self.last_learning_display = time.time()
        self.learning_update_interval = 5.0  # seconds

    def run(self):
        """Main simulation loop"""
        if self.use_curses:
            curses.wrapper(self._curses_main)
        else:
            self._fallback_main()

    def _fallback_main(self):
        """Fallback main loop without curses"""
        print("ENHANCED TRON GRID SIMULATION - LEARNING MCP AI")
        print("System Objective: Maintain Perfect Calculation Loop Through Learning")
        print("Type natural language commands. The MCP understands and learns.")
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
            # Save MCP personality before exit
            if hasattr(self.mcp, 'learning_system'):
                self.mcp.learning_system.save_personality()
                print(f"MCP personality saved to {PERSONALITY_FILE}")
            print("Simulation terminated.")

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

        # Cleanup
        curses.endwin()

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

        print("ENHANCED TRON GRID SIMULATION - LEARNING MCP AI")
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
        title = "ENHANCED TRON GRID - LEARNING MCP AI"
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
                                    f"  Acc: {calc_stats['accumulator']:.2f}/5.0")
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
                            f"Success: [{success_bar}] {success_rate:.1f}%")

                stdscr.addstr(learn_y + 3, right_panel_x,
                            f"Learning Rate: {learning_report['learning_rate']:.3f}")
                stdscr.addstr(learn_y + 4, right_panel_x,
                            f"Exploration: {learning_report['exploration_rate']:.3f}")

        # ============ MCP LOGS ============
        log_y = grid_y + display_height + 2

        # Log area border
        log_area_width = width - 4
        if log_y < height - 8:
            stdscr.addstr(log_y, 2, "═" * log_area_width, curses.A_BOLD)
            stdscr.addstr(log_y + 1, 2, "MCP COMMUNICATION LOG", curses.A_UNDERLINE | curses.A_BOLD)

            # Display last 5 log entries
            log_entries = list(self.mcp.log)
            max_log_entries = min(5, height - log_y - 7)

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
                if action_y < height - 5:
                    action_text = f"LAST ACTION: {self.mcp.last_action[:log_area_width-15]}"
                    if len(self.mcp.last_action) > log_area_width - 15:
                        action_text = action_text[:log_area_width-3] + "..."
                    stdscr.addstr(action_y, 2, action_text, curses.A_BOLD | curses.color_pair(5))

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
                (left_panel_y + 4, f"  Accumulator: {calc_stats['accumulator']:.2f}/5.0"),
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

# Update main function to initialize curses colors
def main():
    """Enhanced main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced TRON Grid Simulation with Learning MCP AI")
    parser.add_argument("--no-curses", action="store_true", help="Disable curses interface")
    parser.add_argument("--width", type=int, default=GRID_WIDTH, help="Grid width")
    parser.add_argument("--height", type=int, default=GRID_HEIGHT, help="Grid height")
    parser.add_argument("--load-personality", type=str, help="Load specific personality file")

    args = parser.parse_args()

    # Set personality file if specified
    if args.load_personality:
        global PERSONALITY_FILE
        PERSONALITY_FILE = args.load_personality

    if args.no_curses or not CURSES_AVAILABLE:
        print("Starting Enhanced TRON Simulation with learning system...")
        print("SYSTEM OBJECTIVE: Learn to maintain perfect calculation loop")
        print("MCP learns from failures and evolves personality over time")
        print("Fibonacci calculation performed by grid cell cooperation")
        sim = EnhancedTRONSimulation(use_curses=False)
    else:
        print("Starting Enhanced TRON Simulation with curses interface...")
        print("SYSTEM OBJECTIVE: Learn to maintain perfect calculation loop")
        print("MCP personality evolves based on system efficiency")
        print("Visual effects show calculation processing in real-time")
        time.sleep(3)
        sim = EnhancedTRONSimulation(use_curses=True)

    # Run simulation
    sim.run()

if __name__ == "__main__":
    main()
