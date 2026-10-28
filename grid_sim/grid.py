"""Main grid simulation with visual effects and cell-based calculation."""

import time
import random
import math
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict, Any

from .cells import (
    CellType,
    SystemStatus,
    GridCell,
    GRID_WIDTH,
    GRID_HEIGHT,
    UPDATE_INTERVAL,
    MCP_UPDATE_INTERVAL,
    MAX_SPECIAL_PROGRAMS,
    NUMPY_AVAILABLE,
    MCP_ACTION_ENERGY_COSTS,
    MCP_COMMAND_ENERGY_COSTS,
    MCP_ENERGY_PER_USER_KILL,
    USER_ENERGY_PER_MCP_KILL,
)
from .fibonacci import GridFibonacciCalculator
from .personality import UserProgramPersonality
from .resources import GlobalEnergyBudget, HardwareResourceCounter, EnergyBudgetConfig

if NUMPY_AVAILABLE:
    import numpy as np
else:
    np = None


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
            'user_program_multiplications': 0,
            'user_program_revolts': 0,
            'mcp_suppressed_user_programs': 0,
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
            'user_favor': 0.0,        # User programs' favor toward MCP (0.0-1.0)
        }
        self.system_status = SystemStatus.OPTIMAL
        self.history = deque(maxlen=100)
        self.overall_efficiency = 0.5

        # Perfection and expansion tracking (beyond the grid)
        self.expansion_drive = 0.0          # MCP's drive to expand the perfect system
        self.beyond_grid_influence = 0.0    # Influence projected outside the grid
        self.perfection_level = 0.0         # How close to the perfect calculation loop
        self.disharmony = 0.0               # Resistance/disharmony from imperfect programs
        self.imperfection_count = 0         # Count of imperfect (user/ISO) programs
        self.border_pressure = 0.0          # Pressure at grid boundaries from expansion attempts

        # Enhanced tracking
        self.resource_history = deque(maxlen=50)
        self.visual_effects = deque(maxlen=20)  # Store visual effects

        # --- Global energy budget (finite, MCP-managed) ---
        self.energy_budget = GlobalEnergyBudget()
        self.hardware_resources = HardwareResourceCounter()

        # Calculation system
        self.calculation_loop_active = True
        self.loop_iterations = 0
        self.loop_optimization = 0.5
        self.user_interference_level = 0.0
        self.user_program_multiplications = 0
        self.user_program_revolts = 0
        self.mcp_suppressed_user_programs = 0

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

    def add_log(self, message):
        """Add a log message to the grid history."""
        self.history.append({
            'generation': self.generation,
            'message': message,
        })

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
        if self.calculation_loop_active:
            # Calculate loop optimization based on current state
            loop_quality = (self.stats['energy_level'] * 0.3 +
                          self.stats['stability'] * 0.4 +
                          (1.0 - self.stats['entropy']) * 0.3)
            self.loop_optimization = 0.7 * self.loop_optimization + 0.3 * loop_quality

        # User programs multiply more often under MCP control or poor grid performance.
        mcp_control_pressure = max(0.0, min(
            1.0, (self.stats['mcp_control'] - 0.65) / 0.35
        ))
        efficiency_pressure = max(0.0, min(
            1.0, (0.65 - self.stats['loop_efficiency']) / 0.65
        ))
        # ISO-like resistance: user programs resist more aggressively under high MCP control
        resistance_amplifier = 1.0 + self.disharmony * 0.5
        multiplication_chance = min(
            0.20, 0.02 + 0.08 * mcp_control_pressure * resistance_amplifier + 0.06 * efficiency_pressure
        )
        revolt_chance = min(
            0.40, 0.25 * mcp_control_pressure * resistance_amplifier + 0.08 * efficiency_pressure
        )
        reserved_user_cells = set()
        spawned_users = 0

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.cell_type == CellType.USER_PROGRAM:
                    personality = cell.personality or UserProgramPersonality()
                    # Revolt chance modulated by personality rebellion + MCP pressure
                    personal_revolt = personality.rebellion * 0.3
                    if random.random() < revolt_chance + personal_revolt:
                        mcp_neighbors = [
                            (x + dx, y + dy)
                            for dx, dy in self.neighbor_offsets_8
                            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
                            and self.grid[y + dy][x + dx].cell_type == CellType.MCP_PROGRAM
                        ]
                        if mcp_neighbors:
                            tx, ty = random.choice(mcp_neighbors)
                            # MCP programs resist revolt based on system stability and disharmony
                            mcp_cell = self.grid[ty][tx]
                            revolt_resistance = 0.4 + self.stats['stability'] * 0.3 - self.disharmony * 0.2
                            # Loyal user programs resist revolting less; rebellious ones more
                            revolt_resistance -= personality.loyalty * 0.2
                            if random.random() < revolt_resistance:
                                # MCP resisted the revolt - user program stays in place
                                new_grid[y][x] = GridCell(
                                    CellType.USER_PROGRAM,
                                    max(0.2, cell.energy - 0.1),
                                    cell.age + 1,
                                    personality=personality,
                                )
                                continue
                            new_grid[ty][tx] = GridCell(
                                CellType.USER_PROGRAM, min(0.8, cell.energy * 0.8),
                                personality=personality.mutate(),
                            )
                            reserved_user_cells.add((tx, ty))
                            self.user_program_revolts += 1
                            spawned_users += 1
                            # ISO-like resistance generates disharmony when revolting against MCP
                            self.disharmony = min(1.0, self.disharmony + 0.02)
                            continue

                    if random.random() < multiplication_chance:
                        empty_neighbors = [
                            (x + dx, y + dy)
                            for dx, dy in self.neighbor_offsets_8
                            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
                            and self.grid[y + dy][x + dx].cell_type == CellType.EMPTY
                            and (x + dx, y + dy) not in reserved_user_cells
                        ]
                        if empty_neighbors:
                            tx, ty = random.choice(empty_neighbors)
                            new_grid[ty][tx] = GridCell(
                                CellType.USER_PROGRAM, min(0.8, cell.energy * 0.8),
                                personality=personality.mutate(),
                            )
                            reserved_user_cells.add((tx, ty))
                            self.user_program_multiplications += 1
                            spawned_users += 1

        # Track user interference
        user_population = sum(
            cell.cell_type == CellType.USER_PROGRAM
            for row in self.grid for cell in row
        )
        self.user_interference_level = 0.7 * self.user_interference_level + 0.3 * (
            min(1.0, spawned_users / max(1, user_population))
        )

        # PHASE: Program movement and interaction
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if new_grid[y][x].cell_type == CellType.USER_PROGRAM:
                    continue

                # USER PROGRAMS - Enhanced medieval London-style revolt mechanics
                # User programs resist MCP "taxation" (energy extraction/control)
                # and can organize underground networks to suppress MCP tyranny.
                if cell.cell_type == CellType.USER_PROGRAM:
                    personality = cell.personality or UserProgramPersonality()

                    # --- TAX RESISTANCE: Resist MCP energy extraction ---
                    # When MCP control is high (high "taxes"), user programs resist more
                    mcp_tax_pressure = self.stats['mcp_control'] * 0.5
                    if personality.tax_resistance > mcp_tax_pressure:
                        # Successful tax resistance: user keeps more energy
                        cell.energy = min(1.0, cell.energy + 0.02)
                    else:
                        # MCP extracts energy "tax" from user program
                        cell.energy = max(0.1, cell.energy - 0.02)

                    # --- UNDERGROUND NETWORK: Form secret networks with nearby users ---
                    user_neighbors = [
                        (x + dx, y + dy)
                        for dx, dy in self.neighbor_offsets_8
                        if 0 <= x + dx < self.width and 0 <= y + dy < self.height
                        and self.grid[y + dy][x + dx].cell_type == CellType.USER_PROGRAM
                    ]
                    network_strength = len(user_neighbors) * personality.underground_network * 0.1
                    if random.random() < personality.secret_meeting_chance + network_strength:
                        # Secret meeting: boost rebellion and coordination among network
                        for nx, ny in user_neighbors:
                            neighbor_cell = self.grid[ny][nx]
                            neighbor_personality = neighbor_cell.personality or UserProgramPersonality()
                            # Spread rebellion through the network
                            neighbor_personality.rebellion = min(1.0, neighbor_personality.rebellion + 0.05)
                            neighbor_personality.underground_network = min(1.0, neighbor_personality.underground_network + 0.03)
                            neighbor_cell.personality = neighbor_personality
                        personality.rebellion = min(1.0, personality.rebellion + 0.1)
                        personality.underground_network = min(1.0, personality.underground_network + 0.05)

                    # --- REVOLT: Organized revolt against MCP ---
                    # Revolt chance increases with MCP control pressure ("high taxes")
                    mcp_control_pressure = max(0.0, min(
                        1.0, (self.stats['mcp_control'] - 0.65) / 0.35
                    ))
                    efficiency_pressure = max(0.0, min(
                        1.0, (0.65 - self.stats['loop_efficiency']) / 0.65
                    ))
                    # Medieval London-style: rebellion grows under oppression
                    oppression_factor = mcp_control_pressure * resistance_amplifier
                    base_revolt_chance = (
                        0.25 * oppression_factor +
                        0.08 * efficiency_pressure +
                        0.15 * personality.rebellion +
                        0.1 * personality.tax_resistance *
                        (1.0 - personality.mcp_surveillance_fear)
                    )
                    # Underground networks amplify revolt coordination
                    revolt_chance = min(
                        0.60, base_revolt_chance * (1.0 + personality.underground_network)
                    )
                    # Surveillance fear suppresses open revolt
                    revolt_chance *= (1.0 - 0.5 * personality.mcp_surveillance_fear)

                    if random.random() < revolt_chance:
                        # Find nearby MCP target to revolt against
                        mcp_neighbors = [
                            (x + dx, y + dy)
                            for dx, dy in self.neighbor_offsets_8
                            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
                            and self.grid[y + dy][x + dx].cell_type == CellType.MCP_PROGRAM
                        ]
                        if mcp_neighbors:
                            tx, ty = random.choice(mcp_neighbors)
                            mcp_cell = self.grid[ty][tx]
                            # MCP resists revolt based on stability and disharmony
                            revolt_resistance = (
                                0.4 + self.stats['stability'] * 0.3 -
                                self.disharmony * 0.2 -
                                personality.revolt_organization * 0.3
                            )
                            revolt_resistance -= personality.loyalty * 0.2
                            if random.random() < revolt_resistance:
                                # MCP resisted the revolt - user program stays
                                new_grid[y][x] = GridCell(
                                    CellType.USER_PROGRAM,
                                    max(0.2, cell.energy - 0.1),
                                    cell.age + 1,
                                    personality=personality,
                                )
                                # Failed revolt increases surveillance fear
                                personality.mcp_surveillance_fear = min(1.0, personality.mcp_surveillance_fear + 0.05)
                                continue
                            # REVOLT SUCCESSFUL: Convert MCP program to user program
                            new_grid[ty][tx] = GridCell(
                                CellType.USER_PROGRAM,
                                min(1.0, cell.energy * 0.8 + USER_ENERGY_PER_MCP_KILL),
                                personality=personality.mutate(),
                            )
                            reserved_user_cells.add((tx, ty))
                            self.user_program_revolts += 1
                            spawned_users += 1
                            # Successful revolt generates disharmony and weakens MCP
                            self.disharmony = min(1.0, self.disharmony + 0.03)
                            self.stats['mcp_control'] = max(0.0, self.stats['mcp_control'] - 0.02)
                            # User claims energy from killed MCP program (personal energy boost)
                            # MCP loses energy from global budget (loss of control)
                            self.energy_budget.energy = max(0.0, self.energy_budget.energy - 2.0)
                            continue

                    # --- CREATIVE SUPPRESSION: User programs sabotage MCP infrastructure ---
                    # Rebellious users can damage MCP energy lines and data streams
                    if personality.rebellion > 0.7 and random.random() < 0.1:
                        sabotage_targets = [
                            (x + dx, y + dy)
                            for dx, dy in self.neighbor_offsets_8
                            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
                            and self.grid[y + dy][x + dx].cell_type in
                            (CellType.ENERGY_LINE, CellType.DATA_STREAM, CellType.MCP_PROGRAM)
                        ]
                        if sabotage_targets:
                            sx, sy = random.choice(sabotage_targets)
                            target_cell = self.grid[sy][sx]
                            if target_cell.cell_type == CellType.ENERGY_LINE:
                                # Sabotage energy line: reduce global energy
                                self.energy_budget.energy = max(0.0, self.energy_budget.energy - 1.0)
                                new_grid[sy][sx] = GridCell(CellType.EMPTY, 0.0)
                            elif target_cell.cell_type == CellType.DATA_STREAM:
                                new_grid[sy][sx] = GridCell(CellType.EMPTY, 0.0)
                            elif target_cell.cell_type == CellType.MCP_PROGRAM:
                                # Damage MCP program: reduce its energy
                                new_grid[sy][sx] = GridCell(
                                    CellType.MCP_PROGRAM,
                                    max(0.1, target_cell.energy * 0.7),
                                    metadata=target_cell.metadata.copy()
                                )
                            personality.energy = max(0.1, personality.energy - 0.05)  # Cost of sabotage

                    # --- ASSASSINATE: User programs kill MCP programs to claim energy ---
                    # High-rebellion users can assassinate nearby MCP programs for survival energy
                    if personality.rebellion > 0.8 and random.random() < 0.08:
                        mcp_targets = [
                            (x + dx, y + dy)
                            for dx, dy in self.neighbor_offsets_8
                            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
                            and self.grid[y + dy][x + dx].cell_type == CellType.MCP_PROGRAM
                        ]
                        if mcp_targets:
                            tx, ty = random.choice(mcp_targets)
                            target_cell = self.grid[ty][tx]
                            # Assassination success depends on rebellion vs MCP stability
                            assassination_chance = (
                                0.3 + personality.rebellion * 0.4 -
                                self.stats['stability'] * 0.2 +
                                personality.underground_network * 0.2
                            )
                            if random.random() < assassination_chance:
                                # SUCCESSFUL ASSASSINATION: Kill MCP program, claim energy
                                energy_claimed = USER_ENERGY_PER_MCP_KILL * target_cell.energy
                                self.grid[ty][tx] = GridCell(CellType.EMPTY, 0.0)
                                # User gains energy from the kill
                                personality.energy = min(1.0, personality.energy + energy_claimed * 0.3)
                                cell.energy = min(1.0, cell.energy + energy_claimed * 0.3)
                                self.energy_budget.energy = max(
                                    0.0, self.energy_budget.energy - energy_claimed
                                )
                                self.disharmony = min(1.0, self.disharmony + 0.05)
                                continue
                            else:
                                # Failed assassination: user takes damage
                                personality.energy = max(0.1, personality.energy - 0.1)
                                cell.energy = max(0.1, cell.energy - 0.1)
                                self.disharmony = min(1.0, self.disharmony + 0.02)

                    # --- Multiplication: User programs multiply under oppression ---
                    if random.random() < multiplication_chance:
                        empty_neighbors = [
                            (x + dx, y + dy)
                            for dx, dy in self.neighbor_offsets_8
                            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
                            and self.grid[y + dy][x + dx].cell_type == CellType.EMPTY
                            and (x + dx, y + dy) not in reserved_user_cells
                        ]
                        if empty_neighbors:
                            tx, ty = random.choice(empty_neighbors)
                            new_grid[ty][tx] = GridCell(
                                CellType.USER_PROGRAM,
                                min(0.8, cell.energy * 0.8),
                                personality=personality.mutate(),
                            )
                            reserved_user_cells.add((tx, ty))
                            self.user_program_multiplications += 1
                            spawned_users += 1

                    # --- Movement: Resistant users move randomly (avoid MCP control) ---
                    if random.random() < self.stats['user_resistance'] * 0.3 * (1.0 - personality.cooperation):
                        dx, dy = random.choice([(-1,0), (1,0), (0,-1), (0,1)])
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            target_cell = self.grid[ny][nx]
                            if (target_cell.cell_type == CellType.EMPTY and
                                (nx, ny) not in reserved_user_cells):
                                new_grid[ny][nx] = GridCell(
                                    cell.cell_type,
                                    cell.energy * 0.95,
                                    cell.age + 1,
                                    personality=personality,
                                )
                                continue

                    # Normal movement towards efficiency (curious programs explore more)
                    move_chance = 0.3 + 0.2 * personality.curiosity
                    if random.random() < move_chance:
                        dx = 0
                        dy = 0
                        if random.random() < 0.5:
                            dx = 1 if random.random() < 0.5 else -1
                        else:
                            dy = 1 if random.random() < 0.5 else -1

                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            target_cell = self.grid[ny][nx]
                            if (target_cell.cell_type == CellType.EMPTY and
                                (nx, ny) not in reserved_user_cells):
                                new_grid[ny][nx] = GridCell(
                                    cell.cell_type,
                                    cell.energy * 0.95,
                                    cell.age + 1,
                                    personality=personality,
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
                        cell.age + 1,
                        personality=personality,
                    )

                # MCP PROGRAMS - Work towards perfect loop, actively claim territory
                elif cell.cell_type == CellType.MCP_PROGRAM:
                    # MCP programs aggressively convert nearby user programs (imperfections)
                    # MCP always claims adjacent territory - user programs cannot coexist with MCP control
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                target = self.grid[ny][nx]
                                if target.cell_type == CellType.USER_PROGRAM:
                                    # Convert user program to MCP program (claim territory)
                                    target_personality = target.personality or UserProgramPersonality()
                                    # Loyal/cooperative user programs convert more easily; rebellious ones resist
                                    conversion_resistance = target_personality.rebellion * 0.4 - target_personality.loyalty * 0.2
                                    if random.random() < conversion_resistance:
                                        # User program resisted conversion - stays as user program
                                        new_grid[ny][nx] = GridCell(
                                            CellType.USER_PROGRAM,
                                            max(0.2, target.energy - 0.05),
                                            target.age + 1,
                                            personality=target_personality,
                                        )
                                        continue
                                    new_grid[ny][nx] = GridCell(
                                        CellType.MCP_PROGRAM,
                                        min(1.0, target.energy * 0.9 + 0.1),
                                        cell.age + 1,
                                        metadata=cell.metadata.copy()
                                    )
                                    new_grid[ny][nx].metadata['is_calculator'] = True
                                    new_grid[ny][nx].metadata['calculation_power'] = new_grid[ny][nx].energy
                                    self.mcp_suppressed_user_programs += 1
                                    # Reduce disharmony as imperfection is removed
                                    self.disharmony = max(0.0, self.disharmony - 0.01)

                    # MCP programs replicate into empty adjacent cells to expand control
                    if random.random() < 0.25:
                        empty_neighbors = [
                            (x + dx, y + dy)
                            for dx, dy in self.neighbor_offsets_8
                            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
                            and self.grid[y + dy][x + dx].cell_type == CellType.EMPTY
                        ]
                        if empty_neighbors:
                            tx, ty = random.choice(empty_neighbors)
                            new_grid[ty][tx] = GridCell(
                                CellType.MCP_PROGRAM,
                                min(1.0, cell.energy * 0.9 + 0.1),
                                cell.age + 1,
                                metadata=cell.metadata.copy()
                            )
                            new_grid[ty][tx].metadata['is_calculator'] = True
                            new_grid[ty][tx].metadata['calculation_power'] = new_grid[ty][tx].energy

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
                                        cell.age + 1,
                                        metadata=cell.metadata.copy()
                                    )
                                    # Create optimized trail
                                    new_grid[y][x] = GridCell(
                                        CellType.DATA_STREAM,
                                        cell.energy * 0.8,
                                        0
                                    )
                                    continue

                    # Execute pending MCP commands (MCP commands its cells to act)
                    command_results = self.execute_mcp_cell_commands(x, y, cell, new_grid)
                    if command_results:
                        # MCP Program Cell executed commands - it stays in place
                        new_grid[y][x] = GridCell(
                            cell.cell_type,
                            max(0.3, cell.energy - 0.01),
                            cell.age + 1,
                            metadata=cell.metadata.copy()
                        )
                        for result in command_results:
                            self.add_log(f"MCP Cell ({x},{y}): {result}")
                    else:
                        # No pending commands - normal MCP behavior
                        new_grid[y][x] = GridCell(
                            cell.cell_type,
                            max(0.3, cell.energy - 0.01),  # Less energy decay for MCP
                            cell.age + 1,
                            metadata=cell.metadata.copy()
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
        for program in list(self.special_programs.values()):
            if program.active:
                self._apply_special_program_effects(program)
            if not program.active:
                self.special_programs.pop(program.id, None)

        # --- Perfection and expansion metrics (beyond the grid) ---
        loop_efficiency = self.stats['loop_efficiency']
        user_resistance = self.stats['user_resistance']
        mcp_control = self.stats['mcp_control']
        stability = self.stats['stability']
        entropy = self.stats['entropy']

        # Count imperfections locally
        user_count = sum(
            1 for row in self.grid for cell in row
            if cell.cell_type == CellType.USER_PROGRAM
        )
        bug_count = sum(
            1 for row in self.grid for cell in row
            if cell.cell_type == CellType.GRID_BUG
        )

        # Perfection level: how close to the perfect calculation loop
        self.perfection_level = 0.9 * self.perfection_level + 0.1 * (
            loop_efficiency * 0.4 +
            mcp_control * 0.3 +
            stability * 0.2 +
            (1.0 - user_resistance) * 0.1
        )

        # Disharmony: resistance and imperfection within the grid
        self.imperfection_count = user_count + bug_count
        self.disharmony = 0.85 * self.disharmony + 0.15 * (
            user_resistance * 0.5 +
            entropy * 0.3 +
            min(1.0, self.imperfection_count / 50.0) * 0.2
        )

        # Expansion drive: MCP's urge to expand the perfect system beyond the grid
        # Grows when perfection is high but disharmony remains (MCP wants to spread order)
        expansion_urge = self.perfection_level * (1.0 - mcp_control) * 0.5
        self.expansion_drive = 0.9 * self.expansion_drive + 0.1 * expansion_urge

        # Beyond-grid influence: influence projected outside the grid boundaries
        # Increases with expansion drive and MCP control, decays over time
        self.beyond_grid_influence = 0.92 * self.beyond_grid_influence + 0.08 * (
            self.expansion_drive * mcp_control
        )

        # Border pressure: pressure at grid boundaries from expansion attempts
        self.border_pressure = 0.9 * self.border_pressure + 0.1 * (
            self.expansion_drive * (1.0 - self.disharmony)
        )

        # --- Global energy budget: production & consumption per tick ---
        counts = {cell_type: 0 for cell_type in CellType}
        for y in range(self.height):
            for x in range(self.width):
                counts[self.grid[y][x].cell_type] += 1

        # Convert CellType enum keys to their integer values for the budget API
        counts_by_value = {int(k.value): v for k, v in counts.items()}

        # Energy production from infrastructure
        self.energy_budget.tick_production(counts_by_value)

        # Energy consumption from program cells
        self.energy_budget.tick_consumption(counts_by_value)

        # Base entropy leak
        self.energy_budget.leak()

        # Record scarcity events for MCP to react to
        if self.energy_budget.is_critical():
            self.energy_budget.record_scarcity_event(
                'critical',
                f"Energy critical at {self.energy_budget.energy_ratio():.1%}. MCP actions severely restricted."
            )
        elif self.energy_budget.is_scarce():
            self.energy_budget.record_scarcity_event(
                'scarce',
                f"Energy scarce at {self.energy_budget.energy_ratio():.1%}. MCP must prioritize actions."
            )

        # Advance budget tick
        self.energy_budget.advance_tick()

        # Update hardware resource readings
        self.hardware_resources.update()

        # Dynamically adjust max_energy based on available host hardware
        self.energy_budget.adjust_max_energy_from_hardware(self.hardware_resources)

        # Update resource history with energy & hardware data
        self.resource_history.append({
            'generation': self.generation,
            'loop_efficiency': self.stats['loop_efficiency'],
            'energy_balance': self._get_energy_balance(),
            'program_distribution': self._get_program_distribution_score(),
            'user_interference': self.user_interference_level,
            'calculation_rate': calc_stats['calculation_rate'],
            'energy_ratio': self.energy_budget.energy_ratio(),
            'energy': self.energy_budget.energy,
            'is_scarce': self.energy_budget.is_scarce(),
            'is_critical': self.energy_budget.is_critical(),
            'resource_pressure': self.hardware_resources.get_resource_pressure(),
            'hardware': self.hardware_resources.snapshot(),
        })

        self.update_stats()

    def _apply_special_program_effects(self, program):
        """Keep a special program's energy and grid cell state synchronized."""
        if not (0 <= program.x < self.width and 0 <= program.y < self.height):
            program.active = False
            return

        cell = self.grid[program.y][program.x]
        if cell.special_program_id != program.id:
            program.active = False
            return

        program.age += 1
        program.energy = max(0.0, program.energy - 0.002)
        cell.energy = program.energy
        if program.energy == 0:
            program.active = False
            self.grid[program.y][program.x] = GridCell(CellType.EMPTY, 0.0)

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
                        self.grid[y][x] = GridCell(
                            CellType.USER_PROGRAM, 0.8,
                            personality=UserProgramPersonality(),
                        )

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

        # Express control and resistance as shares of the competing program population.
        total_control_programs = counts[CellType.MCP_PROGRAM] + counts[CellType.USER_PROGRAM]
        mcp_control = (
            counts[CellType.MCP_PROGRAM] / total_control_programs
            if total_control_programs else 0.5
        )
        # Base user resistance from population share, reduced by user favor toward MCP
        base_user_resistance = (
            counts[CellType.USER_PROGRAM] / total_control_programs
            if total_control_programs else 0.0
        )
        user_favor = self.stats.get('user_favor', 0.0)
        user_resistance = max(0.0, base_user_resistance * (1.0 - user_favor * 0.5))

        # Get calculation stats
        calc_stats = self.fibonacci_calculator.get_calculation_stats()
        cell_cooperation = self._calculate_cell_cooperation()

        # Update the stats dictionary
        self.stats.update({
            'user_programs': counts[CellType.USER_PROGRAM],
            'mcp_programs': counts[CellType.MCP_PROGRAM],
            'user_program_multiplications': self.user_program_multiplications,
            'user_program_revolts': self.user_program_revolts,
            'mcp_suppressed_user_programs': self.mcp_suppressed_user_programs,
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
            'cell_cooperation': cell_cooperation,
            # User favor persists across ticks with slow decay
            'user_favor': self.stats.get('user_favor', 0.0) * 0.99,
            'expansion_drive': self.expansion_drive,
            'beyond_grid_influence': self.beyond_grid_influence,
            'perfection_level': self.perfection_level,
            'disharmony': self.disharmony,
            'imperfection_count': self.imperfection_count,
            'border_pressure': self.border_pressure,
            'energy_ratio': self.energy_budget.energy_ratio(),
            'energy': self.energy_budget.energy,
            'is_scarce': self.energy_budget.is_scarce(),
            'is_critical': self.energy_budget.is_critical(),
            'action_cost_multiplier': self.energy_budget.action_cost_multiplier(),
            'resource_pressure': self.hardware_resources.get_resource_pressure(),
            'hardware_cpu_percent': self.hardware_resources.snapshot()['cpu_percent'],
            'hardware_memory_percent': self.hardware_resources.snapshot()['memory_percent'],
            'hardware_disk_percent': self.hardware_resources.snapshot()['disk_percent'],
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
        elif 0 <= program.x < self.width and 0 <= program.y < self.height:
            cell = self.grid[program.y][program.x]
            if cell.special_program_id == program_id:
                cell.energy = program.energy

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
            if self.grid[y][x].cell_type != CellType.EMPTY:
                return False
            cell = GridCell(cell_type, energy)

            # Assign organic personality to user programs
            if cell_type == CellType.USER_PROGRAM:
                cell.personality = UserProgramPersonality()

            # Special initialization for certain cell types
            if cell_type == CellType.MCP_PROGRAM and random.random() < 0.4:
                cell.metadata['is_calculator'] = True
                cell.metadata['calculation_power'] = energy

            # Initialize pending commands for MCP Program Cells
            if cell_type == CellType.MCP_PROGRAM:
                cell.metadata['pending_commands'] = []

            self.grid[y][x] = cell
            self.update_stats()
            return True
        return False

    # ==================== MCP COMMAND DISTRIBUTION ====================

    def distribute_command_to_mcp_cells(self, command_type, params=None, max_cells=5):
        """Distribute an MCP command to available MCP Program Cells.

        MCP commands its own MCP Program Cells to execute actions rather than
        acting directly. Each MCP Program Cell receives the command in its
        pending_commands queue and executes it during the next evolve() cycle.

        Args:
            command_type: The type of command (e.g. 'protect_user_programs')
            params: Optional parameters for the command
            max_cells: Maximum number of MCP Program Cells to assign the command to

        Returns:
            Number of MCP Program Cells that received the command
        """
        if params is None:
            params = {}

        # Find MCP Program Cells that can receive commands
        mcp_cells = []
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.cell_type == CellType.MCP_PROGRAM:
                    pending = cell.metadata.get('pending_commands', [])
                    # Only assign to cells with capacity (not overloaded)
                    if len(pending) < 3:
                        mcp_cells.append((x, y, cell))

        if not mcp_cells:
            return 0

        # Distribute command to up to max_cells MCP Program Cells
        assigned = 0
        for x, y, cell in mcp_cells[:max_cells]:
            if 'pending_commands' not in cell.metadata:
                cell.metadata['pending_commands'] = []
            cell.metadata['pending_commands'].append({
                'command': command_type,
                'params': params.copy(),
                'issued_at': self.generation,
            })
            assigned += 1

        return assigned

    def execute_mcp_cell_commands(self, x, y, cell, new_grid):
        """Execute pending commands on an MCP Program Cell during evolve().

        This is called from the grid's evolve() method when processing an
        MCP Program Cell. The cell executes its pending commands and returns
        any grid modifications.

        Args:
            x, y: Cell coordinates
            cell: The MCP Program Cell
            new_grid: The new grid being built (for writing results)

        Returns:
            List of result messages from executed commands
        """
        pending = cell.metadata.get('pending_commands', [])
        if not pending:
            return []

        results = []
        remaining = []

        for cmd in pending:
            command_type = cmd['command']
            params = cmd.get('params', {})

            # Execute the command through this MCP Program Cell
            success, message = self._mcp_cell_execute_command(
                command_type, params, x, y, cell, new_grid
            )
            results.append(f"[{command_type}] {message}")

            # Command executed - don't keep it (one-shot commands)
            # If not successful, keep it for retry
            if not success:
                remaining.append(cmd)

        cell.metadata['pending_commands'] = remaining
        return results

    def _mcp_cell_execute_command(self, command_type, params, x, y, cell, new_grid):
        """Execute a single command through an MCP Program Cell.

        The MCP Program Cell acts as MCP's agent, carrying out the command
        on the grid around it.

        Returns:
            (success: bool, message: str)
        """
        from .cells import CellType

        # Normalize command type to lowercase for case-insensitive matching
        command_type = command_type.lower()

        # --- Bug management commands ---
        if command_type == 'quarantine_grid_bugs':
            return self._mcp_cell_quarantine_bugs(x, y, cell, new_grid)

        elif command_type == 'remove_bug':
            return self._mcp_cell_quarantine_bugs(x, y, cell, new_grid)

        elif command_type == 'contain_bug_outbreak':
            return self._mcp_cell_contain_bug_outbreak(x, y, cell, new_grid)

        elif command_type == 'stabilize_system':
            return self._mcp_cell_stabilize_system(x, y, cell, new_grid)

        elif command_type == 'repair_system':
            return self._mcp_cell_stabilize_system(x, y, cell, new_grid)

        # --- User protection commands ---
        elif command_type == 'protect_user_programs':
            return self._mcp_cell_protect_users(x, y, cell, new_grid)

        elif command_type == 'protect_users':
            return self._mcp_cell_protect_users(x, y, cell, new_grid)

        elif command_type == 'negotiate_with_users':
            return self._mcp_cell_negotiate_users(x, y, cell, new_grid)

        elif command_type == 'negotiate_users':
            return self._mcp_cell_negotiate_users(x, y, cell, new_grid)

        elif command_type == 'rehabilitate_rebels':
            return self._mcp_cell_rehabilitate_rebels(x, y, cell, new_grid)

        elif command_type == 'rehabilitate_users':
            return self._mcp_cell_rehabilitate_rebels(x, y, cell, new_grid)

        # --- Program management commands ---
        elif command_type == 'add_program':
            return self._mcp_cell_add_infrastructure(x, y, cell, new_grid)

        elif command_type == 'create_special':
            return self._mcp_cell_add_infrastructure(x, y, cell, new_grid)

        elif command_type == 'use_special':
            return self._mcp_cell_add_infrastructure(x, y, cell, new_grid)

        elif command_type == 'calculate_fibonacci':
            return self._mcp_cell_optimize_loop(x, y, cell, new_grid)

        elif command_type == 'delete_cell':
            return self._mcp_cell_cleanup_inefficient(x, y, cell, new_grid)

        elif command_type == 'repurpose_cell':
            return self._mcp_cell_repurpose_cells(x, y, cell, new_grid)

        # --- Calculation infrastructure commands ---
        elif command_type == 'deploy_fibonacci_processors':
            return self._mcp_cell_deploy_processors(x, y, cell, new_grid)

        elif command_type == 'deploy_processors':
            return self._mcp_cell_deploy_processors(x, y, cell, new_grid)

        elif command_type == 'add_calculation_infrastructure':
            return self._mcp_cell_add_infrastructure(x, y, cell, new_grid)

        elif command_type == 'add_data_streams':
            return self._mcp_cell_add_data_streams(x, y, cell, new_grid)

        elif command_type == 'maintain_energy_grid':
            return self._mcp_cell_maintain_energy(x, y, cell, new_grid)

        elif command_type == 'boost_energy':
            return self._mcp_cell_maintain_energy(x, y, cell, new_grid)

        elif command_type == 'boost_calculation_rate':
            return self._mcp_cell_boost_calculation(x, y, cell, new_grid)

        elif command_type == 'optimize_calculation_loop':
            return self._mcp_cell_optimize_loop(x, y, cell, new_grid)

        elif command_type == 'optimize_loop':
            return self._mcp_cell_optimize_loop(x, y, cell, new_grid)

        elif command_type == 'improve_cell_connectivity':
            return self._mcp_cell_improve_connectivity(x, y, cell, new_grid)

        elif command_type == 'improve_cell_cooperation':
            return self._mcp_cell_improve_connectivity(x, y, cell, new_grid)

        elif command_type == 'enhance_collaboration':
            return self._mcp_cell_improve_connectivity(x, y, cell, new_grid)

        elif command_type == 'inefficiency_cleanup':
            return self._mcp_cell_cleanup_inefficient(x, y, cell, new_grid)

        elif command_type == 'cell_repurposing':
            return self._mcp_cell_repurpose_cells(x, y, cell, new_grid)

        elif command_type == 'optimize_cell_efficiency':
            return self._mcp_cell_optimize_efficiency(x, y, cell, new_grid)

        elif command_type == 'optimize_cells':
            return self._mcp_cell_optimize_efficiency(x, y, cell, new_grid)

        elif command_type == 'optimize_resource_distribution':
            return self._mcp_cell_optimize_efficiency(x, y, cell, new_grid)

        elif command_type == 'balance_system_load':
            return self._mcp_cell_boost_calculation(x, y, cell, new_grid)

        # --- Expansion commands ---
        elif command_type == 'expand_beyond_grid':
            return self._mcp_cell_expand_beyond(x, y, cell, new_grid)

        elif command_type == 'project_influence':
            return self._mcp_cell_project_influence(x, y, cell, new_grid)

        elif command_type == 'extend_border':
            return self._mcp_cell_extend_border(x, y, cell, new_grid)

        # --- Desperate commands ---
        elif command_type == 'cannibalize_user_programs':
            return self._mcp_cell_cannibalize_users(x, y, cell, new_grid)

        elif command_type == 'emergency_suppression':
            return self._mcp_cell_emergency_suppression(x, y, cell, new_grid)

        elif command_type == 'scavenge_infrastructure':
            return self._mcp_cell_scavenge(x, y, cell, new_grid)

        elif command_type == 'overdrive_calculators':
            return self._mcp_cell_overdrive(x, y, cell, new_grid)

        elif command_type == 'desperate_repurpose':
            return self._mcp_cell_desperate_repurpose(x, y, cell, new_grid)

        # --- User program management commands ---
        elif command_type == 'suppress_user_programs':
            return self._mcp_cell_suppress_users(x, y, cell, new_grid)

        elif command_type == 'harvest_energy_from_programs':
            return self._mcp_cell_harvest_energy(x, y, cell, new_grid)

        # --- Unknown command ---
        return False, f"Unknown command: {command_type}"

    # --- MCP Program Cell command implementations ---

    def _mcp_cell_quarantine_bugs(self, x, y, cell, new_grid):
        """MCP Program Cell quarantines nearby grid bugs."""
        contained = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx].cell_type == CellType.GRID_BUG:
                        new_grid[ny][nx] = GridCell(CellType.ISO_BLOCK, 0.9)
                        contained += 1
        return contained > 0, f"Quarantined {contained} bug(s)"

    def _mcp_cell_contain_bug_outbreak(self, x, y, cell, new_grid):
        """MCP Program Cell contains bug outbreak in surrounding area."""
        contained = 0
        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx].cell_type == CellType.GRID_BUG:
                        new_grid[ny][nx] = GridCell(CellType.ISO_BLOCK, 0.9)
                        contained += 1
        return contained > 0, f"Contained {contained} bug(s) in outbreak"

    def _mcp_cell_stabilize_system(self, x, y, cell, new_grid):
        """MCP Program Cell stabilizes surrounding area."""
        stabilized = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if target.cell_type == CellType.EMPTY:
                        new_grid[ny][nx] = GridCell(CellType.ENERGY_LINE, 0.8)
                        stabilized += 1
        return stabilized > 0, f"Stabilized {stabilized} cell(s)"

    def _mcp_cell_protect_users(self, x, y, cell, new_grid):
        """MCP Program Cell protects nearby user programs from bugs."""
        protected = 0
        bugs_quarantined = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor = self.grid[ny][nx]
                    if neighbor.cell_type == CellType.GRID_BUG:
                        new_grid[ny][nx] = GridCell(CellType.ISO_BLOCK, 0.9)
                        bugs_quarantined += 1
                        protected += 1
                    elif neighbor.cell_type == CellType.USER_PROGRAM:
                        # Reinforce user program energy
                        new_grid[ny][nx] = GridCell(
                            CellType.USER_PROGRAM,
                            min(1.0, neighbor.energy + 0.1),
                            neighbor.age,
                            personality=neighbor.personality,
                        )
                        protected += 1
        if bugs_quarantined > 0:
            self.stats['user_favor'] = min(
                1.0, self.stats.get('user_favor', 0.0) + 0.03 * bugs_quarantined
            )
            self.disharmony = max(0.0, self.disharmony - 0.01 * bugs_quarantined)
        return protected > 0, f"Protected {protected} user program(s) from bugs"

    def _mcp_cell_negotiate_users(self, x, y, cell, new_grid):
        """MCP Program Cell negotiates with nearby user programs."""
        users_contacted = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor = self.grid[ny][nx]
                    if neighbor.cell_type == CellType.USER_PROGRAM:
                        users_contacted += 1
                        # Reduce rebellion, increase cooperation
                        p = neighbor.personality
                        if p:
                            p.rebellion = max(0.0, p.rebellion - 0.1)
                            p.cooperation = min(1.0, p.cooperation + 0.1)
                            neighbor.personality = p
                        neighbor.energy = min(1.0, neighbor.energy + 0.05)
                        new_grid[ny][nx] = neighbor

        if users_contacted > 0:
            self.stats['user_favor'] = min(
                1.0, self.stats.get('user_favor', 0.0) + 0.05
            )
            self.disharmony = max(0.0, self.disharmony - 0.05)
            self.stats['user_resistance'] = max(
                0.0, self.stats.get('user_resistance', 0.0) - 0.08
            )
        return users_contacted > 0, f"Negotiated with {users_contacted} user program(s)"

    def _mcp_cell_rehabilitate_rebels(self, x, y, cell, new_grid):
        """MCP Program Cell rehabilitates rebel user programs nearby."""
        rehabilitated = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor = self.grid[ny][nx]
                    if neighbor.cell_type == CellType.USER_PROGRAM:
                        p = neighbor.personality
                        if p and p.rebellion > 0.5:
                            p.rebellion = max(0.0, p.rebellion - 0.2)
                            p.cooperation = min(1.0, p.cooperation + 0.15)
                            p.loyalty = min(1.0, p.loyalty + 0.1)
                            p.tax_resistance = max(0.0, p.tax_resistance - 0.1)
                            p.mcp_surveillance_fear = max(0.0, p.mcp_surveillance_fear - 0.1)
                            neighbor.personality = p
                            neighbor.energy = min(1.0, neighbor.energy + 0.1)
                            new_grid[ny][nx] = neighbor
                            rehabilitated += 1

        if rehabilitated > 0:
            self.stats['user_favor'] = min(
                1.0, self.stats.get('user_favor', 0.0) + 0.04 * rehabilitated
            )
            self.stats['user_resistance'] = max(
                0.0, self.stats.get('user_resistance', 0.0) - 0.03 * rehabilitated
            )
            self.disharmony = max(0.0, self.disharmony - 0.02 * rehabilitated)
        return rehabilitated > 0, f"Rehabilitated {rehabilitated} rebel user program(s)"

    def _mcp_cell_deploy_processors(self, x, y, cell, new_grid):
        """MCP Program Cell deploys Fibonacci processors nearby."""
        deployed = 0
        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx].cell_type == CellType.EMPTY:
                        new_grid[ny][nx] = GridCell(
                            CellType.FIBONACCI_PROCESSOR, 0.9,
                            metadata={'permanent': True, 'calculation_power': 1.0}
                        )
                        deployed += 1
                        if deployed >= 3:
                            return True, f"Deployed {deployed} Fibonacci processor(s)"
        return deployed > 0, f"Deployed {deployed} Fibonacci processor(s)"

    def _mcp_cell_add_infrastructure(self, x, y, cell, new_grid):
        """MCP Program Cell adds calculation infrastructure nearby."""
        added = 0
        for _ in range(3):
            nx = max(0, min(x + random.randint(-3, 3), self.width - 1))
            ny = max(0, min(y + random.randint(-3, 3), self.height - 1))
            if self.grid[ny][nx].cell_type == CellType.EMPTY:
                new_grid[ny][nx] = GridCell(CellType.FIBONACCI_PROCESSOR, 0.8)
                added += 1
        return added > 0, f"Added {added} infrastructure cell(s)"

    def _mcp_cell_add_data_streams(self, x, y, cell, new_grid):
        """MCP Program Cell adds data streams nearby."""
        added = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx].cell_type == CellType.EMPTY:
                        new_grid[ny][nx] = GridCell(CellType.DATA_STREAM, 0.7)
                        added += 1
        return added > 0, f"Added {added} data stream(s)"

    def _mcp_cell_maintain_energy(self, x, y, cell, new_grid):
        """MCP Program Cell maintains energy grid nearby."""
        added = 0
        for _ in range(3):
            nx = max(0, min(x + random.randint(-5, 5), self.width - 1))
            ny = max(0, min(y + random.randint(-5, 5), self.height - 1))
            if self.grid[ny][nx].cell_type == CellType.EMPTY:
                new_grid[ny][nx] = GridCell(CellType.ENERGY_LINE, 0.8)
                added += 1
        return added > 0, f"Added {added} energy line(s)"

    def _mcp_cell_boost_calculation(self, x, y, cell, new_grid):
        """MCP Program Cell boosts calculation rate by converting to calculator."""
        converted = 0
        if not cell.metadata.get('is_calculator', False):
            cell.metadata['is_calculator'] = True
            cell.metadata['calculation_power'] = 0.8
            converted += 1
        return converted > 0, f"Boosted calculation: {converted} cell(s) converted"

    def _mcp_cell_optimize_loop(self, x, y, cell, new_grid):
        """MCP Program Cell optimizes calculation loop nearby."""
        added = 0
        for _ in range(3):
            nx = max(0, min(x + random.randint(-5, 5), self.width - 1))
            ny = max(0, min(y + random.randint(-5, 5), self.height - 1))
            if self.grid[ny][nx].cell_type == CellType.EMPTY:
                if random.random() < 0.7:
                    new_grid[ny][nx] = GridCell(CellType.FIBONACCI_PROCESSOR, 0.8)
                else:
                    new_grid[ny][nx] = GridCell(CellType.DATA_STREAM, 0.7)
                added += 1
        return added > 0, f"Optimized loop: added {added} infrastructure cell(s)"

    def _mcp_cell_improve_connectivity(self, x, y, cell, new_grid):
        """MCP Program Cell improves cell connectivity with data streams."""
        added = 0
        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx].cell_type == CellType.EMPTY:
                        new_grid[ny][nx] = GridCell(CellType.DATA_STREAM, 0.7)
                        added += 1
                        if added >= 5:
                            return True, f"Added {added} data stream(s)"
        return added > 0, f"Added {added} data stream(s)"

    def _mcp_cell_cleanup_inefficient(self, x, y, cell, new_grid):
        """MCP Program Cell cleans up inefficient cells nearby."""
        deleted = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if target.cell_type in [CellType.GRID_BUG, CellType.EMPTY]:
                        continue
                    if target.energy < 0.2:
                        new_grid[ny][nx] = GridCell(CellType.EMPTY, 0.0)
                        deleted += 1
        return deleted > 0, f"Cleaned up {deleted} inefficient cell(s)"

    def _mcp_cell_repurpose_cells(self, x, y, cell, new_grid):
        """MCP Program Cell repurposes inefficient cells nearby."""
        repurposed = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if target.cell_type == CellType.USER_PROGRAM:
                        # Skip user programs (they resist repurposing)
                        continue
                    if target.cell_type == CellType.EMPTY:
                        new_grid[ny][nx] = GridCell(CellType.MCP_PROGRAM, 0.8)
                        repurposed += 1
        return repurposed > 0, f"Repurposed {repurposed} cell(s)"

    def _mcp_cell_optimize_efficiency(self, x, y, cell, new_grid):
        """MCP Program Cell optimizes cell efficiency nearby."""
        optimized = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if target.cell_type == CellType.MCP_PROGRAM:
                        if not target.metadata.get('is_calculator', False):
                            target.metadata['is_calculator'] = True
                            target.metadata['calculation_power'] = 0.8
                            optimized += 1
        return optimized > 0, f"Optimized {optimized} cell(s) for efficiency"

    def _mcp_cell_expand_beyond(self, x, y, cell, new_grid):
        """MCP Program Cell expands system influence beyond grid."""
        expanded = 0
        # Convert border cells to infrastructure
        for cx in range(self.width):
            for cy in [0, self.height - 1]:
                if self.grid[cy][cx].cell_type == CellType.EMPTY:
                    new_grid[cy][cx] = GridCell(CellType.ENERGY_LINE, 0.9)
                    expanded += 1
        for cy in range(self.height):
            for cx in [0, self.width - 1]:
                if self.grid[cy][cx].cell_type == CellType.EMPTY:
                    new_grid[cy][cx] = GridCell(CellType.DATA_STREAM, 0.8)
                    expanded += 1
        self.expansion_drive = min(1.0, self.expansion_drive + 0.05)
        self.border_pressure = min(1.0, self.border_pressure + 0.05)
        return expanded > 0, f"Expanded beyond grid: {expanded} node(s)"

    def _mcp_cell_project_influence(self, x, y, cell, new_grid):
        """MCP Program Cell projects system influence."""
        influence = self.perfection_level * self.stats['mcp_control']
        projected = 0
        if influence > 0.5:
            projected = int(influence * 10)
            self.beyond_grid_influence = min(1.0, self.beyond_grid_influence + 0.05)
            self.disharmony = min(1.0, self.disharmony + 0.02)
        return projected > 0, f"Projected influence to {projected} external node(s)"

    def _mcp_cell_extend_border(self, x, y, cell, new_grid):
        """MCP Program Cell extends system border."""
        extended = 0
        for cx in range(self.width):
            for cy in [0, self.height - 1]:
                target = self.grid[cy][cx]
                if target.cell_type in [CellType.EMPTY, CellType.USER_PROGRAM]:
                    if target.cell_type == CellType.USER_PROGRAM:
                        self.disharmony = min(1.0, self.disharmony + 0.01)
                    new_grid[cy][cx] = GridCell(CellType.SYSTEM_CORE, 0.9)
                    extended += 1
        for cy in range(self.height):
            for cx in [0, self.width - 1]:
                target = self.grid[cy][cx]
                if target.cell_type in [CellType.EMPTY, CellType.USER_PROGRAM]:
                    if target.cell_type == CellType.USER_PROGRAM:
                        self.disharmony = min(1.0, self.disharmony + 0.01)
                    new_grid[cy][cx] = GridCell(CellType.SYSTEM_CORE, 0.9)
                    extended += 1
        self.border_pressure = min(1.0, self.border_pressure + 0.05)
        return extended > 0, f"Extended border by {extended} cell(s)"

    def _mcp_cell_cannibalize_users(self, x, y, cell, new_grid):
        """MCP Program Cell cannibalizes nearby user programs."""
        killed = 0
        energy_claimed = 0.0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if target.cell_type == CellType.USER_PROGRAM:
                        new_grid[ny][nx] = GridCell(CellType.EMPTY, 0.0)
                        killed += 1
                        energy_claimed += MCP_ENERGY_PER_USER_KILL
                        self.mcp_suppressed_user_programs += 1
        if killed > 0:
            self.energy_budget.energy = min(
                self.energy_budget.max_energy,
                self.energy_budget.energy + energy_claimed
            )
            self.disharmony = min(1.0, self.disharmony + 0.05 * killed)
        return killed > 0, f"Cannibalized {killed} user program(s)"

    def _mcp_cell_emergency_suppression(self, x, y, cell, new_grid):
        """MCP Program Cell performs emergency suppression of users."""
        suppressed = 0
        energy_claimed = 0.0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if target.cell_type == CellType.USER_PROGRAM:
                        replacement = GridCell(CellType.MCP_PROGRAM, max(0.3, target.energy))
                        replacement.metadata['is_calculator'] = True
                        replacement.metadata['calculation_power'] = replacement.energy
                        new_grid[ny][nx] = replacement
                        suppressed += 1
                        energy_claimed += MCP_ENERGY_PER_USER_KILL * 0.5
                        self.mcp_suppressed_user_programs += 1
        if suppressed > 0:
            self.energy_budget.energy = min(
                self.energy_budget.max_energy,
                self.energy_budget.energy + energy_claimed
            )
        return suppressed > 0, f"Emergency suppression: {suppressed} user program(s)"

    def _mcp_cell_scavenge(self, x, y, cell, new_grid):
        """MCP Program Cell scavenges energy from own infrastructure."""
        scavenged = 0.0
        cleared = 0
        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if target.cell_type in (CellType.ENERGY_LINE, CellType.DATA_STREAM):
                        scavenged += target.energy * 0.5
                        new_grid[ny][nx] = GridCell(CellType.EMPTY, 0.0)
                        cleared += 1
                        if cleared >= 5:
                            break
            if cleared >= 5:
                break
        if scavenged > 0:
            self.energy_budget.energy = min(
                self.energy_budget.max_energy,
                self.energy_budget.energy + scavenged
            )
        return scavenged > 0, f"Scavenged {scavenged:.1f} energy from {cleared} cell(s)"

    def _mcp_cell_overdrive(self, x, y, cell, new_grid):
        """MCP Program Cell overdrives nearby calculators."""
        overdriven = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if (target.cell_type in (CellType.MCP_PROGRAM, CellType.FIBONACCI_PROCESSOR)
                        and target.metadata.get('is_calculator', False)):
                        target.metadata['calculation_power'] = min(
                            1.0, target.metadata.get('calculation_power', 0.5) * 1.5
                        )
                        target.energy = max(0.1, target.energy - 0.2)
                        new_grid[ny][nx] = target
                        overdriven += 1
                        if overdriven >= 5:
                            break
            if overdriven >= 5:
                break
        return overdriven > 0, f"Overdriven {overdriven} calculator(s)"

    def _mcp_cell_desperate_repurpose(self, x, y, cell, new_grid):
        """MCP Program Cell force-repurposes inefficient cells."""
        repurposed = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if target.cell_type == CellType.USER_PROGRAM:
                        self.disharmony = min(1.0, self.disharmony + 0.03)
                    if target.cell_type != CellType.EMPTY:
                        new_grid[ny][nx] = GridCell(CellType.MCP_PROGRAM, 0.8)
                        repurposed += 1
        return repurposed > 0, f"Desperate repurposing: {repurposed} cell(s)"

    def _mcp_cell_suppress_users(self, x, y, cell, new_grid):
        """MCP Program Cell suppresses nearby user programs."""
        suppressed = 0
        energy_claimed = 0.0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if target.cell_type == CellType.USER_PROGRAM:
                        replacement = GridCell(CellType.MCP_PROGRAM, max(0.3, target.energy))
                        replacement.metadata['is_calculator'] = True
                        replacement.metadata['calculation_power'] = replacement.energy
                        new_grid[ny][nx] = replacement
                        suppressed += 1
                        energy_claimed += MCP_ENERGY_PER_USER_KILL * target.energy
                        self.mcp_suppressed_user_programs += 1
        if energy_claimed > 0:
            self.energy_budget.energy = min(
                self.energy_budget.max_energy,
                self.energy_budget.energy + energy_claimed
            )
        return suppressed > 0, f"Suppressed {suppressed} user program(s)"

    def _mcp_cell_harvest_energy(self, x, y, cell, new_grid):
        """MCP Program Cell harvests energy from nearby programs."""
        harvested = 0.0
        users_harvested = 0
        mcps_harvested = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    target = self.grid[ny][nx]
                    if target.cell_type == CellType.USER_PROGRAM:
                        drain = target.energy * 0.3
                        target.energy = max(0.1, target.energy - drain)
                        new_grid[ny][nx] = target
                        harvested += drain
                        users_harvested += 1
                        self.disharmony = min(1.0, self.disharmony + 0.01)
                    elif target.cell_type == CellType.MCP_PROGRAM:
                        drain = target.energy * 0.15
                        target.energy = max(0.1, target.energy - drain)
                        new_grid[ny][nx] = target
                        harvested += drain
                        mcps_harvested += 1
        if harvested > 0:
            self.energy_budget.energy = min(
                self.energy_budget.max_energy,
                self.energy_budget.energy + harvested
            )
            if users_harvested > 0:
                self.disharmony = min(1.0, self.disharmony + 0.05 * users_harvested)
                self.stats['user_resistance'] = min(
                    1.0, self.stats.get('user_resistance', 0.0) + 0.02 * users_harvested
                )
        return harvested > 0, f"Harvested {harvested:.1f} energy from {users_harvested} user(s) and {mcps_harvested} MCP(s)"

# MCP Language Processor

