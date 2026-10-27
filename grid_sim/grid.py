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
)
from .fibonacci import GridFibonacciCalculator

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
                    if random.random() < revolt_chance:
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
                            if random.random() < revolt_resistance:
                                # MCP resisted the revolt - user program stays in place
                                new_grid[y][x] = GridCell(
                                    CellType.USER_PROGRAM,
                                    max(0.2, cell.energy - 0.1),
                                    cell.age + 1
                                )
                                continue
                            new_grid[ty][tx] = GridCell(
                                CellType.USER_PROGRAM, min(0.8, cell.energy * 0.8)
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
                                CellType.USER_PROGRAM, min(0.8, cell.energy * 0.8)
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

                # USER PROGRAMS - May resist optimal configuration
                if cell.cell_type == CellType.USER_PROGRAM:
                    # User programs sometimes work against optimal loop
                    if random.random() < self.stats['user_resistance'] * 0.3:
                        # Move randomly instead of optimally
                        dx, dy = random.choice([(-1,0), (1,0), (0,-1), (0,1)])
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            target_cell = self.grid[ny][nx]
                            if (target_cell.cell_type == CellType.EMPTY and
                                (nx, ny) not in reserved_user_cells):
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
                            if (target_cell.cell_type == CellType.EMPTY and
                                (nx, ny) not in reserved_user_cells):
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

                    # Stay in place, efficient energy usage
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

        # Express control and resistance as shares of the competing program population.
        total_control_programs = counts[CellType.MCP_PROGRAM] + counts[CellType.USER_PROGRAM]
        mcp_control = (
            counts[CellType.MCP_PROGRAM] / total_control_programs
            if total_control_programs else 0.5
        )
        user_resistance = (
            counts[CellType.USER_PROGRAM] / total_control_programs
            if total_control_programs else 0.0
        )

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
            'expansion_drive': self.expansion_drive,
            'beyond_grid_influence': self.beyond_grid_influence,
            'perfection_level': self.perfection_level,
            'disharmony': self.disharmony,
            'imperfection_count': self.imperfection_count,
            'border_pressure': self.border_pressure,
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

            # Special initialization for certain cell types
            if cell_type == CellType.MCP_PROGRAM and random.random() < 0.4:
                cell.metadata['is_calculator'] = True
                cell.metadata['calculation_power'] = energy

            self.grid[y][x] = cell
            self.update_stats()
            return True
        return False

# MCP Language Processor

