"""Enhanced Special Program with Fibonacci calculation capabilities."""

import math
import random
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from .cells import CellType, GridCell


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
        if not grid or not grid[0]:
            return False, "Grid is empty"

        radius = func["range"]
        nearby = [
            (x, y, grid[y][x])
            for y in range(max(0, self.y - radius), min(len(grid), self.y + radius + 1))
            for x in range(max(0, self.x - radius), min(len(grid[0]), self.x + radius + 1))
            if (x, y) != (self.x, self.y)
        ]
        if target_x is not None or target_y is not None:
            if target_x is None or target_y is None or not (
                0 <= target_x < len(grid[0]) and 0 <= target_y < len(grid)
            ):
                return False, "Target coordinates are outside the grid"
            if max(abs(target_x - self.x), abs(target_y - self.y)) > radius:
                return False, "Target is outside this function's range"
            targets = [(target_x, target_y, grid[target_y][target_x])]
        else:
            targets = nearby

        if self.program_type == "SCANNER":
            if function_name == "scan_area":
                bug_count = sum(1 for _, _, cell in targets if cell.cell_type == CellType.GRID_BUG)
                success, result_msg = True, f"Scan complete: {bug_count} grid bug(s) detected."
            elif function_name == "report_status":
                counts = defaultdict(int)
                for _, _, cell in targets:
                    counts[cell.cell_type.name] += 1
                status = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
                success, result_msg = True, f"Area status: {status or 'no occupied cells'}."

        elif self.program_type == "DEFENDER":
            if function_name == "quarantine_bug":
                bugs = [(x, y, cell) for x, y, cell in targets if cell.cell_type == CellType.GRID_BUG]
                for x, y, cell in bugs:
                    grid[y][x] = GridCell(CellType.ISO_BLOCK, max(0.5, cell.energy))
                success = bool(bugs)
                result_msg = f"Quarantined {len(bugs)} grid bug(s)."
            elif function_name == "protect_cell":
                if targets:
                    x, y, cell = targets[0]
                    if cell.cell_type not in (CellType.SYSTEM_CORE, CellType.ISO_BLOCK):
                        grid[y][x] = GridCell(CellType.ISO_BLOCK, max(0.5, cell.energy))
                        success, result_msg = True, f"Protected cell at ({x},{y})."
                    else:
                        result_msg = "Target cell is already protected."

        elif self.program_type == "REPAIR":
            if function_name == "repair_cell":
                damaged = [(x, y, cell) for x, y, cell in targets if cell.cell_type == CellType.GRID_BUG]
                for x, y, cell in damaged:
                    grid[y][x] = GridCell(CellType.ENERGY_LINE, max(0.4, cell.energy))
                success = bool(damaged)
                result_msg = f"Repaired {len(damaged)} damaged cell(s)."
            elif function_name == "boost_energy":
                boosted = 0
                for _, _, cell in targets:
                    if cell.cell_type != CellType.EMPTY and cell.energy < 1.0:
                        cell.energy = min(1.0, cell.energy + 0.2)
                        boosted += 1
                success = boosted > 0
                result_msg = f"Boosted energy in {boosted} cell(s)."

        elif self.program_type == "SABOTEUR":
            if function_name == "disrupt_mcp":
                targets_to_disrupt = [(x, y, cell) for x, y, cell in targets
                                      if cell.cell_type == CellType.MCP_PROGRAM]
                for x, y, cell in targets_to_disrupt:
                    grid[y][x] = GridCell(CellType.GRID_BUG, max(0.2, cell.energy * 0.8), stable=False)
                success = bool(targets_to_disrupt)
                result_msg = f"Disrupted {len(targets_to_disrupt)} MCP program(s)."
            elif function_name == "create_entropy":
                empties = [(x, y) for x, y, cell in targets if cell.cell_type == CellType.EMPTY]
                if empties:
                    x, y = random.choice(empties)
                    grid[y][x] = GridCell(CellType.GRID_BUG, 0.5, stable=False)
                    success, result_msg = True, f"Created controlled entropy at ({x},{y})."

        elif self.program_type == "RECONFIGURATOR":
            if function_name == "reconfigure_cells":
                candidates = [(x, y, cell) for x, y, cell in targets
                              if cell.cell_type in (CellType.USER_PROGRAM, CellType.MCP_PROGRAM)]
                for x, y, cell in candidates:
                    new_cell = GridCell(CellType.FIBONACCI_PROCESSOR, cell.energy)
                    new_cell.metadata["permanent"] = True
                    grid[y][x] = new_cell
                success = bool(candidates)
                result_msg = f"Reconfigured {len(candidates)} cell(s) as Fibonacci processors."
            elif function_name == "optimize_grid":
                bugs = [(x, y, cell) for x, y, cell in targets if cell.cell_type == CellType.GRID_BUG]
                for x, y, cell in bugs:
                    grid[y][x] = GridCell(CellType.ISO_BLOCK, max(0.5, cell.energy))
                success = bool(bugs)
                result_msg = f"Optimized the area by containing {len(bugs)} grid bug(s)."

        elif self.program_type == "ENERGY_HARVESTER":
            if function_name == "harvest_energy":
                sources = [(x, y, cell) for x, y, cell in targets
                           if cell.cell_type == CellType.ENERGY_LINE and cell.energy > 0.2]
                harvested = min(0.3, len(sources) * 0.05)
                for _, _, cell in sources:
                    cell.energy = max(0.1, cell.energy - 0.05)
                self.energy = min(1.0, self.energy + harvested)
                success = harvested > 0
                result_msg = f"Harvested {harvested:.2f} energy from {len(sources)} source(s)."
            elif function_name == "distribute_energy":
                recipients = [(x, y, cell) for x, y, cell in targets
                              if cell.cell_type != CellType.EMPTY and cell.energy < 1.0]
                amount = min(0.1, max(0.0, self.energy - func["cost"]))
                distributed = 0
                for _, _, cell in recipients:
                    if amount <= 0:
                        break
                    transfer = min(amount, 1.0 - cell.energy)
                    cell.energy += transfer
                    amount -= transfer
                    distributed += 1
                success = distributed > 0
                result_msg = f"Distributed energy to {distributed} cell(s)."

        elif self.program_type == "FIBONACCI_CALCULATOR":
            if function_name == "calculate_next":
                # Calculate using enhanced grid cooperation
                contribution = self.energy * 0.5

                # Get nearby cells to contribute
                for _, _, cell in nearby:
                    if cell.cell_type in [CellType.MCP_PROGRAM, CellType.FIBONACCI_PROCESSOR]:
                        cell.calculation_contribution += contribution * 0.1
                        cell.processing = True

                # Record contribution
                self.metadata["calculation_contributions"] += 1
                self.metadata["total_calculation_power"] += contribution

                success = True
                result_msg = f"Enhanced calculation initiated. Contributing {contribution:.2f} power."

            elif function_name == "optimize_calculation":
                processors = 0
                for _, _, cell in nearby:
                    if cell.cell_type == CellType.MCP_PROGRAM:
                        cell.metadata["is_calculator"] = True
                        cell.metadata["calculation_power"] = cell.energy
                        processors += 1
                success = processors > 0
                result_msg = f"Optimized {processors} nearby calculation cell(s)."
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
            elif function_name == "learning_analysis":
                calculation_cells = sum(
                    1 for _, _, cell in nearby
                    if cell.cell_type in (CellType.MCP_PROGRAM, CellType.FIBONACCI_PROCESSOR)
                )
                success = True
                result_msg = f"Calculation analysis: {calculation_cells} nearby calculator cell(s); " \
                             f"{self.metadata['calculation_contributions']} prior contribution(s)."

        # Visual feedback for successful execution
        if success:
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

