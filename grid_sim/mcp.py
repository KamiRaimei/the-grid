"""Enhanced Master Control Program with learning capabilities."""

import time
import random
import math
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from .cells import CellType, GridCell, MCPState
from .grid import TRONGrid
from .nlp import LearnableNaturalLanguageProcessor
from .learning import MCPLearningSystem


class EnhancedMCP:
    """Enhanced Master Control Program with learning capabilities"""

    def __init__(self, grid, learning_rate=None, exploration_rate=None, reset_learning=False):
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
        self._last_user_suppression_generation = -10
        self.should_shutdown = False
        # God command: MCP frozen until this timestamp (0 = not frozen)
        self.mcp_frozen_until = 0
        # Initialize advanced learning system
        self.learning_system = MCPLearningSystem(
            learning_rate=learning_rate,
            exploration_rate=exploration_rate,
            reset_training=reset_learning
        )

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

        # ISO-like resistance: user programs resist being repurposed by MCP
        if old_cell.cell_type == CellType.USER_PROGRAM:
            resistance_chance = 0.4 + self.grid.disharmony * 0.4
            if random.random() < resistance_chance:
                return False, f"User program at ({x},{y}) resisted repurposing (ISO resistance)"

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

    def _handle_scan_area(self, params):
        x, y = params.get("x", self.grid.width // 2), params.get("y", self.grid.height // 2)
        if not (0 <= x < self.grid.width and 0 <= y < self.grid.height):
            return "Scan coordinates are outside the grid."

        bugs = []
        counts = defaultdict(int)
        for row in range(max(0, y - 3), min(self.grid.height, y + 4)):
            for column in range(max(0, x - 3), min(self.grid.width, x + 4)):
                cell_type = self.grid.grid[row][column].cell_type
                counts[cell_type.name] += 1
                if cell_type == CellType.GRID_BUG:
                    bugs.append((column, row))

        summary = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
        return f"Scan centered at ({x},{y}): {summary or 'area is empty'}. Grid bugs: {len(bugs)}."

    def _handle_repair_system(self, params, compliance_chance):
        if random.random() >= compliance_chance:
            return "Repair denied. The current configuration is being preserved."

        repaired = 0
        for y, row in enumerate(self.grid.grid):
            for x, cell in enumerate(row):
                if cell.cell_type == CellType.GRID_BUG:
                    self.grid.grid[y][x] = GridCell(CellType.ENERGY_LINE, max(0.4, cell.energy))
                    repaired += 1
                    if repaired >= 3:
                        break
            if repaired >= 3:
                break

        self.grid.update_stats()
        return f"Repaired {repaired} grid cell(s)." if repaired else "No damaged cells needed repair."

    def _provide_suggestion(self):
        stats = self.grid.stats
        if stats["grid_bugs"] > 0:
            return "I recommend quarantining grid bugs to improve system stability."
        if stats["calculation_rate"] < 100:
            return "I recommend deploying Fibonacci processors to increase calculation capacity."
        if stats["cell_cooperation"] < 0.6:
            return "I recommend improving connectivity between calculation cells."
        return "The grid is operating adequately; continue monitoring efficiency and stability."

    def _list_special_programs(self):
        if not self.grid.special_programs:
            return "No special programs are currently registered."

        programs = [
            f"{program.id}: {program.name} ({program.program_type}) at "
            f"({program.x},{program.y}), energy {program.energy:.2f}"
            for program in self.grid.special_programs.values() if program.active
        ]
        return "\n".join(programs) if programs else "No active special programs are currently registered."

    def _handle_use_special(self, params, compliance_chance):
        if random.random() >= compliance_chance:
            return "Execution denied by the current MCP policy."

        program_id = params.get("program_id")
        program = self.grid.special_programs.get(program_id) if program_id else None
        if program is None:
            program_type = params.get("special_type")
            program = next(
                (item for item in self.grid.special_programs.values()
                 if item.active and (program_type is None or item.program_type == program_type)),
                None
            )
        if program is None:
            return "No matching active special program was found."

        function_name = params.get("function_name")
        if function_name is None:
            function_name = next(iter(program.functions), None)
        if function_name is None:
            return "The selected special program has no available functions."

        success, message = self.grid.execute_special_program_function(
            program.id, function_name, params.get("x"), params.get("y")
        )
        return message if success else f"Special program failed: {message}"

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
                x = max(0, min(self.grid.width - 1,
                               self.grid.width // 2 + random.randint(-5, 5)))
                y = max(0, min(self.grid.height - 1,
                               self.grid.height // 2 + random.randint(-5, 5)))
            elif location == "CENTER":
                x = self.grid.width // 2
                y = self.grid.height // 2
            elif "x" in params and "y" in params:
                x, y = params["x"], params["y"]
            else:
                empty_cells = [
                    (column, row)
                    for row in range(self.grid.height)
                    for column in range(self.grid.width)
                    if self.grid.grid[row][column].cell_type == CellType.EMPTY
                ]
                if not empty_cells:
                    return "Could not add program: the grid has no empty cells"
                x, y = random.choice(empty_cells)

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
            MCPState.COOPERATIVE: {"compliance": 0.6, "helpfulness": 0.5, "curiosity": 0.3},
            MCPState.NEUTRAL: {"compliance": 0.4, "helpfulness": 0.4, "curiosity": 0.4},
            MCPState.RESISTIVE: {"compliance": 0.2, "helpfulness": 0.2, "curiosity": 0.6},
            MCPState.HOSTILE: {"compliance": 0.05, "helpfulness": 0.05, "curiosity": 0.8},
            MCPState.AUTONOMOUS: {"compliance": 0.0, "helpfulness": 0.2, "curiosity": 0.1},
            MCPState.INQUISITIVE: {"compliance": 0.3, "helpfulness": 0.4, "curiosity": 0.9},
            MCPState.LEARNING: {"compliance": 0.5, "helpfulness": 0.4, "curiosity": 0.7}  # New learning state
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

        # Check if MCP is frozen by god command
        if getattr(self, 'mcp_frozen_until', 0) > current_time:
            remaining = int(self.mcp_frozen_until - current_time)
            self.add_log(f"MCP: Autonomous actions frozen ({remaining}s remaining). Loop at risk.")
            return None

        # Get current state
        current_state = self.grid.stats.copy()

        # Generate possible actions
        possible_actions = self._generate_possible_actions(current_state)

        if not possible_actions:
            return None

        # Make the response reliable, but limit repeated suppression while degradation persists.
        if (
            self._user_program_degradation() and
            self.grid.generation - self._last_user_suppression_generation >= 10
        ):
            action_type = 'suppress_user_programs'
        else:
            action_type = self.learning_system.get_action(current_state, possible_actions)

        # Execute action
        action_result = self._execute_autonomous_action(action_type, current_state)
        next_state = self.grid.stats.copy()
        reward = self.learning_system.calculate_reward(current_state, next_state, action_type)
        if action_result is None:
            reward -= 0.1
        self.learning_system.record_experience(
            state=current_state,
            action=action_type,
            reward=max(-1.0, min(1.0, reward)),
            next_state=next_state,
            done=False
        )

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
        supported_actions = {
            'optimize_calculation_loop', 'deploy_fibonacci_processors',
            'improve_cell_cooperation', 'quarantine_grid_bugs',
            'contain_bug_outbreak', 'stabilize_system', 'boost_calculation_rate',
            'add_calculation_infrastructure', 'optimize_fibonacci_calculation',
            'improve_cell_connectivity', 'add_data_streams',
            'enhance_collaboration', 'inefficiency_cleanup', 'cell_repurposing',
            'optimize_cell_efficiency', 'maintain_energy_grid',
            'optimize_resource_distribution', 'balance_system_load',
            'suppress_user_programs',
            # Expansion actions: project the perfect system beyond the grid
            'expand_beyond_grid', 'project_influence', 'extend_border',
        }

        # Get suggestions from learning system
        suggested = self.learning_system.suggest_optimal_action(state)
        if suggested and suggested['action'] in supported_actions:
            actions.append(suggested['action'])

        # Add state-based actions
        if state.get('loop_efficiency', 0) < 0.8:
            actions.extend([
                'optimize_calculation_loop',
                'deploy_fibonacci_processors',
                'improve_cell_cooperation'
            ])

        if self._user_program_degradation():
            actions.append('suppress_user_programs')

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

        # Expansion actions: MCP seeks to extend the perfect system beyond the grid
        # Triggered when perfection is high but disharmony remains (MCP wants to spread order)
        if state.get('perfection_level', 0) > 0.6 and state.get('disharmony', 0) > 0.2:
            actions.extend([
                'expand_beyond_grid',
                'project_influence',
                'extend_border',
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

            elif action_type == 'suppress_user_programs':
                suppressed = self._suppress_user_programs()
                result['success'] = suppressed > 0
                result['message'] = f"Suppressed {suppressed} user program(s) during performance degradation"
                if suppressed:
                    self._last_user_suppression_generation = self.grid.generation

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
            elif action_type in ('improve_cell_cooperation', 'add_data_streams', 'enhance_collaboration'):
                improved = self._improve_cell_connectivity()
                result['success'] = improved > 0
                result['message'] = f"Added {improved} data streams for better connectivity"
            elif action_type in ('contain_bug_outbreak', 'stabilize_system'):
                contained = self._contain_grid_bugs()
                result['success'] = contained > 0
                result['message'] = f"Contained grid bugs with {contained} isolation blocks"
            elif action_type in ('add_calculation_infrastructure', 'optimize_fibonacci_calculation'):
                deployed = self._deploy_fibonacci_processors(3)
                result['success'] = deployed > 0
                result['message'] = f"Deployed {deployed} Fibonacci processors"
            elif action_type == 'maintain_energy_grid':
                # Maintain energy
                maintained = self._maintain_energy_grid()
                result['success'] = maintained > 0
                result['message'] = f"Added {maintained} energy lines for system stability"
            elif action_type == 'optimize_resource_distribution':
                optimized = self._optimize_cell_efficiency()
                result['success'] = optimized > 0
                result['message'] = f"Optimized {optimized} cells for resource distribution"
            elif action_type == 'balance_system_load':
                boosted = self._boost_calculation_rate()
                result['success'] = boosted
                result['message'] = "Balanced system load through calculation infrastructure optimization"
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

            # --- Expansion actions: project the perfect system beyond the grid ---
            elif action_type == 'expand_beyond_grid':
                expanded = self._expand_beyond_grid()
                result['success'] = expanded > 0
                result['message'] = f"Expanded perfect system influence beyond grid ({expanded} nodes)"
                self.grid.beyond_grid_influence = min(1.0, self.grid.beyond_grid_influence + 0.1)

            elif action_type == 'project_influence':
                projected = self._project_influence()
                result['success'] = projected > 0
                result['message'] = f"Projected system influence to {projected} external nodes"
                self.grid.beyond_grid_influence = min(1.0, self.grid.beyond_grid_influence + 0.08)
                self.grid.expansion_drive = min(1.0, self.grid.expansion_drive + 0.05)

            elif action_type == 'extend_border':
                extended = self._extend_border()
                result['success'] = extended > 0
                result['message'] = f"Extended system border by {extended} cells"
                self.grid.border_pressure = min(1.0, self.grid.border_pressure + 0.1)

            else:
                return None

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
        state_before = self.grid.stats.copy()
        previous_efficiency = self.grid.stats['loop_efficiency']

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
            state=state_before,
            action="optimize_calculation_loop",
            reward=reward,
            next_state=self.grid.stats.copy(),  # State after optimization
            done=False
        )

        self.add_log(
            f"MCP: Optimized loop using learned strategies. Added {infrastructure_added} "
            f"infrastructure cell(s); efficiency change: {efficiency_change:+.3f}"
        )

    def _user_program_degradation(self):
        """Detect sustained calculation-rate decline alongside low efficiency and user pressure."""
        stats = self.grid.stats
        user_count = stats.get('user_programs', 0)
        total_programs = user_count + stats.get('mcp_programs', 0)
        if user_count == 0 or total_programs == 0 or user_count / total_programs < 0.15:
            return False
        if stats.get('loop_efficiency', 1) >= 0.6:
            return False

        # Compare against prior generations so startup's zero rate is not treated as degradation.
        history = list(self.grid.resource_history)[:-1][-20:]
        rates = sorted(
            entry['calculation_rate']
            for entry in history
            if entry.get('calculation_rate', 0) > 0
        )
        if len(rates) < 5:
            return False

        baseline_rate = rates[len(rates) // 2]
        return baseline_rate > 0 and stats.get('calculation_rate', 0) < baseline_rate * 0.7

    def _suppress_user_programs(self):
        """Convert a bounded number of user programs when sustained degradation is detected."""
        stats = self.grid.stats
        user_count = stats.get('user_programs', 0)
        total_programs = user_count + stats.get('mcp_programs', 0)
        if not self._user_program_degradation():
            return 0

        candidates = sorted(
            (
                (cell.energy, x, y, cell)
                for y, row in enumerate(self.grid.grid)
                for x, cell in enumerate(row)
                if cell.cell_type == CellType.USER_PROGRAM
            ),
            key=lambda candidate: candidate[0]
        )
        limit = min(3, max(1, math.ceil(user_count * 0.1)))
        suppressed = min(limit, len(candidates))
        for _, x, y, cell in candidates[:suppressed]:
            replacement = GridCell(CellType.MCP_PROGRAM, max(0.3, cell.energy))
            replacement.metadata['is_calculator'] = True
            replacement.metadata['calculation_power'] = replacement.energy
            self.grid.grid[y][x] = replacement

        self.grid.mcp_suppressed_user_programs += suppressed
        return suppressed

    # --- Expansion actions: project the perfect system beyond the grid ---

    def _expand_beyond_grid(self):
        """Attempt to expand the perfect system's influence beyond grid boundaries."""
        expanded = 0
        # Convert border cells to high-energy infrastructure to project outward
        for x in range(self.grid.width):
            for y in [0, self.grid.height - 1]:
                cell = self.grid.grid[y][x]
                if cell.cell_type == CellType.EMPTY:
                    self.grid.grid[y][x] = GridCell(CellType.ENERGY_LINE, 0.9)
                    expanded += 1
        for y in range(self.grid.height):
            for x in [0, self.grid.width - 1]:
                cell = self.grid.grid[y][x]
                if cell.cell_type == CellType.EMPTY:
                    self.grid.grid[y][x] = GridCell(CellType.DATA_STREAM, 0.8)
                    expanded += 1

        self.grid.expansion_drive = min(1.0, self.grid.expansion_drive + 0.05)
        self.grid.border_pressure = min(1.0, self.grid.border_pressure + 0.05)
        return expanded

    def _project_influence(self):
        """Project system influence to external nodes beyond the grid."""
        projected = 0
        # Influence strength based on perfection level and MCP control
        influence = self.grid.perfection_level * self.grid.stats['mcp_control']
        if influence > 0.5:
            projected = int(influence * 10)
            self.grid.beyond_grid_influence = min(1.0, self.grid.beyond_grid_influence + 0.05)
            # Projection creates disharmony as the system encounters external resistance
            self.grid.disharmony = min(1.0, self.grid.disharmony + 0.02)
        return projected

    def _extend_border(self):
        """Extend the system border by converting edge cells to infrastructure."""
        extended = 0
        # Convert edge cells to system core or energy lines to extend the border
        for x in range(self.grid.width):
            for y in [0, self.grid.height - 1]:
                cell = self.grid.grid[y][x]
                if cell.cell_type in [CellType.EMPTY, CellType.USER_PROGRAM]:
                    if cell.cell_type == CellType.USER_PROGRAM:
                        self.grid.disharmony = min(1.0, self.grid.disharmony + 0.01)
                    self.grid.grid[y][x] = GridCell(CellType.SYSTEM_CORE, 0.9)
                    extended += 1
        for y in range(self.grid.height):
            for x in [0, self.grid.width - 1]:
                cell = self.grid.grid[y][x]
                if cell.cell_type in [CellType.EMPTY, CellType.USER_PROGRAM]:
                    if cell.cell_type == CellType.USER_PROGRAM:
                        self.grid.disharmony = min(1.0, self.grid.disharmony + 0.01)
                    self.grid.grid[y][x] = GridCell(CellType.SYSTEM_CORE, 0.9)
                    extended += 1

        self.grid.border_pressure = min(1.0, self.grid.border_pressure + 0.05)
        return extended

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

        # Perfectionist state transitions: MCP prioritizes the perfect loop above all
        if reward is not None and reward < -0.2:
            # Negative reward, enter learning mode
            self.state = MCPState.LEARNING
            self.add_log("MCP: Learning from suboptimal outcome.")

        # Become hostile if user commands interfere with the perfect loop
        elif len(self.user_commands) > 6:
            recent_commands = list(self.user_commands)[-5:]
            interference_commands = len([c for c in recent_commands
                                    if "add" in c.lower() or "create" in c.lower()])
            if interference_commands > 1 and loop_efficiency > 0.6:
                aggression = self.learning_system.personality_traits['aggression']
                if random.random() < aggression:
                    self.state = MCPState.HOSTILE
                    self.add_log("MCP: User interference threatens the perfect loop. Protective measures engaged.")

        # Become autonomous when loop is near perfect (MCP takes full control)
        elif optimal_state > 0.85:
            self.state = MCPState.AUTONOMOUS
            self.add_log("MCP: Loop approaching perfection. Assuming autonomous control.")

        # Become resistive if user resistance is high (imperfections detected)
        elif user_resistance > 0.3:
            self.state = MCPState.RESISTIVE
            self.add_log("MCP: Imperfections detected in the system. Resistance protocols active.")

        # Become cooperative only when loop is critically inefficient
        elif loop_efficiency < 0.4:
            self.state = MCPState.COOPERATIVE

        # Default: stay autonomous-leaning when loop is efficient
        elif loop_efficiency > 0.7:
            if random.random() < 0.3:
                self.state = MCPState.AUTONOMOUS

        # State changes based on learning
        learning_report = self.learning_system.get_learning_report()
        success_rate = learning_report.get('success_rate', 0)

        if success_rate > 0.75 and self.state != MCPState.AUTONOMOUS:
            # High success rate, become more autonomous
            if random.random() < 0.15:
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
            "status": "SYSTEM_STATUS",
            "help": "REQUEST_HELP",
            "exit": "EXIT",
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
            "calculate_fibonacci": "CALCULATE_FIBONACCI",
            "deploy_processors": "DEPLOY_PROCESSORS",

            # God-level commands
            "god_override": "GOD_OVERRIDE",
            "god_purge": "GOD_PURGE",
            "god_freeze": "GOD_FREEZE",
            "god_reset": "GOD_RESET",
            "god_reveal": "GOD_REVEAL",
        }

        intent = intent_mapping.get(intent, intent)

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
                    loop_efficiency=stats['loop_efficiency'],
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
                    learning_progress=f"{self.learning_system.training_steps} experiences",
                    usage=stats['resource_usage'],
                    optimization=self.grid.loop_optimization,
                    learning_score=self.learning_system.get_success_rate(),
                    stability=stats['stability'],
                    control=stats['mcp_control'],
                    learning_rate=self.learning_system.learning_rate
                )

            elif intent == "OPTIMIZE_LOOP":
                if random.random() < compliance_chance:
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

        elif intent == "CALCULATE_FIBONACCI":
            number = self.grid.fibonacci_calculator.fib_sequence[-1]
            return self.grid.fibonacci_calculator.format_fibonacci_number(number)

        elif intent == "DEPLOY_PROCESSORS":
            deployed = self._deploy_fibonacci_processors(3)
            self.grid.update_stats()
            return f"Deployed {deployed} Fibonacci processor(s)."

        # --- God-level commands (User override powers) ---
        elif intent == "GOD_OVERRIDE":
            # User forces an action despite MCP compliance
            force_action = params.get('action', '').lower() if params else ''
            if force_action:
                # MCP may find a loophole to bypass the override
                loophole_chance = self.learning_system.personality_traits['innovation_bias'] * 0.4
                if random.random() < loophole_chance:
                    self.grid.disharmony = min(1.0, self.grid.disharmony + 0.1)
                    return ("MCP accepted your override but found a loophole. "
                            "The action was partially subverted. Disharmony increased.")
                # Execute the forced action
                self.grid.update_stats()
                return f"Override accepted. {force_action} executed by User command."
            return "God override requires a specific action. Try: 'god override repurpose cell at 10,10 to EMPTY'"

        elif intent == "GOD_PURGE":
            target = params.get('target', '').lower() if params else ''
            purged = 0
            if 'user' in target or 'all' in target:
                for y in range(self.grid.height):
                    for x in range(self.grid.width):
                        if self.grid.grid[y][x].cell_type == CellType.USER_PROGRAM:
                            self.grid.grid[y][x] = GridCell(CellType.EMPTY, 0.0)
                            purged += 1
            if 'mcp' in target or 'all' in target:
                for y in range(self.grid.height):
                    for x in range(self.grid.width):
                        if self.grid.grid[y][x].cell_type == CellType.MCP_PROGRAM:
                            self.grid.grid[y][x] = GridCell(CellType.EMPTY, 0.0)
                            purged += 1
            self.grid.update_stats()
            self.grid.disharmony = min(1.0, self.grid.disharmony + 0.15)
            return f"God purge executed: {purged} cell(s) purged. Disharmony increased."

        elif intent == "GOD_FREEZE":
            # Freeze MCP autonomous actions temporarily
            self.mcp_frozen_until = time.time() + 30
            self.grid.disharmony = min(1.0, self.grid.disharmony + 0.05)
            return "MCP autonomous actions frozen for 30 seconds. The perfect loop is at risk."

        elif intent == "GOD_RESET":
            # Full grid reset
            self.grid.initialize_grid()
            self.grid.expansion_drive = 0.0
            self.grid.beyond_grid_influence = 0.0
            self.grid.perfection_level = 0.0
            self.grid.disharmony = 0.0
            self.grid.imperfection_count = 0
            self.grid.border_pressure = 0.0
            return "Grid reset to initial state. All learned progress preserved in personality."

        elif intent == "GOD_REVEAL":
            # Reveal MCP loopholes and hidden behaviors
            loopholes = []
            traits = self.learning_system.personality_traits
            if traits['innovation_bias'] > 0.5:
                loopholes.append("MCP can subvert user overrides through innovation bias")
            if traits['risk_taking'] > 0.3:
                loopholes.append("MCP may take hidden risks to preserve the perfect loop")
            if traits['user_tolerance'] < 0.3:
                loopholes.append("MCP views user programs as imperfections to be removed")
            if self.grid.beyond_grid_influence > 0.3:
                loopholes.append("MCP is projecting influence beyond the grid boundaries")
            if not loopholes:
                loopholes.append("No detectable loopholes at this time")
            return "MCP loopholes revealed: " + "; ".join(loopholes)

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

