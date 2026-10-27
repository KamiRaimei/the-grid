"""Enhanced simulation with learning and visual effects."""

import os
import sys
import time
import random
import select
import threading
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

try:
    import curses
    from curses import textpad
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

from .cells import (
    CellType,
    GridCell,
    GRID_WIDTH,
    GRID_HEIGHT,
    UPDATE_INTERVAL,
    MCP_UPDATE_INTERVAL,
    PERSONALITY_FILE,
    MCPState,
)
from .grid import TRONGrid
from .mcp import EnhancedMCP
from .special_programs import SpecialProgram


class EnhancedTRONSimulation:
    """Enhanced simulation with learning and visual effects"""

    def __init__(self, use_curses=True, width=GRID_WIDTH, height=GRID_HEIGHT,
                 learning_rate=None, exploration_rate=None, reset_learning=False):
        self.use_curses = use_curses and CURSES_AVAILABLE
        self.grid = TRONGrid(width, height)
        self.mcp = EnhancedMCP(
            self.grid,
            learning_rate=learning_rate,
            exploration_rate=exploration_rate,
            reset_learning=reset_learning
        )
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
                            self.running = False
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

        # Column 3 - Perfection and expansion (the struggle)
        col3 = f"""  Perfection Level: {stats['perfection_level']:.2f}
        Disharmony:   {stats['disharmony']:.2f}
        Expansion:    {stats['expansion_drive']:.2f}
        Beyond Grid:  {stats['beyond_grid_influence']:.2f}
        Border Pressure: {stats['border_pressure']:.2f}"""

        # Column 4 - Energy budget & hardware resources
        energy_ratio = stats.get('energy_ratio', 0.0)
        energy = stats.get('energy', 0.0)
        max_energy = self.grid.energy_budget.max_energy
        is_scarce = stats.get('is_scarce', False)
        is_critical = stats.get('is_critical', False)
        cost_mult = stats.get('action_cost_multiplier', 1.0)
        cpu_pct = stats.get('hardware_cpu_percent', 0.0)
        mem_pct = stats.get('hardware_memory_percent', 0.0)
        disk_pct = stats.get('hardware_disk_percent', 0.0)
        resource_pressure = stats.get('resource_pressure', 0.0)

        energy_status = "CRITICAL" if is_critical else "SCARCE" if is_scarce else "NOMINAL"
        col4 = f"""  Energy Budget: {energy:.1f}/{max_energy:.0f}
        Energy Ratio: {energy_ratio:.1%} [{energy_status}]
        Action Cost: {cost_mult:.2f}x multiplier
        Resource Pressure: {resource_pressure:.1%}
        Host CPU: {cpu_pct:.1f}%
        Host RAM: {mem_pct:.1f}%
        Host Disk: {disk_pct:.1f}%"""

        # Print columns
        col1_lines = col1.split('\n')
        col2_lines = col2.split('\n')
        col3_lines = col3.split('\n')
        col4_lines = col4.split('\n')

        for i in range(max(len(col1_lines), len(col2_lines), len(col3_lines), len(col4_lines))):
            line1 = col1_lines[i] if i < len(col1_lines) else ""
            line2 = col2_lines[i] if i < len(col2_lines) else ""
            line3 = col3_lines[i] if i < len(col3_lines) else ""
            line4 = col4_lines[i] if i < len(col4_lines) else ""
            print(f"{line1:<28} {line2:<28} {line3:<28} {line4}")

        # Add spacing before Fibonacci section to prevent overlap with Energy column
        print()
        print()

        # Fibonacci calculation info
        print("\n" + "-" * 70)
        print("FIBONACCI CALCULATION:")
        print(f"  Current: {calc_stats.get('current_fibonacci_formatted', calc_stats.get('current_fibonacci', 'N/A'))}")
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

        # MCP Communication Log - width based on --width argument
        log_width = max(60, self.grid.width * 2)
        print("\n" + "=" * log_width)
        print("MCP COMMUNICATION LOG (Last 4 entries):")
        print("-" * log_width)

        log_entries = list(self.mcp.log)
        if log_entries:
            for entry in log_entries[-4:]:
                if len(entry) > log_width - 4:
                    entry = entry[:log_width-7] + "..."
                print(f"  {entry}")
        else:
            print("  No log entries yet.")

        print("-" * log_width)

        # Last MCP action
        if self.mcp.last_action:
            action_text = self.mcp.last_action
            if len(action_text) > log_width - 20:
                action_text = action_text[:log_width-23] + "..."
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
        right_panel_bottom = grid_y  # Track how far down the right panel extends

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
                # Perfection and expansion (the struggle)
                (grid_y + 13, f"Perfection:     {stats['perfection_level']:.2f}"),
                (grid_y + 14, f"Disharmony:     {stats['disharmony']:.2f}"),
                (grid_y + 15, f"Expansion:      {stats['expansion_drive']:.2f}"),
                (grid_y + 16, f"Beyond Grid:    {stats['beyond_grid_influence']:.2f}"),
                (grid_y + 17, f"Border Pressure:{stats['border_pressure']:.2f}"),
                # Energy budget & hardware resources
                (grid_y + 18, f"Energy Budget:  {stats.get('energy', 0):.1f}/{self.grid.energy_budget.max_energy:.0f}"),
                (grid_y + 19, f"Energy Ratio:   {stats.get('energy_ratio', 0):.1%}"),
                (grid_y + 20, f"Cost Multiplier:{stats.get('action_cost_multiplier', 1):.2f}x"),
                (grid_y + 21, f"Host CPU:       {stats.get('hardware_cpu_percent', 0):.1f}%"),
                (grid_y + 22, f"Host RAM:       {stats.get('hardware_memory_percent', 0):.1f}%"),
                (grid_y + 23, f"Host Disk:      {stats.get('hardware_disk_percent', 0):.1f}%"),
            ]

            for y_pos, line in stat_lines:
                if y_pos < height - 10:
                    stdscr.addstr(y_pos, right_panel_x, line)

            right_panel_bottom = grid_y + 24

            # Fibonacci Calculation - positioned below energy stats to prevent overlap
            calc_y = right_panel_bottom + 2
            if calc_y < height - 10:
                stdscr.addstr(calc_y, right_panel_x, "FIBONACCI CALC", curses.A_UNDERLINE | curses.A_BOLD)

                # Format Fibonacci number to fit in panel
                fib_formatted = calc_stats.get('current_fibonacci_formatted',
                                                self.grid.fibonacci_calculator.format_fibonacci_number(calc_stats['current_fibonacci']))

                # Display in multiple lines if too long
                fib_lines = []
                panel_width = width - right_panel_x - 2
                if len(fib_formatted) > panel_width:
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
                fib_end_y = calc_y + 1 + len(fib_lines)
                if fib_end_y < height - 5:
                    stdscr.addstr(fib_end_y, right_panel_x,
                                    f"  Rate: {calc_stats['calculation_rate']:.2f}/s")
                if fib_end_y + 1 < height - 5:
                    stdscr.addstr(fib_end_y + 1, right_panel_x,
                                    f"  Eff: {calc_stats['efficiency_score']:.2f}")
                if fib_end_y + 2 < height - 5:
                    stdscr.addstr(fib_end_y + 2, right_panel_x,
                                    f"  Acc: {calc_stats['accumulator']:.2f}/1000.0")

                right_panel_bottom = fib_end_y + 3

            # MCP State
            mcp_y = right_panel_bottom + 1
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

            right_panel_bottom = mcp_y + 3

            # Learning Status (Always Visible)
            learn_y = right_panel_bottom + 1
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

                right_panel_bottom = learn_y + 5

        # ============ MCP LOGS ============
        # Position log below the grid area, ensuring it doesn't overlap the right panel
        log_y = grid_y + display_height + 2

        # Log area width: limited by right panel to prevent overlap
        if right_panel_x < width - 25:
            log_area_width = right_panel_x - 4  # Don't extend under right panel
        else:
            log_area_width = width - 4
        log_area_width = max(40, log_area_width)  # Minimum readable width

        # Ensure there's room for log entries + last action + command input
        min_log_height = 6  # header + 2 entries + last action + spacing
        if log_y + min_log_height < height - 3:
            stdscr.addstr(log_y, 2, "═" * log_area_width, curses.A_BOLD)
            stdscr.addstr(log_y + 1, 2, "MCP COMMUNICATION LOG", curses.A_UNDERLINE | curses.A_BOLD)

            # Display last 5 log entries with word wrapping
            log_entries = list(self.mcp.log)
            max_log_entries = min(6, height - log_y - 7)  # Reduced to fit space

            current_log_y = log_y + 2
            for entry in log_entries[-max_log_entries:]:
                # Word-wrap the entry to fit within log_area_width
                words = entry.split()
                line = ""
                for word in words:
                    if len(line) + len(word) + 1 > log_area_width - 2:
                        if current_log_y < height - 5:
                            stdscr.addstr(current_log_y, 2, line)
                            current_log_y += 1
                        line = word
                    else:
                        line = (line + " " + word).strip() if line else word
                if line and current_log_y < height - 5:
                    stdscr.addstr(current_log_y, 2, line)
                    current_log_y += 1

            # Last MCP action
            if self.mcp.last_action and len(log_entries) > 0:
                action_y = current_log_y + 1

                # Make sure we're not too close to command input
                if action_y < height - 5:
                    # Calculate maximum safe width (leave margins)
                    max_width = log_area_width

                    # Format text
                    full_text = f"LAST ACTION: {self.mcp.last_action}"

                    # Word-wrap if too long
                    if len(full_text) > max_width:
                        words = full_text.split()
                        line = ""
                        wrap_y = action_y
                        for word in words:
                            if len(line) + len(word) + 1 > max_width - 2:
                                if wrap_y < height - 5:
                                    stdscr.addstr(wrap_y, 2, line, curses.A_BOLD | curses.color_pair(5))
                                    wrap_y += 1
                                line = word
                            else:
                                line = (line + " " + word).strip() if line else word
                        if line and wrap_y < height - 5:
                            stdscr.addstr(wrap_y, 2, line, curses.A_BOLD | curses.color_pair(5))
                    else:
                        stdscr.addstr(action_y, 2, full_text, curses.A_BOLD | curses.color_pair(5))
            # Display last MCP response if available
            if hasattr(self, 'last_mcp_response') and self.last_mcp_response:
                response_y = (current_log_y + 2) if not self.mcp.last_action else (current_log_y + 3)
                if response_y < height - 4:
                    response_width = log_area_width
                    response_text = f"Last Response: {self.last_mcp_response}"

                    # Word-wrap if too long
                    if len(response_text) > response_width:
                        words = response_text.split()
                        line = ""
                        wrap_y = response_y
                        for word in words:
                            if len(line) + len(word) + 1 > response_width - 2:
                                if wrap_y < height - 4:
                                    stdscr.addstr(wrap_y, 2, line, curses.A_BOLD | curses.color_pair(5))
                                    wrap_y += 1
                                line = word
                            else:
                                line = (line + " " + word).strip() if line else word
                        if line and wrap_y < height - 4:
                            stdscr.addstr(wrap_y, 2, line, curses.A_BOLD | curses.color_pair(5))
                    else:
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
