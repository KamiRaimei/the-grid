#!/usr/bin/env python3
"""
Grid Simulation - Cell-based Fibonacci calculation.
System directive: Maintain a perfect calculation loop with efficiency through adaptive learning.
"""

"""Cell types, grid constants, and base cell data structures."""

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

# ==================== MCP ACTION ENERGY COSTS ====================
# Energy cost for each MCP autonomous action (drawn from global energy budget)
MCP_ACTION_ENERGY_COSTS = {
    'optimize_calculation_loop': 5.0,
    'deploy_fibonacci_processors': 10.0,
    'quarantine_grid_bugs': 3.0,
    'contain_bug_outbreak': 5.0,
    'stabilize_system': 4.0,
    'boost_calculation_rate': 6.0,
    'add_calculation_infrastructure': 8.0,
    'optimize_fibonacci_calculation': 5.0,
    'improve_cell_connectivity': 4.0,
    'add_data_streams': 3.0,
    'enhance_collaboration': 3.0,
    'inefficiency_cleanup': 4.0,
    'cell_repurposing': 5.0,
    'optimize_cell_efficiency': 6.0,
    'maintain_energy_grid': 4.0,
    'optimize_resource_distribution': 5.0,
    'balance_system_load': 4.0,
    'suppress_user_programs': 8.0,
    # Expansion actions (expensive - project beyond grid)
    'expand_beyond_grid': 15.0,
    'project_influence': 12.0,
    'extend_border': 10.0,
}

# Energy cost for MCP command-driven actions
MCP_COMMAND_ENERGY_COSTS = {
    'ADD_PROGRAM': 3.0,
    'DELETE_CELL': 2.0,
    'REPURPOSE_CELL': 4.0,
    'OPTIMIZE_CELLS': 6.0,
    'BOOST_ENERGY': 5.0,
    'REPAIR_SYSTEM': 4.0,
    'REMOVE_BUG': 3.0,
    'CREATE_SPECIAL': 7.0,
    'USE_SPECIAL': 3.0,
    'DEPLOY_PROCESSORS': 10.0,
    'CALCULATE_FIBONACCI': 2.0,
    # God-level commands (expensive overrides)
    'GOD_OVERRIDE': 20.0,
    'GOD_PURGE': 25.0,
    'GOD_FREEZE': 15.0,
    'GOD_RESET': 50.0,
    'GOD_REVEAL': 10.0,
}

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
    # Organic personality for user programs (None for non-user cells)
    personality: Optional[Dict[str, float]] = None

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

