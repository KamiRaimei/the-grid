"""Learnable natural language processor for MCP commands."""

import os
import re
import json
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from .cells import CellType, PERSONALITY_FILE


class LearnableNaturalLanguageProcessor:
    """Simple natural language processing for MCP commands"""

    def __init__(self):
        self.learned_patterns_file = "mcp_learned_patterns.json"
        self.base_intents = self._initialize_base_intents()
        self.learned_intents = self._load_learned_intents()
        self.patterns = self._initialize_patterns()
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
                "remove_bug": ["remove bug", "remove bugs", "delete bug", "eliminate bug", "kill bug"],
                "quarantine_bug": ["quarantine bug", "isolate bug", "contain bug"],

                # System operations
                "boost_energy": ["boost energy", "add energy", "increase power", "power up"],
                "repair_system": ["repair", "repair system", "fix system", "restore system", "heal system"],
                "scan_area": ["scan", "scan area", "analyze", "examine"],
                "optimize_loop": ["optimize loop", "improve efficiency", "enhance loop", "optimize calculation"],

                # Special programs
                "list_programs": ["list programs", "show programs", "view programs", "special programs"],
                "create_scanner": ["create scanner", "build scanner", "make scanner", "deploy scanner"],
                "create_defender": ["create defender", "build defender", "make defender"],
                "create_repair": ["create repair program", "build repair", "make repair"],
                "create_fibonacci_calculator": [
                    "create fibonacci calculator", "create fibonacci_calculator",
                    "build fibonacci", "deploy calculator"
                ],

                # MCP interaction
                "who_are_you": ["who are you", "what are you", "identify yourself", "your name"],
                "what_should_i_do": ["what should i do", "what do you suggest", "recommend something", "advise me"],
                "why_did_you": ["why did you", "why did mcp", "explain that action", "why that action"],

                # Cell repurpose
                "delete_cell": ["delete cell", "remove cell", "erase cell", "clear cell"],
                "repurpose_cell": ["repurpose cell", "convert cell", "transform cell", "change cell type"],
                "optimize_cells": ["optimize cells", "improve cells", "enhance cells", "tune cells for calculation"],

                # God-level commands (User override powers)
                "god_override": ["god override", "override mcp", "user override", "admin override", "force command"],
                "god_purge": ["god purge", "purge all", "purge user programs", "purge mcp programs", "clean slate"],
                "god_freeze": ["god freeze", "freeze mcp", "halt mcp", "stop mcp actions", "suspend mcp"],
                "god_reset": ["god reset", "reset system", "reset grid", "factory reset", "wipe grid"],
                "god_reveal": ["god reveal", "reveal loopholes", "show loopholes", "explain bypass", "hidden actions"],

                # Learning specific
                "learning_status": [
                    "learning status", "learning_status", "mcp learning",
                    "personality status", "learning report"
                ],
                "cell_cooperation": [
                    "cell cooperation", "cell_cooperation", "cooperation level",
                    "cell collaboration"
                ],
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

        if intent == "deploy_processors":
            params['special_type'] = 'FIBONACCI_CALCULATOR'

        if intent == "USE_SPECIAL":
            if 'scanner' in command_lower:
                params['special_type'] = 'SCANNER'
            elif 'defender' in command_lower:
                params['special_type'] = 'DEFENDER'
            elif 'repair' in command_lower:
                params['special_type'] = 'REPAIR'
            elif 'saboteur' in command_lower:
                params['special_type'] = 'SABOTEUR'
            elif 'reconfigurator' in command_lower:
                params['special_type'] = 'RECONFIGURATOR'
            elif 'harvester' in command_lower:
                params['special_type'] = 'ENERGY_HARVESTER'
            elif 'fibonacci' in command_lower or 'calculator' in command_lower:
                params['special_type'] = 'FIBONACCI_CALCULATOR'
            function_match = re.search(r"\b(scan_area|report_status|quarantine_bug|protect_cell|"
                                      r"repair_cell|boost_energy|disrupt_mcp|create_entropy|"
                                      r"reconfigure_cells|optimize_grid|harvest_energy|"
                                      r"distribute_energy|calculate_next|optimize_calculation|"
                                      r"deploy_processors|learning_analysis)\b", command_lower)
            if function_match:
                params['function_name'] = function_match.group(1)
            program_match = re.search(r"\bSP_\d+\b", command, re.IGNORECASE)
            if program_match:
                params['program_id'] = program_match.group(0).upper()

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
            r"\buse\s+(scanner|defender|repair|saboteur|reconfigurator|harvester|fibonacci_calculator)\b": "USE_SPECIAL",

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

        for pattern, intent in self.patterns.items():
            if pattern.search(normalized_command):
                return intent, pattern.pattern

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
        """Match a complete phrase rather than arbitrary substrings."""
        normalized_pattern = self._normalize_input(pattern)
        return bool(re.search(
            rf"\b{re.escape(normalized_pattern)}\b",
            command.lower()
        ))

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

