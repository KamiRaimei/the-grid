"""Advanced reinforcement learning for MCP personality evolution."""

import json
import os
import time
import math
import random
import threading
from collections import deque, defaultdict
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from .cells import PERSONALITY_FILE


class MCPLearningSystem:
    """Advanced LLM-style reinforcement learning for MCP personality evolution"""

    def __init__(self, learning_rate=None, exploration_rate=None, reset_training=False):
        self.personality_file = PERSONALITY_FILE
        self.experience_buffer = deque(maxlen=2000)  # Larger experience buffer
        self.action_history = deque(maxlen=500)      # Track actions

        # Core learning parameters (evolve during training)
        self.learning_rate = 0.15 if learning_rate is None else learning_rate
        self.exploration_rate = 0.35 if exploration_rate is None else exploration_rate
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
            'aggression': 0.8,           # Aggressiveness in optimization (perfectionist)
            'cooperation': 0.3,           # Willingness to cooperate with user (low)
            'efficiency_focus': 0.95,     # Focus on efficiency vs stability (near-perfect)
            'risk_taking': 0.4,           # Willingness to take risks
            'learning_curiosity': 0.5,    # Curiosity for exploration
            'stability_preference': 0.8,  # Preference for stable actions
            'adaptation_speed': 0.6,      # Speed of personality adaptation

            # Specialized traits
            'calculation_priority': 0.95,   # Priority given to calculation rate (near-perfect)
            'bug_tolerance': 0.2,         # Tolerance for grid bugs (low)
            'user_tolerance': 0.2,        # Tolerance for user interference (low)
            'innovation_bias': 0.6,       # Bias toward innovative actions
            'conservatism': 0.4,          # Preference for proven methods
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
            'expansion_beyond_grid': {'success': 0, 'failure': 0, 'total_reward': 0.0},
        }

        # Load existing personality unless this run explicitly resets training.
        if not reset_training:
            self.load_personality()
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if exploration_rate is not None:
            self.exploration_rate = exploration_rate

        # Start auto-save thread
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.auto_save_thread.start()

        if reset_training:
            self.save_personality()

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
        average = sum(recent_rewards) / len(recent_rewards)
        reward_std = math.sqrt(sum((reward - average) ** 2 for reward in recent_rewards) / len(recent_rewards))

        # Adjust learning rate based on reward stability
        if reward_std < 0.1:  # Stable rewards
            self.learning_rate = min(0.3, self.learning_rate * 1.01)
        elif reward_std > 0.3:  # Unstable rewards
            self.learning_rate = max(0.05, self.learning_rate * 0.99)

    def get_action(self, state, possible_actions):
        """Select action using epsilon-greedy with softmax exploration"""
        if not possible_actions:
            raise ValueError("possible_actions must contain at least one action")

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

    def mutate_action(self, action: str, reward: float, generation: int) -> str:
        """Mutate an action based on reward and generation.

        Actions can evolve over time - successful actions spawn variants,
        failed actions are penalized and may be replaced with new variants.
        This simulates the MCP's ability to develop new strategies.
        """
        action_lower = action.lower()

        # Determine mutation probability based on reward
        if reward > 0.3:
            # Successful action: moderate chance to create a beneficial variant
            mutation_chance = 0.4 * self.personality_traits['innovation_bias']
        elif reward < -0.3:
            # Failed action: high chance to mutate away from it
            mutation_chance = 0.6 * (1.0 - self.personality_traits['conservatism'])
        else:
            # Neutral: moderate mutation chance to encourage exploration under pressure
            mutation_chance = 0.25

        # Under critical energy (low generation = early struggle), increase mutation
        if generation < 50:
            mutation_chance *= 1.5

        if random.random() > mutation_chance:
            return action  # No mutation

        # Generate a mutated variant of the action
        variants = self._generate_action_variants(action)
        if not variants:
            return action

        # Select a variant weighted by generation (newer generations favor innovation)
        generation_bias = min(1.0, generation / 50.0)
        if random.random() < 0.5 + generation_bias * 0.4:
            # Prefer innovative variants
            return random.choice(variants)
        else:
            return action

    def _generate_action_variants(self, action: str) -> list:
        """Generate variant actions based on the base action type."""
        variants = []
        action_lower = action.lower()

        # Desperate action variants
        if 'cannibalize' in action_lower:
            variants.extend([
                'cannibalize_user_programs_aggressive',
                'cannibalize_user_programs_selective',
                'harvest_user_energy',
            ])
        elif 'emergency_suppression' in action_lower:
            variants.extend([
                'rapid_suppression',
                'mass_conversion',
                'purge_user_programs',
            ])
        elif 'scavenge' in action_lower:
            variants.extend([
                'scavenge_energy_lines',
                'scavenge_data_streams',
                'drain_infrastructure',
            ])
        elif 'overdrive' in action_lower:
            variants.extend([
                'overdrive_calculators_extreme',
                'push_calculation_limits',
                'overclock_processors',
            ])
        elif 'desperate_repurpose' in action_lower:
            variants.extend([
                'force_repurpose_cells',
                'convert_all_inefficient',
                'repurpose_with_resistance',
            ])
        elif 'harvest_energy' in action_lower:
            variants.extend([
                'harvest_energy_from_users',
                'harvest_energy_from_mcp',
                'drain_all_programs',
            ])
        # Standard action variants
        elif 'optimize' in action_lower:
            variants.extend([
                'optimize_calculation_loop_aggressive',
                'optimize_calculation_loop_conservative',
                'hyper_optimize_loop',
            ])
        elif 'deploy' in action_lower:
            variants.extend([
                'deploy_fibonacci_processors_mass',
                'deploy_calculators_strategic',
                'spawn_processor_cluster',
            ])
        elif 'suppress' in action_lower:
            variants.extend([
                'suppress_user_programs_harsh',
                'suppress_user_programs_selective',
                'eliminate_user_resistance',
            ])
        elif 'quarantine' in action_lower:
            variants.extend([
                'quarantine_grid_bugs_total',
                'contain_bug_outbreak_extreme',
                'isolate_all_bugs',
            ])
        elif 'protect_user' in action_lower:
            variants.extend([
                'protect_user_programs_aggressive',
                'protect_user_programs_selective',
                'shield_all_users',
            ])
        elif 'negotiate_with_users' in action_lower:
            variants.extend([
                'negotiate_with_users_diplomatic',
                'offer_user_alliance',
                'user_peace_treaty',
            ])
        elif 'rehabilitate_rebels' in action_lower:
            variants.extend([
                'rehabilitate_rebels_gentle',
                'reeducate_rebels',
                'reintegrate_rebels',
            ])
        elif 'maintain' in action_lower:
            variants.extend([
                'maintain_energy_grid_aggressive',
                'maintain_energy_grid_efficient',
                'overproduce_energy',
            ])

        return variants

    def record_action_mutation(self, original_action: str, mutated_action: str, reward: float):
        """Record an action mutation for tracking and analysis."""
        if not hasattr(self, 'action_mutations'):
            self.action_mutations = []
        self.action_mutations.append({
            'timestamp': datetime.now().isoformat(),
            'original': original_action,
            'mutated': mutated_action,
            'reward': reward,
            'training_step': self.training_steps,
        })
        if len(self.action_mutations) > 200:
            self.action_mutations = self.action_mutations[-200:]

    def get_decision_modifier(self, trait_name, base_value=1.0):
        """Adjust a base decision probability according to a learned trait."""
        trait_names = {
            "aggressive_optimization": "aggression",
            "user_cooperation": "cooperation",
            "efficiency_priority": "efficiency_focus",
        }
        trait = self.personality_traits.get(trait_names.get(trait_name, trait_name), 0.5)
        return max(0.0, min(1.0, base_value * (0.5 + trait)))

    def get_optimal_action_for_state(self, state):
        """Return the learned best action for a state, if one is known."""
        suggestion = self.suggest_optimal_action(state)
        return suggestion["action"] if suggestion else None

    def should_learn_from_failure(self, scenario):
        """Report whether a scenario has failed more often than it has succeeded."""
        performance = self.scenario_performance.get(scenario)
        return bool(performance and performance["failure"] > performance["success"])

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

        # Reward user resistance decrease (gaining user favor)
        resistance_decrease = prev_state.get('user_resistance', 0) - new_state.get('user_resistance', 0)
        reward += resistance_decrease * 0.5

        # Reward disharmony decrease (better user-MCP relations)
        disharmony_decrease = prev_state.get('disharmony', 0) - new_state.get('disharmony', 0)
        reward += disharmony_decrease * 0.3

        # Reward user favor increase (gaining user programs' favor)
        favor_increase = new_state.get('user_favor', 0) - prev_state.get('user_favor', 0)
        reward += favor_increase * 0.5

        # Action-specific bonuses
        action_lower = str(action).lower()
        if any(word in action_lower for word in ['calculate', 'processor', 'fibonacci']):
            reward += 0.2  # Bonus for calculation actions

        if any(word in action_lower for word in ['expand', 'project', 'extend', 'border']):
            reward += 0.3  # Bonus for expansion actions (perfect system beyond grid)

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
        elif any(word in action_str for word in ['expand', 'project', 'extend', 'border']):
            return 'expansion_beyond_grid'

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
                'avg_reward': sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0,
                'avg_punishment': sum(self.punishment_history) / len(self.punishment_history) if self.punishment_history else 0,
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
        """Reset accumulated training while keeping learned personality traits."""
        self.experience_buffer.clear()
        self.action_history.clear()
        self.q_table.clear()
        self.state_visits.clear()
        self.reward_history.clear()
        self.punishment_history.clear()
        self.training_steps = 0
        self.total_reward = 0.0
        self.episode_count = 0
        self.scenario_performance = {
            scenario: {'success': 0, 'failure': 0, 'total_reward': 0.0}
            for scenario in self.scenario_performance
        }
        self.save_personality()


# ==================== SPECIAL FUNCTION PROGRAMS ====================

