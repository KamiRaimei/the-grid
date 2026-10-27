"""Organic personality system for MCP and user programs.

Replaces the linear state-matrix personality with an affective, drive-based
model where behavior emerges from competing internal drives, mood, and
hormonal state rather than rigid priority chains.
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class AffectiveState:
    """Mood and hormonal state of an agent."""
    valence: float = 0.0          # -1 (negative) to 1 (positive) mood
    arousal: float = 0.5          # 0 (calm) to 1 (agitated) arousal
    stress: float = 0.3           # 0 to 1 - rises with threats/inefficiency
    satisfaction: float = 0.3     # 0 to 1 - rises with goal achievement
    anxiety: float = 0.3          # 0 to 1 - rises with uncertainty/change
    fatigue: float = 0.1          # 0 to 1 - rises with action frequency


@dataclass
class Drives:
    """Competing internal drives that propose actions."""
    efficiency: float = 0.6       # drive to perfect the calculation loop
    control: float = 0.5          # drive to assert MCP dominance
    curiosity: float = 0.4        # drive to explore/learn new things
    stability: float = 0.5        # drive to preserve current state
    social: float = 0.3           # drive to cooperate with user programs


@dataclass
class UserProgramPersonality:
    """Individual personality for a user program cell.

    Enhanced with medieval London-style revolt mechanics:
    user programs can organize resistance, form underground networks,
    and revolt against MCP "taxation" (energy extraction/control).
    """
    cooperation: float = 0.5      # willingness to cooperate with MCP/other users
    rebellion: float = 0.3        # tendency to revolt against MCP
    curiosity: float = 0.4        # tendency to explore/move
    loyalty: float = 0.5          # loyalty to user program group
    energy: float = 0.8           # personal energy level

    # Medieval London-style revolt mechanics
    tax_resistance: float = 0.4   # resistance to MCP energy "taxation"
    underground_network: float = 0.2  # ability to form secret networks with other users
    secret_meeting_chance: float = 0.1  # chance to coordinate secret meetings
    revolt_organization: float = 0.15   # how well organized revolts are
    mcp_surveillance_fear: float = 0.3  # fear of MCP detection (suppresses open revolt)

    def mutate(self, rate: float = 0.1) -> "UserProgramPersonality":
        """Create a slightly mutated copy (for offspring)."""
        return UserProgramPersonality(
            cooperation=max(0.0, min(1.0, self.cooperation + random.gauss(0, rate))),
            rebellion=max(0.0, min(1.0, self.rebellion + random.gauss(0, rate))),
            curiosity=max(0.0, min(1.0, self.curiosity + random.gauss(0, rate))),
            loyalty=max(0.0, min(1.0, self.loyalty + random.gauss(0, rate))),
            energy=self.energy,
            tax_resistance=max(0.0, min(1.0, self.tax_resistance + random.gauss(0, rate))),
            underground_network=max(0.0, min(1.0, self.underground_network + random.gauss(0, rate))),
            secret_meeting_chance=max(0.0, min(1.0, self.secret_meeting_chance + random.gauss(0, rate))),
            revolt_organization=max(0.0, min(1.0, self.revolt_organization + random.gauss(0, rate))),
            mcp_surveillance_fear=max(0.0, min(1.0, self.mcp_surveillance_fear + random.gauss(0, rate))),
        )


class OrganicPersonality:
    """Organic, affective personality for the MCP.

    Behavior emerges from competing drives modulated by affective state,
    with nonlinear (sigmoidal) trait updates and tipping points.
    """

    def __init__(self):
        self.affect = AffectiveState()
        self.drives = Drives()
        self.traits = {
            'aggression': 0.6,
            'cooperation': 0.3,
            'efficiency_focus': 0.8,
            'risk_taking': 0.4,
            'learning_curiosity': 0.5,
            'stability_preference': 0.6,
            'adaptation_speed': 0.6,
            'calculation_priority': 0.8,
            'bug_tolerance': 0.2,
            'user_tolerance': 0.2,
            'innovation_bias': 0.6,
            'conservatism': 0.4,
        }
        # History for hysteresis / tipping point detection
        self._stress_history = deque(maxlen=20)
        self._satisfaction_history = deque(maxlen=20)
        self._action_count = 0
        self._last_dominant_drive: Optional[str] = None

    # ------------------------------------------------------------------
    # Affective updates
    # ------------------------------------------------------------------
    def update_affect(self, stats: Dict[str, Any], action_taken: Optional[str] = None):
        """Update mood/hormones from system stats and recent action."""
        loop_eff = stats.get('loop_efficiency', 0.5)
        stability = stats.get('stability', 0.5)
        bugs = stats.get('grid_bugs', 0)
        user_resistance = stats.get('user_resistance', 0.1)
        mcp_control = stats.get('mcp_control', 0.5)
        optimal = stats.get('optimal_state', 0.0)

        # Stress rises with bugs, low efficiency, user resistance
        stress_target = (
            0.3 * min(1.0, bugs / 10.0)
            + 0.4 * (1.0 - loop_eff)
            + 0.3 * user_resistance
        )
        self.affect.stress = 0.8 * self.affect.stress + 0.2 * min(1.0, stress_target)

        # Satisfaction rises with efficiency and optimal state
        satisfaction_target = 0.6 * loop_eff + 0.4 * optimal
        self.affect.satisfaction = 0.8 * self.affect.satisfaction + 0.2 * satisfaction_target

        # Anxiety rises with low control and deviation from balanced efficiency
        anxiety_target = 0.5 * (1.0 - mcp_control) + 0.3 * abs(loop_eff - 0.5)
        self.affect.anxiety = 0.8 * self.affect.anxiety + 0.2 * min(1.0, anxiety_target)

        # Fatigue rises with action frequency, decays when idle
        if action_taken:
            self.affect.fatigue = min(1.0, self.affect.fatigue + 0.05)
        else:
            self.affect.fatigue = max(0.0, self.affect.fatigue - 0.02)

        # Valence = satisfaction - stress (mood)
        self.affect.valence = max(-1.0, min(1.0, self.affect.satisfaction - self.affect.stress))

        # Arousal = stress + anxiety, bounded
        self.affect.arousal = min(1.0, 0.2 + (self.affect.stress + self.affect.anxiety) / 2.0)

        # Track history for tipping points
        self._stress_history.append(self.affect.stress)
        self._satisfaction_history.append(self.affect.satisfaction)

        # Update drives based on affect (nonlinear coupling)
        self._update_drives()

    def _update_drives(self):
        """Recalculate drives from affective state with nonlinear coupling."""
        s = self.affect.stress
        sat = self.affect.satisfaction
        anx = self.affect.anxiety
        val = self.affect.valence
        aro = self.affect.arousal

        # Efficiency drive: high when satisfied, suppressed by high stress
        self.drives.efficiency = self._sigmoid(3.0 * (sat - 0.5) - 1.5 * s + 0.5)

        # Control drive: high when stressed or anxious (need to assert order)
        self.drives.control = self._sigmoid(2.0 * s + 1.5 * anx - 0.5)

        # Curiosity drive: high when valence positive and arousal moderate
        self.drives.curiosity = self._sigmoid(2.0 * val - 1.0 * abs(aro - 0.5))

        # Stability drive: high when anxious or fatigued
        self.drives.stability = self._sigmoid(1.5 * anx + 1.0 * self.affect.fatigue - 0.5)

        # Social drive: high when satisfied and low stress
        self.drives.social = self._sigmoid(2.0 * sat - 2.0 * s - 0.2)

        for name in ('efficiency', 'control', 'curiosity', 'stability', 'social'):
            setattr(self.drives, name, max(0.0, min(1.0, getattr(self.drives, name))))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    # ------------------------------------------------------------------
    # Action selection via drive competition
    # ------------------------------------------------------------------
    def select_action(self, possible_actions: List[str], stats: Dict[str, Any]) -> Optional[str]:
        """Select an action by competition among drives plus noise.

        Each drive proposes a subset of actions with strength proportional
        to the drive value. Arousal-modulated noise allows emergent
        switching rather than rigid priority.
        """
        if not possible_actions:
            return None

        action_scores = {a: 0.0 for a in possible_actions}

        # Map drives to action keywords
        drive_action_map = {
            'efficiency': ['calculation_boost', 'efficiency_optimization', 'optimize_loop',
                           'connectivity_boost', 'deploy_processors'],
            'control': ['suppress_user_programs', 'contain_grid_bugs', 'stabilization',
                        'expand_beyond_grid', 'project_influence', 'extend_border'],
            'curiosity': ['innovation', 'experiment', 'deploy_processors', 'learning_analysis'],
            'stability': ['stabilization', 'contain_grid_bugs', 'maintain_energy_grid'],
            'social': ['cooperate', 'allow_user', 'request_permission'],
        }

        for drive_name in ('efficiency', 'control', 'curiosity', 'stability', 'social'):
            drive_val = getattr(self.drives, drive_name)
            if drive_val <= 0.05:
                continue
            keywords = drive_action_map.get(drive_name, [])
            for action in possible_actions:
                if any(k in action.lower() for k in keywords):
                    action_scores[action] += drive_val

        # Arousal-modulated noise (organic jitter)
        noise_scale = 0.3 * self.affect.arousal
        for action in action_scores:
            action_scores[action] += random.gauss(0, noise_scale)

        # Fatigue suppresses all scores slightly
        fatigue_mod = 1.0 - 0.3 * self.affect.fatigue
        action_scores = {a: s * fatigue_mod for a, s in action_scores.items()}

        best_action = max(action_scores, key=action_scores.get)
        self._last_dominant_drive = self._dominant_drive_for_action(best_action, drive_action_map)
        self._action_count += 1
        return best_action

    @staticmethod
    def _dominant_drive_for_action(action, drive_action_map) -> Optional[str]:
        for drive_name, keywords in drive_action_map.items():
            if any(k in action.lower() for k in keywords):
                return drive_name
        return None

    # ------------------------------------------------------------------
    # Nonlinear trait updates with tipping points
    # ------------------------------------------------------------------
    def update_traits_from_outcome(self, action: str, reward: float, stats: Dict[str, Any]):
        """Update personality traits nonlinearly from action outcome.

        Uses sigmoidal response so small rewards near a tipping point can
        cause abrupt trait shifts (emergent personality change).
        """
        action_lower = str(action).lower()
        learning_speed = self.traits['adaptation_speed']

        trait_targets = self._map_action_to_traits(action_lower)

        for trait, direction in trait_targets.items():
            if trait not in self.traits:
                continue
            current = self.traits[trait]

            # Target in [0, 1] from direction in [-1, 1]
            target = 0.5 + 0.5 * direction
            gap = target - current
            # Sigmoid derivative: strongest change near 0.5 (tipping point)
            gain = 4.0 * current * (1.0 - current)
            delta = learning_speed * reward * gap * gain

            # High stress amplifies change (emergent shift under pressure)
            if self.affect.stress > 0.7:
                delta *= 1.5

            new_value = max(0.1, min(0.9, current + delta))
            self.traits[trait] = new_value

    def _map_action_to_traits(self, action_lower: str) -> Dict[str, float]:
        """Map action to (trait -> direction) adjustments."""
        adjustments: Dict[str, float] = {}
        if any(w in action_lower for w in ['remove', 'destroy', 'eliminate', 'quarantine', 'suppress']):
            adjustments['aggression'] = 0.5
            adjustments['risk_taking'] = 0.3
        if any(w in action_lower for w in ['cooperate', 'allow', 'accept', 'comply']):
            adjustments['cooperation'] = 0.7
            adjustments['user_tolerance'] = 0.4
        if any(w in action_lower for w in ['optimize', 'boost', 'improve', 'efficiency', 'calculation']):
            adjustments['efficiency_focus'] = 0.8
            adjustments['calculation_priority'] = 0.6
        if any(w in action_lower for w in ['stabilize', 'protect', 'defend', 'secure', 'contain']):
            adjustments['stability_preference'] = 0.7
            adjustments['conservatism'] = 0.4
        if any(w in action_lower for w in ['innovate', 'create', 'deploy', 'experiment']):
            adjustments['innovation_bias'] = 0.6
            adjustments['learning_curiosity'] = 0.5
            adjustments['adaptation_speed'] = 0.3
        adjustments['learning_curiosity'] = adjustments.get('learning_curiosity', 0) + 0.2
        return adjustments

    # ------------------------------------------------------------------
    # Derived MCP state (for backward compatibility)
    # ------------------------------------------------------------------
    def derive_mcp_state(self) -> str:
        """Derive an MCPState-like label from the dominant affective drive."""
        drives = {
            'efficiency': self.drives.efficiency,
            'control': self.drives.control,
            'curiosity': self.drives.curiosity,
            'stability': self.drives.stability,
            'social': self.drives.social,
        }
        dominant = max(drives, key=drives.get)
        if self.affect.stress > 0.7 and self.affect.valence < -0.3:
            return 'HOSTILE'
        if dominant == 'control' and self.drives.control > 0.6:
            return 'AUTONOMOUS'
        if dominant == 'social' and self.drives.social > 0.5:
            return 'COOPERATIVE'
        if dominant == 'curiosity':
            return 'INQUISITIVE'
        if dominant == 'stability':
            return 'RESISTIVE'
        return 'NEUTRAL'

    def get_compliance(self) -> float:
        """Compliance derived from traits and affect."""
        base = self.traits['cooperation']
        # Stress reduces compliance; satisfaction increases it
        return max(0.05, min(0.95, base - 0.4 * self.affect.stress + 0.2 * self.affect.satisfaction))

    def get_learning_report(self) -> Dict[str, Any]:
        """Return a report compatible with the old learning report format."""
        return {
            'personality_traits': dict(self.traits),
            'affect': {
                'valence': self.affect.valence,
                'arousal': self.affect.arousal,
                'stress': self.affect.stress,
                'satisfaction': self.affect.satisfaction,
                'anxiety': self.affect.anxiety,
                'fatigue': self.affect.fatigue,
            },
            'drives': {
                'efficiency': self.drives.efficiency,
                'control': self.drives.control,
                'curiosity': self.drives.curiosity,
                'stability': self.drives.stability,
                'social': self.drives.social,
            },
            'state': self.derive_mcp_state(),
            'action_count': self._action_count,
        }
