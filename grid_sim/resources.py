"""Global energy budget and hardware resource counter for the Grid simulation.

Provides:
- GlobalEnergyBudget: a finite energy pool that MCP must manage and exploit.
- HardwareResourceCounter: reads live host hardware (CPU, memory, disk) via psutil.
- Resource scarcity mechanics that force MCP to prioritize actions.
"""

import os
import time
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when psutil missing
    PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hardware resource counter
# ---------------------------------------------------------------------------

class HardwareResourceCounter:
    """Reads live host hardware resources and exposes them as simulation stats.

    Falls back to /proc parsing when psutil is unavailable, and to static
    defaults when neither is available.
    """

    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self._last_update = 0.0
        self._cpu_count = os.cpu_count() or 2
        self._cpu_percent = 0.0
        self._cpu_times = None
        self._memory_total = 0
        self._memory_available = 0
        self._memory_percent = 0.0
        self._disk_total = 0
        self._disk_free = 0
        self._disk_percent = 0.0
        self._load_avg = (0.0, 0.0, 0.0)
        self._boot_time = 0.0
        self._sample_count = 0
        self._lock = threading.Lock()
        self._init_proc_values()

    # -- initialization ----------------------------------------------------

    def _init_proc_values(self) -> None:
        """Read initial /proc values for fallback / delta computation."""
        self._cpu_times = self._read_cpu_times()
        self._memory_total, self._memory_available = self._read_memory_proc()
        self._disk_total, self._disk_free = self._read_disk_proc()
        try:
            self._boot_time = psutil.boot_time() if PSUTIL_AVAILABLE else time.time()
        except Exception:
            self._boot_time = time.time()

    # -- public API --------------------------------------------------------

    def update(self) -> None:
        """Refresh hardware readings (rate-limited to update_interval)."""
        now = time.time()
        if now - self._last_update < self.update_interval:
            return
        self._last_update = now

        with self._lock:
            self._sample_count += 1
            self._read_cpu()
            self._read_memory()
            self._read_disk()
            self._read_load()

    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of current hardware resources."""
        with self._lock:
            return {
                'cpu_count': self._cpu_count,
                'cpu_percent': round(self._cpu_percent, 2),
                'cpu_times': self._cpu_times,
                'memory_total_kb': self._memory_total,
                'memory_available_kb': self._memory_available,
                'memory_used_kb': max(0, self._memory_total - self._memory_available),
                'memory_percent': round(self._memory_percent, 2),
                'disk_total_kb': self._disk_total,
                'disk_free_kb': self._disk_free,
                'disk_used_kb': max(0, self._disk_total - self._disk_free),
                'disk_percent': round(self._disk_percent, 2),
                'load_avg_1m': round(self._load_avg[0], 2),
                'load_avg_5m': round(self._load_avg[1], 2),
                'load_avg_15m': round(self._load_avg[2], 2),
                'boot_time': self._boot_time,
                'sample_count': self._sample_count,
                'psutil_available': PSUTIL_AVAILABLE,
            }

    def get_resource_pressure(self) -> float:
        """Return a 0.0-1.0 combined resource pressure score (higher = scarcer)."""
        with self._lock:
            cpu_pressure = self._cpu_percent / 100.0
            mem_pressure = self._memory_percent / 100.0
            disk_pressure = self._disk_percent / 100.0
            load_pressure = min(1.0, self._load_avg[0] / max(1, self._cpu_count))
            return round((cpu_pressure + mem_pressure + disk_pressure + load_pressure) / 4.0, 4)

    # -- private readers ---------------------------------------------------

    def _read_cpu(self) -> None:
        if PSUTIL_AVAILABLE:
            try:
                self._cpu_percent = psutil.cpu_percent(interval=None)
                self._cpu_count = psutil.cpu_count(logical=True) or self._cpu_count
                return
            except Exception:
                pass
        # Fallback: /proc/stat delta
        new_times = self._read_cpu_times()
        if self._cpu_times is not None and new_times is not None:
            try:
                prev_idle = self._cpu_times[3] + self._cpu_times[4]
                prev_total = sum(self._cpu_times[:4]) + prev_idle
                new_idle = new_times[3] + new_times[4]
                new_total = sum(new_times[:4]) + new_idle
                if new_total > prev_total:
                    self._cpu_percent = max(0.0, min(100.0,
                        100.0 * (1.0 - (new_idle - prev_idle) / (new_total - prev_total))))
            except Exception:
                pass
        self._cpu_times = new_times

    def _read_memory(self) -> None:
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                self._memory_total = mem.total // 1024
                self._memory_available = mem.available // 1024
                self._memory_percent = mem.percent
                return
            except Exception:
                pass
        total, available = self._read_memory_proc()
        self._memory_total = total
        self._memory_available = available
        self._memory_percent = (1.0 - available / max(1, total)) * 100.0 if total else 0.0

    def _read_disk(self) -> None:
        if PSUTIL_AVAILABLE:
            try:
                disk = psutil.disk_usage('/')
                self._disk_total = disk.total // 1024
                self._disk_free = disk.free // 1024
                self._disk_percent = disk.percent
                return
            except Exception:
                pass
        total, free = self._read_disk_proc()
        self._disk_total = total
        self._disk_free = free
        self._disk_percent = (1.0 - free / max(1, total)) * 100.0 if total else 0.0

    def _read_load(self) -> None:
        if PSUTIL_AVAILABLE:
            try:
                self._load_avg = psutil.getloadavg()
                return
            except Exception:
                pass
        try:
            with open('/proc/loadavg', 'r') as f:
                parts = f.read().split()
                self._load_avg = (float(parts[0]), float(parts[1]), float(parts[2]))
        except Exception:
            self._load_avg = (0.0, 0.0, 0.0)

    # -- /proc fallbacks ---------------------------------------------------

    @staticmethod
    def _read_cpu_times() -> Optional[tuple]:
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
                parts = line.split()
                if parts[0] == 'cpu':
                    return tuple(float(x) for x in parts[1:9])
        except Exception:
            pass
        return None

    @staticmethod
    def _read_memory_proc() -> tuple:
        total = 0
        available = 0
        try:
            meminfo: Dict[str, int] = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        meminfo[key.strip()] = int(val.split()[0])
            total = meminfo.get('MemTotal', 0)
            available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
        except Exception:
            pass
        return total, available

    @staticmethod
    def _read_disk_proc() -> tuple:
        total = 0
        free = 0
        try:
            stat = os.statvfs('/')
            total = stat.f_blocks * stat.f_frsize // 1024
            free = stat.f_bavail * stat.f_frsize // 1024
        except Exception:
            pass
        return total, free


# ---------------------------------------------------------------------------
# Global energy budget
# ---------------------------------------------------------------------------

@dataclass
class EnergyBudgetConfig:
    """Configuration for the global energy budget."""
    initial_energy: float = 25000.0         # Starting energy pool
    max_energy: float = 100000.0            # Hard cap on energy storage (raised ceiling)
    production_per_energy_line: float = 5.0  # Energy produced per ENERGY_LINE per tick
    production_per_system_core: float = 25.0 # Energy produced per SYSTEM_CORE per tick
    consumption_per_user_program: float = 0.5  # Energy consumed per USER_PROGRAM per tick
    consumption_per_mcp_program: float = 0.4   # Energy consumed per MCP_PROGRAM per tick
    consumption_per_special_program: float = 0.25 # Energy consumed per SPECIAL_PROGRAM per tick
    consumption_per_fib_processor: float = 0.15   # Energy consumed per FIBONACCI_PROCESSOR per tick
    # Energy cost for cell replication (charged to the global budget when MCP approves)
    consumption_per_user_replication: float = 10.0   # Energy MCP must spend to approve a user program replication
    consumption_per_mcp_replication: float = 8.0     # Energy MCP must spend to approve an MCP program replication
    base_leak_rate: float = 0.005           # Base energy loss per tick (entropy)
    scarcity_threshold: float = 0.3         # Below this ratio, energy is "scarce"
    critical_threshold: float = 0.1         # Below this ratio, energy is "critical"


class GlobalEnergyBudget:
    """A finite, global energy pool that the MCP must manage and exploit.

    Energy is produced by infrastructure cells (ENERGY_LINE, SYSTEM_CORE)
    and consumed by program cells and MCP autonomous actions. When energy
    falls below scarcity thresholds, MCP actions become restricted or more
    expensive, forcing prioritization.
    """

    def __init__(self, config: Optional[EnergyBudgetConfig] = None):
        self.config = config or EnergyBudgetConfig()
        self.energy = self.config.initial_energy
        self.max_energy = self.config.max_energy
        self.total_produced = 0.0
        self.total_consumed = 0.0
        self.total_spent_on_actions = 0.0
        self.history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._tick = 0
        self._last_scarcity_event = 0.0
        self.scarcity_events: List[Dict[str, Any]] = []

    # -- core budget operations -------------------------------------------

    def produce(self, amount: float, source: str = "infrastructure") -> float:
        """Add energy to the budget (capped at max_energy). Returns actual added."""
        with self._lock:
            amount = max(0.0, amount)
            space = self.max_energy - self.energy
            added = min(amount, space)
            self.energy += added
            self.total_produced += added
            return added

    def consume(self, amount: float, reason: str = "unspecified") -> bool:
        """Attempt to consume energy. Returns True if sufficient energy available."""
        with self._lock:
            amount = max(0.0, amount)
            if self.energy >= amount:
                self.energy -= amount
                self.total_consumed += amount
                self._record_history('consume', amount, reason)
                return True
            # Partial consumption: take whatever remains
            if self.energy > 0:
                taken = self.energy
                self.total_consumed += taken
                self.energy = 0.0
                self._record_history('consume_partial', taken, reason)
                return False
            self._record_history('consume_failed', amount, reason)
            return False

    def spend_on_action(self, amount: float, action_name: str, force: bool = False) -> bool:
        """Spend energy on an MCP autonomous action. Returns True if affordable.

        If force=True, allows deficit spending (energy can go to zero/negative)
        for desperate last-resort actions.
        """
        with self._lock:
            amount = max(0.0, amount)
            if self.energy >= amount:
                self.energy -= amount
                self.total_consumed += amount
                self._record_history('consume', amount, reason=f"mcp_action:{action_name}")
                return True
            # Partial consumption: take whatever remains
            if self.energy > 0:
                taken = self.energy
                self.total_consumed += taken
                self.energy = 0.0
                self._record_history('consume_partial', taken, reason=f"mcp_action:{action_name}")
                if force:
                    return True
                return False
            if force:
                self._record_history('consume_forced', amount, reason=f"mcp_action:{action_name}")
                return True
            self._record_history('consume_failed', amount, reason=f"mcp_action:{action_name}")
            return False

    def can_afford(self, amount: float) -> bool:
        """Check if the budget can afford an amount without spending."""
        with self._lock:
            return self.energy >= amount

    def leak(self, rate: Optional[float] = None) -> float:
        """Apply base entropy leak to the budget. Returns amount lost."""
        with self._lock:
            rate = rate if rate is not None else self.config.base_leak_rate
            loss = self.energy * rate
            self.energy = max(0.0, self.energy - loss)
            self.total_consumed += loss
            return loss

    # -- scarcity / prioritization ----------------------------------------

    def energy_ratio(self) -> float:
        """Return energy / max_energy (0.0 = empty, 1.0 = full)."""
        with self._lock:
            return self.energy / self.max_energy if self.max_energy > 0 else 0.0

    def is_scarce(self) -> bool:
        """True when energy is below the scarcity threshold."""
        return self.energy_ratio() < self.config.scarcity_threshold

    def is_critical(self) -> bool:
        """True when energy is below the critical threshold."""
        return self.energy_ratio() < self.config.critical_threshold

    def action_cost_multiplier(self) -> float:
        """Return the action cost multiplier.

        Always returns 1.0 — no energy cost penalty. MCP actions cost the same
        regardless of energy level, so MCP is never artificially restricted by
        scarcity-based cost inflation.
        """
        return 1.0

    def affordable_actions(self, action_costs: Dict[str, float]) -> Dict[str, float]:
        """Filter action costs to only those affordable, applying scarcity multiplier."""
        multiplier = self.action_cost_multiplier()
        affordable = {}
        for action, base_cost in action_costs.items():
            actual_cost = base_cost * multiplier
            if self.can_afford(actual_cost):
                affordable[action] = actual_cost
        return affordable

    # -- tick-based production/consumption --------------------------------

    def tick_production(self, counts: Dict[int, int]) -> float:
        """Calculate and add energy production from infrastructure cells.

        `counts` maps CellType enum value -> count of cells of that type.
        """
        energy_line_count = counts.get(5, 0)   # CellType.ENERGY_LINE.value == 5
        system_core_count = counts.get(7, 0)   # CellType.SYSTEM_CORE.value == 7
        produced = (
            energy_line_count * self.config.production_per_energy_line +
            system_core_count * self.config.production_per_system_core
        )
        return self.produce(produced, source="infrastructure")

    def tick_consumption(self, counts: Dict[int, int]) -> float:
        """Calculate and consume energy for all program cells.

        `counts` maps CellType enum value -> count of cells of that type.
        Returns total consumed.
        """
        user_count = counts.get(1, 0)      # CellType.USER_PROGRAM.value == 1
        mcp_count = counts.get(2, 0)       # CellType.MCP_PROGRAM.value == 2
        special_count = counts.get(8, 0)   # CellType.SPECIAL_PROGRAM.value == 8
        fib_count = counts.get(9, 0)       # CellType.FIBONACCI_PROCESSOR.value == 9

        required = (
            user_count * self.config.consumption_per_user_program +
            mcp_count * self.config.consumption_per_mcp_program +
            special_count * self.config.consumption_per_special_program +
            fib_count * self.config.consumption_per_fib_processor
        )
        self.consume(required, reason="cell_upkeep")
        return required

    # -- history / reporting ----------------------------------------------

    def _record_history(self, event_type: str, amount: float, reason: str) -> None:
        self.history.append({
            'tick': self._tick,
            'time': time.time(),
            'event': event_type,
            'amount': amount,
            'energy_after': self.energy,
            'reason': reason,
        })
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def record_scarcity_event(self, level: str, message: str) -> None:
        """Record a scarcity event for logging/display."""
        self.scarcity_events.append({
            'tick': self._tick,
            'time': time.time(),
            'level': level,  # 'scarce' or 'critical'
            'energy_ratio': self.energy_ratio(),
            'message': message,
        })
        if len(self.scarcity_events) > 100:
            self.scarcity_events = self.scarcity_events[-100:]
        self._last_scarcity_event = time.time()

    def advance_tick(self) -> None:
        """Advance the internal tick counter."""
        self._tick += 1

    def adjust_max_energy_from_hardware(self, hardware: HardwareResourceCounter) -> None:
        """Dynamically adjust max_energy based on available host hardware.

        Available RAM and CPU headroom determine the energy ceiling:
        - More available RAM → higher max_energy (more memory to sustain the grid)
        - More available CPU → higher max_energy (more compute to generate energy)
        - High host pressure → lower max_energy (resources constrained)

        The base max_energy is scaled by a hardware factor (0.5x to 2.0x).
        """
        snapshot = hardware.snapshot()
        mem_percent = snapshot['memory_percent']
        cpu_percent = snapshot['cpu_percent']
        load_avg = snapshot['load_avg_1m']
        cpu_count = snapshot['cpu_count']

        # Available RAM fraction (inverse of usage)
        ram_available = max(0.0, 1.0 - mem_percent / 100.0)

        # Available CPU fraction (inverse of usage + load pressure)
        cpu_load_pressure = min(1.0, load_avg / max(1, cpu_count))
        cpu_available = max(0.0, 1.0 - cpu_percent / 100.0 - cpu_load_pressure * 0.5)

        # Combined hardware factor: 0.5 (very constrained) to 2.0 (abundant)
        hw_factor = 0.5 + ram_available * 0.75 + cpu_available * 0.75
        hw_factor = max(0.5, min(2.0, hw_factor))

        # New max_energy = base max * hardware factor
        new_max = self.config.max_energy * hw_factor

        # If energy exceeds new max, clamp it
        if self.energy > new_max:
            self.energy = new_max

        self.max_energy = new_max

    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of the energy budget state."""
        with self._lock:
            return {
                'energy': round(self.energy, 2),
                'max_energy': self.max_energy,
                'energy_ratio': round(self.energy_ratio(), 4),
                'is_scarce': self.is_scarce(),
                'is_critical': self.is_critical(),
                'action_cost_multiplier': round(self.action_cost_multiplier(), 2),
                'total_produced': round(self.total_produced, 2),
                'total_consumed': round(self.total_consumed, 2),
                'total_spent_on_actions': round(self.total_spent_on_actions, 2),
                'tick': self._tick,
                'recent_scarcity_events': self.scarcity_events[-5:],
            }
