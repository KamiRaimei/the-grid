"""CLI entry point for the Grid Simulation."""

import os
import sys
import time
import traceback
import argparse

from .cells import GRID_WIDTH, GRID_HEIGHT, PERSONALITY_FILE, CURSES_AVAILABLE
from .simulation import EnhancedTRONSimulation


def main():
    """Enhanced main entry point with advanced learning"""
    parser = argparse.ArgumentParser(description="Grid Simulation - Fibonacci Sequence")
    parser.add_argument("--no-curses", action="store_true", help="Disable curses interface")
    parser.add_argument("--width", type=int, default=GRID_WIDTH, help="Grid width")
    parser.add_argument("--height", type=int, default=GRID_HEIGHT, help="Grid height")
    parser.add_argument("--load-personality", type=str, help="Load specific personality file")
    parser.add_argument("--reset-learning", action="store_true", help="Reset learning data")
    parser.add_argument("--learning-rate", type=float, help="Set initial learning rate")
    parser.add_argument("--exploration", type=float, help="Set initial exploration rate")

    args = parser.parse_args()

    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive integers")
    if args.learning_rate is not None and not 0 < args.learning_rate <= 1:
        parser.error("--learning-rate must be greater than 0 and at most 1")
    if args.exploration is not None and not 0 <= args.exploration <= 1:
        parser.error("--exploration must be between 0 and 1")

    # Set personality file if specified
    if args.load_personality:
        global PERSONALITY_FILE
        PERSONALITY_FILE = args.load_personality

    print("=" * 70)
    print("GRID SIMULATION - REINFORCEMENT LEARNING MCP AI")
    print("System Objective: Learn optimal policy through reward/punishment training")
    print(f"Personality file: {PERSONALITY_FILE}")

    if args.reset_learning:
        print("Resetting learned training data...")
    elif os.path.exists(PERSONALITY_FILE):
        print("Found existing personality. Loading learned behavior...")
    else:
        print("No existing personality. Starting new training session...")
    print("=" * 70)

    time.sleep(2)

    if args.no_curses or not CURSES_AVAILABLE:
        print("Starting in fallback mode (no curses)...")
        sim = EnhancedTRONSimulation(
            use_curses=False,
            width=args.width,
            height=args.height,
            learning_rate=args.learning_rate,
            exploration_rate=args.exploration,
            reset_learning=args.reset_learning
        )
    else:
        print("Starting with enhanced visual interface...")
        time.sleep(1)
        sim = EnhancedTRONSimulation(
            use_curses=True,
            width=args.width,
            height=args.height,
            learning_rate=args.learning_rate,
            exploration_rate=args.exploration,
            reset_learning=args.reset_learning
        )

    # Run simulation
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n\nShutdown requested. Saving personality...")
        # Save personality on exit
        if hasattr(sim, 'mcp') and hasattr(sim.mcp, 'learning_system'):
            sim.mcp.learning_system.save_personality()
            report = sim.mcp.learning_system.get_learning_report()
            print(f"\nLearning Summary:")
            print(f"  Training Steps: {report['training_summary']['training_steps']}")
            print(f"  Success Rate: {report['training_summary']['success_rate']}%")
            print(f"  Total Reward: {report['training_summary']['total_reward']}")
        print("Simulation terminated.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
