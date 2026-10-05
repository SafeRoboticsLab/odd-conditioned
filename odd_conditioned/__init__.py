"""ODD-conditioned safety filters: certified mode switching under runtime changes of the operating design domain.

    scenarios     the evaluated tasks, ODD schedules and per-regime switching thresholds
    automaton     the safety filter (an automaton over specification modes) and its baselines: ``rollout``
    certificates  value sweeps, handoff ramps and region grids: does V_stand track certifiability?
    policies      the reach-avoid twins (safety-stable-baselines) and the nominal walker (go2_atomic_skills)
    sim           the robot-safety-sandbox environment layer
    evaluate / figures / videos   the reported results, regenerated
"""
__version__ = "1.0.0"
