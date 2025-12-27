#!/usr/bin/env python3
"""
Realistic Latency Modeling for LOB Simulation.

Architecture:
1. MarketConditions (Environment State) - Updated ONCE per timestep
   - Current regime (Calm/Stressed)
   - Global congestion delay (volume-dependent)

2. AgentLatencyProfile (Agent State) - Static per agent type
   - Base latency (hardware floor)
   - Jitter sigma (variance in their stack)

3. Total Latency = Base + Global_Congestion + Jitter

Key Insight:
- HFT vs Institutional is the only battle that matters
- HFT: Low base, low variance → consistent execution
- Institutional: Higher base, higher variance → gets picked off during stress

References:
- Real market latency is right-tailed (Log-Normal)
- Latency "snaps" between regimes rather than drifting
- High volume → exchange queue backup → higher latency for everyone
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MarketRegime(Enum):
    """Market regime affects latency distribution for all participants."""
    CALM = "calm"       # Normal trading, low variance
    STRESSED = "stressed"  # High activity, high variance, adverse selection risk


@dataclass
class AgentLatencyProfile:
    """
    Agent-specific latency profile (static attributes).
    
    Two agent types that matter:
    - HFT: Co-located, FPGA, ~0.5ms base, low variance
    - Institutional: Good infrastructure, ~10ms base, higher variance
    
    Retail is modeled as random order flow, not as latency-sensitive agents.
    """
    name: str
    base_latency_ns: int      # Minimum latency (speed of light + processing)
    jitter_sigma_calm: float  # Log-normal sigma in CALM regime
    jitter_sigma_stressed: float  # Log-normal sigma in STRESSED regime
    
    @classmethod
    def hft(cls) -> "AgentLatencyProfile":
        """
        High-frequency trader: co-located, FPGA.
        
        Key characteristic: LOW VARIANCE
        They win by being consistent, not just fast.
        """
        return cls(
            name="HFT",
            base_latency_ns=500_000,        # 0.5ms base
            jitter_sigma_calm=0.2,          # Very tight in calm
            jitter_sigma_stressed=0.4,      # Still tight in stress
        )
    
    @classmethod
    def institutional(cls) -> "AgentLatencyProfile":
        """
        Institutional algo: good infrastructure but not co-located.
        
        Key characteristic: HIGH VARIANCE IN STRESS
        They get picked off during volatility because their latency spikes.
        """
        return cls(
            name="Institutional",
            base_latency_ns=10_000_000,     # 10ms base
            jitter_sigma_calm=0.3,          # Moderate in calm
            jitter_sigma_stressed=0.8,      # High variance in stress!
        )
    
    @classmethod
    def custom(cls, name: str, base_ms: float, 
               sigma_calm: float = 0.3, sigma_stressed: float = 0.6) -> "AgentLatencyProfile":
        """Create custom agent profile."""
        return cls(
            name=name,
            base_latency_ns=int(base_ms * 1_000_000),
            jitter_sigma_calm=sigma_calm,
            jitter_sigma_stressed=sigma_stressed,
        )


class MarketConditions:
    """
    Environment state: Updated ONCE per simulation timestep.
    
    This class answers: "What are the current market conditions that
    affect ALL participants equally?"
    
    Components:
    1. Regime (Calm/Stressed) - binary state with transition probabilities
    2. Global Congestion - volume-dependent delay affecting everyone
    
    CRITICAL: Call update() exactly ONCE per simulation step, not per agent.
    """
    
    def __init__(
        self,
        # Volume sensitivity
        volume_sensitivity: float = 0.001,  # η: ns of delay per unit of excess volume
        baseline_volume: float = 1000.0,    # V_0: "normal" volume level
        # Regime transition probabilities  
        p_calm_to_stressed: float = 0.02,   # Base probability
        p_stressed_to_calm: float = 0.15,   # Recovery probability
        # Regime amplification from volume
        volume_stress_factor: float = 0.5,  # How much volume increases stress probability
        # Random seed
        seed: Optional[int] = None,
    ):
        self.volume_sensitivity = volume_sensitivity
        self.baseline_volume = baseline_volume
        self.p_calm_to_stressed = p_calm_to_stressed
        self.p_stressed_to_calm = p_stressed_to_calm
        self.volume_stress_factor = volume_stress_factor
        
        self.rng = np.random.default_rng(seed)
        
        # Current state
        self.regime = MarketRegime.CALM
        self.global_congestion_ns = 0
        self.current_volume = baseline_volume
        
        # Statistics
        self._regime_history = []
        self._congestion_history = []
    
    def update(self, volume: float):
        """
        Update market conditions based on current volume.
        
        CALL THIS EXACTLY ONCE PER SIMULATION TIMESTEP.
        
        Args:
            volume: Current market volume (e.g., events processed, trades, etc.)
        """
        self.current_volume = volume
        
        # 1. Update regime
        self._update_regime(volume)
        
        # 2. Calculate global congestion delay
        self._update_congestion(volume)
        
        # Track history
        self._regime_history.append(self.regime)
        self._congestion_history.append(self.global_congestion_ns)
    
    def _update_regime(self, volume: float):
        """
        Update market regime based on volume and transition probabilities.
        
        High volume increases probability of STRESSED regime.
        """
        volume_ratio = volume / self.baseline_volume if self.baseline_volume > 0 else 1.0
        
        if self.regime == MarketRegime.CALM:
            # Higher volume → higher chance of transitioning to stressed
            # P(calm→stressed) = base_p * (1 + factor * excess_volume_ratio)
            excess_ratio = max(0, volume_ratio - 1.0)
            transition_prob = self.p_calm_to_stressed * (1 + self.volume_stress_factor * excess_ratio)
            transition_prob = min(transition_prob, 0.5)  # Cap at 50%
            
            if self.rng.random() < transition_prob:
                self.regime = MarketRegime.STRESSED
        else:
            # Stressed regime recovers with fixed probability
            # (stress doesn't last forever)
            if self.rng.random() < self.p_stressed_to_calm:
                self.regime = MarketRegime.CALM
    
    def _update_congestion(self, volume: float):
        """
        Calculate global congestion delay affecting all participants.
        
        Formula: congestion = η * max(0, V - V_0)
        
        When volume exceeds baseline, the exchange matching engine
        queues up, adding delay for everyone.
        """
        excess_volume = max(0, volume - self.baseline_volume)
        self.global_congestion_ns = int(self.volume_sensitivity * excess_volume * 1_000_000)
    
    def get_statistics(self) -> dict:
        """Get statistics on market conditions history."""
        if not self._regime_history:
            return {}
        
        stressed_count = sum(1 for r in self._regime_history if r == MarketRegime.STRESSED)
        congestion = np.array(self._congestion_history)
        
        return {
            "n_steps": len(self._regime_history),
            "stressed_fraction": stressed_count / len(self._regime_history),
            "mean_congestion_ms": np.mean(congestion) / 1e6 if len(congestion) > 0 else 0,
            "max_congestion_ms": np.max(congestion) / 1e6 if len(congestion) > 0 else 0,
        }
    
    def reset(self):
        """Reset market conditions to initial state."""
        self.regime = MarketRegime.CALM
        self.global_congestion_ns = 0
        self.current_volume = self.baseline_volume
        self._regime_history = []
        self._congestion_history = []


class AgentLatencyModel:
    """
    Computes total latency for an agent given current market conditions.
    
    Total Latency = Base + Global_Congestion + Jitter
    
    Where:
    - Base: Agent's hardware floor (static)
    - Global_Congestion: From MarketConditions (same for all agents)
    - Jitter: Log-normal noise (sigma depends on regime and agent type)
    
    The key insight: HFT has low variance even in stress, while
    Institutional variance explodes → adverse selection.
    """
    
    def __init__(
        self,
        profile: AgentLatencyProfile,
        seed: Optional[int] = None,
    ):
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        
        # Statistics
        self._latency_history = []
        self._jitter_history = []
    
    def sample(self, market: MarketConditions) -> int:
        """
        Sample total latency in nanoseconds given current market conditions.
        
        Args:
            market: Current MarketConditions (should be updated once per step)
            
        Returns:
            Total latency in nanoseconds
        """
        # 1. Base latency (agent's hardware floor)
        base_ns = self.profile.base_latency_ns
        
        # 2. Global congestion (from market, same for everyone)
        congestion_ns = market.global_congestion_ns
        
        # 3. Jitter (log-normal, sigma depends on regime)
        jitter_ns = self._sample_jitter(market.regime)
        
        # Total
        total_ns = base_ns + congestion_ns + jitter_ns
        
        self._latency_history.append(total_ns)
        self._jitter_history.append(jitter_ns)
        
        return total_ns
    
    def _sample_jitter(self, regime: MarketRegime) -> int:
        """
        Sample jitter using Log-Normal distribution.
        
        Sigma depends on current regime:
        - CALM: Lower sigma → tighter distribution
        - STRESSED: Higher sigma → fat tails, HFT vs Institutional diverges
        """
        if regime == MarketRegime.CALM:
            sigma = self.profile.jitter_sigma_calm
        else:
            sigma = self.profile.jitter_sigma_stressed
        
        # Log-normal with mean=0 gives median=1
        # We scale by a fraction of base latency
        jitter_factor = self.rng.lognormal(mean=0, sigma=sigma) - 1
        jitter_factor = max(0, jitter_factor)  # Can't be negative
        
        # Scale: jitter is a fraction of base latency
        jitter_ns = int(self.profile.base_latency_ns * jitter_factor)
        
        return jitter_ns
    
    def get_statistics(self) -> dict:
        """Get statistics on sampled latencies."""
        if not self._latency_history:
            return {}
        
        latencies = np.array(self._latency_history)
        jitters = np.array(self._jitter_history)
        
        return {
            "n_samples": len(latencies),
            "mean_ms": np.mean(latencies) / 1e6,
            "median_ms": np.median(latencies) / 1e6,
            "std_ms": np.std(latencies) / 1e6,
            "p95_ms": np.percentile(latencies, 95) / 1e6,
            "p99_ms": np.percentile(latencies, 99) / 1e6,
            "max_ms": np.max(latencies) / 1e6,
            "mean_jitter_ms": np.mean(jitters) / 1e6,
        }
    
    def reset(self):
        """Reset statistics."""
        self._latency_history = []
        self._jitter_history = []


def create_agent_latency(
    agent_type: str = "institutional",
    seed: Optional[int] = None,
) -> AgentLatencyModel:
    """
    Factory function to create agent latency models.
    
    Args:
        agent_type: "hft", "institutional", or custom "base_ms:sigma_calm:sigma_stressed"
        seed: Random seed for reproducibility
        
    Returns:
        Configured AgentLatencyModel
    """
    agent_type_lower = agent_type.lower()
    
    if agent_type_lower == "hft":
        profile = AgentLatencyProfile.hft()
    elif agent_type_lower == "institutional":
        profile = AgentLatencyProfile.institutional()
    elif ":" in agent_type:
        # Custom format: "base_ms:sigma_calm:sigma_stressed" or "base_ms:sigma"
        parts = agent_type.split(":")
        base_ms = float(parts[0])
        sigma_calm = float(parts[1]) if len(parts) > 1 else 0.3
        sigma_stressed = float(parts[2]) if len(parts) > 2 else sigma_calm * 2
        profile = AgentLatencyProfile.custom(
            f"Custom_{base_ms}ms", base_ms, sigma_calm, sigma_stressed
        )
    else:
        # Assume it's a base latency in ms
        try:
            base_ms = float(agent_type)
            profile = AgentLatencyProfile.custom(f"Custom_{base_ms}ms", base_ms)
        except ValueError:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    return AgentLatencyModel(profile, seed=seed)


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Latency Model: HFT vs Institutional Battle")
    print("=" * 70)
    
    # Create market conditions (shared by all agents)
    market = MarketConditions(seed=42)
    
    # Create two agents
    hft = create_agent_latency("hft", seed=100)
    inst = create_agent_latency("institutional", seed=200)
    
    print(f"\nAgent Profiles:")
    print(f"  HFT:          {hft.profile.base_latency_ns/1e6:.2f}ms base, "
          f"σ_calm={hft.profile.jitter_sigma_calm}, σ_stress={hft.profile.jitter_sigma_stressed}")
    print(f"  Institutional: {inst.profile.base_latency_ns/1e6:.2f}ms base, "
          f"σ_calm={inst.profile.jitter_sigma_calm}, σ_stress={inst.profile.jitter_sigma_stressed}")
    
    # Simulate market with volume spikes
    print("\n" + "=" * 70)
    print("Simulation: Normal → Volume Spike → Recovery")
    print("=" * 70)
    
    volumes = (
        [1000] * 50 +   # Normal
        [5000] * 20 +   # Spike (news event)
        [1000] * 30     # Recovery
    )
    
    hft_latencies = []
    inst_latencies = []
    
    for i, vol in enumerate(volumes):
        # 1. Update market conditions ONCE per step
        market.update(vol)
        
        # 2. Each agent samples their latency
        hft_lat = hft.sample(market)
        inst_lat = inst.sample(market)
        
        hft_latencies.append(hft_lat)
        inst_latencies.append(inst_lat)
        
        # Print key moments
        if i == 0:
            print(f"\nStep {i:3d} [NORMAL]: Vol={vol}")
            print(f"  Regime: {market.regime.value}, Congestion: {market.global_congestion_ns/1e6:.2f}ms")
            print(f"  HFT: {hft_lat/1e6:.2f}ms | Inst: {inst_lat/1e6:.2f}ms")
        elif i == 50:
            print(f"\nStep {i:3d} [SPIKE STARTS]: Vol={vol}")
            print(f"  Regime: {market.regime.value}, Congestion: {market.global_congestion_ns/1e6:.2f}ms")
            print(f"  HFT: {hft_lat/1e6:.2f}ms | Inst: {inst_lat/1e6:.2f}ms")
        elif i == 69:
            print(f"\nStep {i:3d} [SPIKE PEAK]: Vol={vol}")
            print(f"  Regime: {market.regime.value}, Congestion: {market.global_congestion_ns/1e6:.2f}ms")
            print(f"  HFT: {hft_lat/1e6:.2f}ms | Inst: {inst_lat/1e6:.2f}ms")
    
    # Statistics
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    
    hft_arr = np.array(hft_latencies) / 1e6
    inst_arr = np.array(inst_latencies) / 1e6
    
    # During normal (first 50 steps)
    print("\nDuring NORMAL trading (steps 0-49):")
    print(f"  HFT:  Mean={np.mean(hft_arr[:50]):.2f}ms, P99={np.percentile(hft_arr[:50], 99):.2f}ms")
    print(f"  Inst: Mean={np.mean(inst_arr[:50]):.2f}ms, P99={np.percentile(inst_arr[:50], 99):.2f}ms")
    
    # During spike (steps 50-69)
    print("\nDuring VOLUME SPIKE (steps 50-69):")
    print(f"  HFT:  Mean={np.mean(hft_arr[50:70]):.2f}ms, P99={np.percentile(hft_arr[50:70], 99):.2f}ms")
    print(f"  Inst: Mean={np.mean(inst_arr[50:70]):.2f}ms, P99={np.percentile(inst_arr[50:70], 99):.2f}ms")
    
    # The key metric: how often does HFT beat Institutional?
    hft_wins = sum(1 for h, i in zip(hft_latencies, inst_latencies) if h < i)
    print(f"\nHFT beats Institutional: {hft_wins}/{len(volumes)} = {100*hft_wins/len(volumes):.1f}%")
    
    # Adverse selection: during stress, what's the latency gap?
    stress_steps = [i for i, r in enumerate(market._regime_history) if r == MarketRegime.STRESSED]
    if stress_steps:
        gaps = [inst_latencies[i] - hft_latencies[i] for i in stress_steps]
        print(f"\nDuring STRESSED regime ({len(stress_steps)} steps):")
        print(f"  Mean latency gap: {np.mean(gaps)/1e6:.2f}ms (Inst slower than HFT)")
        print(f"  Max latency gap:  {np.max(gaps)/1e6:.2f}ms")
    
    print(f"\nMarket Stats: {market.get_statistics()}")
