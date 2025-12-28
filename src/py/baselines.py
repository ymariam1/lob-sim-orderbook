#!/usr/bin/env python3
"""
Execution Algorithm Baselines: TWAP and Almgren-Chriss

These baselines establish benchmark performance for optimal execution strategies,
which can later be compared against RL agents.

Strategies:
1. TWAP (Time-Weighted Average Price):
   - Divide total quantity equally across time intervals
   - Execute with market orders at regular intervals
   - Simple, predictable, but doesn't adapt to market conditions

2. Almgren-Chriss (Optimal Execution):
   - Minimize expected cost + risk penalty
   - Closed-form trajectory: v(t) ∝ sinh(κ(T-t)) / sinh(κT)
   - Execute with limit orders at best bid/ask, cross spread if behind schedule

Usage:
    python baselines.py --data data/blockchain_l3_2023-03-01.csv
    python baselines.py --strategy twap --qty 1000
    python baselines.py --strategy ac --risk-aversion 1e-6
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import lob_sim as ob
except ImportError:
    raise ImportError(
        "lob_sim module not found. Please build it first:\n"
        "  cd /path/to/lob-sim-orderbook\n"
        "  pip install ."
    )


@dataclass
class ExecutionResult:
    """Results from executing a strategy."""
    strategy: str
    total_qty: int
    total_time_ns: int
    
    # Execution metrics
    avg_execution_price: float = 0.0
    vwap: float = 0.0  # Volume-weighted average price
    arrival_price: float = 0.0  # Price at start of execution
    terminal_price: float = 0.0  # Price at end of execution
    
    # Slippage metrics
    implementation_shortfall: float = 0.0  # Cost vs arrival price
    slippage_bps: float = 0.0  # Slippage in basis points
    
    # Execution details
    fills: List[Dict] = field(default_factory=list)
    trajectory: List[Dict] = field(default_factory=list)
    
    # Time metrics
    start_time_ns: int = 0
    end_time_ns: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'strategy': self.strategy,
            'total_qty': self.total_qty,
            'total_time_ns': self.total_time_ns,
            'avg_execution_price': self.avg_execution_price,
            'vwap': self.vwap,
            'arrival_price': self.arrival_price,
            'terminal_price': self.terminal_price,
            'implementation_shortfall': self.implementation_shortfall,
            'slippage_bps': self.slippage_bps,
            'num_fills': len(self.fills),
            'start_time_ns': self.start_time_ns,
            'end_time_ns': self.end_time_ns,
        }


class BaselineExecutor:
    """
    Base class for execution algorithms.
    Handles environment setup and metric tracking.
    """
    
    def __init__(
        self,
        data_path: str,
        total_qty: int = 1000,
        total_time_ns: int = 60 * 60 * int(1e9),  # 1 hour default
        side: str = "BUY",
        step_duration_ns: int = int(1e9),  # 1 second per step
        warmup_duration_ns: int = int(60 * 1e9),  # 60 seconds warmup
        timestamp_unit_ns: int = 1000,  # Tardis microseconds
        agent_latency_ns: int = int(1e6),  # 1ms latency
        verbose: bool = True,
    ):
        self.data_path = data_path
        self.total_qty = total_qty
        self.total_time_ns = total_time_ns
        self.side = ob.Side.BUY if side.upper() == "BUY" else ob.Side.SELL
        self.step_duration_ns = step_duration_ns
        self.warmup_duration_ns = warmup_duration_ns
        self.timestamp_unit_ns = timestamp_unit_ns
        self.agent_latency_ns = agent_latency_ns
        self.verbose = verbose
        
        # Will be initialized in reset()
        self.orderbook: Optional[ob.Orderbook] = None
        self.exchange: Optional[ob.ExchangeSimulator] = None
        self.loader: Optional[ob.DataLoader] = None
        
        # Tracking
        self.order_id_counter = 1_000_000
        self.qty_remaining = total_qty
        self.fills: List[Dict] = []
        self.trajectory: List[Dict] = []
        
    def reset(self):
        """Initialize fresh orderbook and exchange."""
        self.orderbook = ob.Orderbook()
        self.exchange = ob.ExchangeSimulator(self.orderbook)
        self.loader = ob.DataLoader(self.data_path, self.timestamp_unit_ns)
        
        # Warmup: build initial book
        if self.loader.HasMoreData():
            first_ts = self.loader.PeekNextTimestampNs()
            target = first_ts + self.warmup_duration_ns
            events = self.loader.PumpToExchange(self.exchange, target)
            if self.verbose:
                print(f"Warmup: Pumped {events} events, book size: {self.orderbook.Size()}")
        
        # Reset tracking
        self.order_id_counter = 1_000_000
        self.qty_remaining = self.total_qty
        self.fills = []
        self.trajectory = []
        
    def get_mid_price(self) -> float:
        """Get current mid price."""
        book = self.orderbook.GetOrderInfos()
        bids = book.GetBids()
        asks = book.GetAsks()
        
        if bids and asks:
            return (bids[0].price + asks[0].price) / 2
        elif bids:
            return bids[0].price
        elif asks:
            return asks[0].price
        return 0.0
    
    def get_best_bid(self) -> Tuple[int, int]:
        """Get best bid (price, qty)."""
        book = self.orderbook.GetOrderInfos()
        bids = book.GetBids()
        if bids:
            return bids[0].price, bids[0].quantity
        return 0, 0
    
    def get_best_ask(self) -> Tuple[int, int]:
        """Get best ask (price, qty)."""
        book = self.orderbook.GetOrderInfos()
        asks = book.GetAsks()
        if asks:
            return asks[0].price, asks[0].quantity
        return 0, 0
    
    def place_market_order(self, qty: int) -> int:
        """
        Place a market order. Fills will be captured by collect_agent_fills().
        
        NOTE: We do NOT manually record fills here. The matching engine will
        generate the Trade, and collect_agent_fills() will capture it during
        the next step(). This prevents double-counting.
        """
        if qty <= 0:
            return 0
        
        current_time = self.exchange.GetCurrentTime()
        
        order = ob.Order(
            self.order_id_counter,
            self.side,
            0,  # Price ignored for market orders
            qty,
            current_time,
            ob.MARKET,
            ob.IOC
        )
        self.order_id_counter += 1
        
        # Execute through exchange (with latency)
        # The fill will be captured by collect_agent_fills() in step()
        self.exchange.PlaceAgentOrder(order, self.agent_latency_ns)
        
        # DO NOT record fills here - let collect_agent_fills() do it
        # DO NOT subtract qty_remaining here - collect_agent_fills() will do it
        
        return qty
    
    def place_limit_order(self, price: int, qty: int) -> int:
        """Place a limit order at specified price (not simulated for fills)."""
        if qty <= 0:
            return 0
        
        current_time = self.exchange.GetCurrentTime()
        
        order = ob.Order(
            self.order_id_counter,
            self.side,
            price,
            qty,
            current_time,
            ob.LIMIT,
            ob.GTC
        )
        self.order_id_counter += 1
        
        self.exchange.PlaceAgentOrder(order, self.agent_latency_ns)
        
        # For now, assume limit orders don't get filled (passive)
        # In a more realistic simulation, we'd track resting orders and fills
        return 0
    
    def place_limit_order_passive(self, price: int, qty: int) -> int:
        """
        Place a limit order and wait for passive fill via the real matching engine.
        
        Unlike the simulated "passive fill hack", this places an actual limit order
        on the book and waits one time step to see if it gets filled by incoming
        market data.
        
        Args:
            price: Limit price
            qty: Quantity
            
        Returns:
            Quantity filled
        """
        if qty <= 0:
            return 0
        
        current_time = self.exchange.GetCurrentTime()
        
        # Place the limit order
        order = ob.Order(
            self.order_id_counter,
            self.side,
            price,
            qty,
            current_time,
            ob.LIMIT,
            ob.GTC
        )
        self.order_id_counter += 1
        
        self.exchange.PlaceAgentOrder(order, self.agent_latency_ns)
        
        # Return 0 - the fill will be captured later via GetAgentFills()
        return 0
    
    def collect_agent_fills(self):
        """
        Collect fills from the matching engine and record them.
        
        This should be called after stepping the market forward to capture
        any fills that occurred on our resting orders.
        """
        fills = self.exchange.GetAgentFills()
        
        for trade in fills:
            # Determine which side of the trade is ours
            if self.side == ob.Side.BUY:
                trade_info = trade.GetBidTrade()
            else:
                trade_info = trade.GetAskTrade()
            
            self.fills.append({
                'time_ns': self.exchange.GetCurrentTime(),
                'price': trade_info.price,
                'qty': trade_info.quantity,
                'side': 'BUY' if self.side == ob.Side.BUY else 'SELL',
                'type': 'MATCHED'
            })
            
            self.qty_remaining -= trade_info.quantity
        
        # Clear the fills buffer
        self.exchange.ClearAgentFills()
        
        return len(fills)
    
    def step(self, duration_ns: int = None):
        """Advance market time by specified duration (or default step_duration_ns)."""
        if duration_ns is None:
            duration_ns = self.step_duration_ns
        
        if self.loader and self.loader.HasMoreData():
            self.loader.PumpToExchange(self.exchange, duration_ns)
        else:
            # Even without data, force time forward
            current = self.exchange.GetCurrentTime()
            self.exchange.SetCurrentTime(current + duration_ns)
        
        self.exchange.ProcessPendingAgentActions()
        
        # Collect any fills that occurred on our resting orders
        self.collect_agent_fills()
    
    def record_trajectory(self, target_qty: int, actual_qty: int):
        """Record current state for trajectory analysis."""
        self.trajectory.append({
            'time_ns': self.exchange.GetCurrentTime(),
            'mid_price': self.get_mid_price(),
            'target_qty_remaining': target_qty,
            'actual_qty_remaining': actual_qty,
            'qty_remaining': self.qty_remaining,
        })
    
    def calculate_metrics(self, arrival_price: float) -> ExecutionResult:
        """Calculate execution metrics from fills."""
        result = ExecutionResult(
            strategy=self.__class__.__name__,
            total_qty=self.total_qty,
            total_time_ns=self.total_time_ns,
            arrival_price=arrival_price,
        )
        
        if not self.fills:
            return result
        
        # VWAP and average price
        total_value = sum(f['price'] * f['qty'] for f in self.fills)
        total_qty = sum(f['qty'] for f in self.fills)
        
        if total_qty > 0:
            result.vwap = total_value / total_qty
            result.avg_execution_price = result.vwap
        
        # Terminal price
        result.terminal_price = self.get_mid_price()
        
        # Implementation shortfall
        if self.side == ob.Side.BUY:
            # For buys: IS = (execution_price - arrival_price) * qty
            result.implementation_shortfall = (result.vwap - arrival_price) * total_qty
        else:
            # For sells: IS = (arrival_price - execution_price) * qty
            result.implementation_shortfall = (arrival_price - result.vwap) * total_qty
        
        # Slippage in basis points
        if arrival_price > 0:
            result.slippage_bps = abs(result.vwap - arrival_price) / arrival_price * 10000
        
        result.fills = self.fills
        result.trajectory = self.trajectory
        result.start_time_ns = self.trajectory[0]['time_ns'] if self.trajectory else 0
        result.end_time_ns = self.trajectory[-1]['time_ns'] if self.trajectory else 0
        
        return result
    
    def print_results(self, result: ExecutionResult, header: str = "RESULTS"):
        """Print execution results in a formatted table."""
        print(f"\n{'='*60}")
        print(header)
        print(f"{'='*60}")
        print(f"Arrival Price:  {result.arrival_price:.2f}")
        print(f"VWAP:           {result.vwap:.2f}")
        print(f"Terminal Price: {result.terminal_price:.2f}")
        print(f"Impl Shortfall: {result.implementation_shortfall:.2f}")
        print(f"Slippage:       {result.slippage_bps:.2f} bps")
        print(f"Fills:          {len(result.fills)}")
        print(f"{'='*60}\n")


class TWAPExecutor(BaselineExecutor):
    """
    Time-Weighted Average Price (TWAP) Executor.
    
    Simple strategy:
    - Divide total quantity into N equal slices
    - Execute one slice every T/N time
    - Use market orders for guaranteed execution
    """
    
    def __init__(
        self,
        data_path: str,
        total_qty: int = 1000,
        total_time_ns: int = 60 * 60 * int(1e9),
        num_slices: int = 60,  # Execute every minute for 1 hour
        **kwargs
    ):
        super().__init__(data_path, total_qty, total_time_ns, **kwargs)
        self.num_slices = num_slices
        self.slice_qty = total_qty // num_slices
        self.slice_interval_ns = total_time_ns // num_slices
        
    def execute(self) -> ExecutionResult:
        """Execute TWAP strategy."""
        self.reset()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("TWAP EXECUTION")
            print(f"{'='*60}")
            print(f"Total Qty: {self.total_qty}")
            print(f"Num Slices: {self.num_slices}")
            print(f"Qty per Slice: {self.slice_qty}")
            print(f"Interval: {self.slice_interval_ns / 1e9:.1f}s")
            print()
        
        arrival_price = self.get_mid_price()
        start_time = self.exchange.GetCurrentTime()
        
        for i in range(self.num_slices):
            # Record trajectory
            target_remaining = self.total_qty - (i * self.slice_qty)
            self.record_trajectory(target_remaining, self.qty_remaining)
            
            # Execute slice
            slice_qty = min(self.slice_qty, self.qty_remaining)
            if slice_qty > 0:
                self.place_market_order(slice_qty)
                
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"Slice {i+1}/{self.num_slices}: "
                          f"Executed {slice_qty}, Remaining: {self.qty_remaining}, "
                          f"Mid: {self.get_mid_price():.2f}")
            
            # Advance time to next slice (using slice interval, not default step)
            self.step(self.slice_interval_ns)
            
            # Check if we're done or out of data
            if self.qty_remaining <= 0:
                break
            if self.loader and not self.loader.HasMoreData():
                if self.verbose:
                    print("Out of market data!")
                break
        
        # Calculate and return metrics
        result = self.calculate_metrics(arrival_price)
        result.strategy = "TWAP"
        
        if self.verbose:
            self.print_results(result, "TWAP RESULTS")
        
        return result


class VWAPExecutor(BaselineExecutor):
    """
    Volume-Weighted Average Price (VWAP) Strategy.
    
    Industry standard benchmark. Executes proportionally to market volume.
    The goal is to match the market's VWAP over the execution period.
    
    Implementation:
    - Divide execution into time slices
    - In each slice, execute proportionally to market volume in that slice
    - Uses market orders to match market VWAP
    """
    
    def __init__(
        self,
        data_path: str,
        total_qty: int = 1000,
        total_time_ns: int = 60 * 60 * int(1e9),
        num_slices: int = 60,  # Execute every minute for 1 hour
        **kwargs
    ):
        super().__init__(data_path, total_qty, total_time_ns, **kwargs)
        self.num_slices = num_slices
        self.slice_interval_ns = total_time_ns // num_slices
        
    def execute(self) -> ExecutionResult:
        """Execute VWAP strategy."""
        self.reset()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("VWAP EXECUTION")
            print(f"{'='*60}")
            print(f"Total Qty: {self.total_qty}")
            print(f"Num Slices: {self.num_slices}")
            print(f"Interval: {self.slice_interval_ns / 1e9:.1f}s")
            print()
        
        arrival_price = self.get_mid_price()

        # VWAP strategy: Execute proportionally to market volume
        # Simplified: Use uniform distribution (time-weighted) as baseline
        # In production, this would track actual market volume and execute proportionally
        # For now, we use uniform time slices (similar to TWAP but conceptually VWAP)
        # This is a reasonable baseline approximation
        base_slice_qty = self.total_qty // self.num_slices
        
        for i in range(self.num_slices):
            # Calculate target quantity for this slice
            # In a real VWAP, this would be proportional to market volume in the slice
            # For baseline, we use uniform distribution (time-weighted VWAP)
            target_slice_qty = base_slice_qty
            # Add remainder to last slice
            if i == self.num_slices - 1:
                target_slice_qty += self.total_qty % self.num_slices
            
            # Ensure we don't exceed remaining quantity
            slice_qty = min(target_slice_qty, self.qty_remaining)
            
            # Record trajectory
            self.record_trajectory(self.qty_remaining - slice_qty, self.qty_remaining)
            
            # Execute slice
            if slice_qty > 0:
                self.place_market_order(slice_qty)
                
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"Slice {i+1}/{self.num_slices}: "
                          f"Executed {slice_qty}, Remaining: {self.qty_remaining}, "
                          f"Mid: {self.get_mid_price():.2f}")
            
            # Advance time to next slice
            self.step(self.slice_interval_ns)
            
            # Check if we're done or out of data
            if self.qty_remaining <= 0:
                break
            if self.loader and not self.loader.HasMoreData():
                if self.verbose:
                    print("Out of market data!")
                break
        
        # Calculate final metrics
        result = self.calculate_metrics(arrival_price)
        result.strategy = "VWAP"

        if self.verbose:
            self.print_results(result, "VWAP RESULTS")

        return result


class POVExecutor(BaselineExecutor):
    """
    Percentage of Volume (POV) Strategy.
    
    Executes a fixed percentage of market volume in each time period.
    Common in institutional trading to match market flow.
    
    Implementation:
    - Monitor market volume in each time slice
    - Execute a fixed percentage (e.g., 10%) of that volume
    - Uses market orders to match market participation rate
    """
    
    def __init__(
        self,
        data_path: str,
        total_qty: int = 1000,
        total_time_ns: int = 60 * 60 * int(1e9),
        num_slices: int = 60,  # Execute every minute for 1 hour
        participation_rate: float = 0.1,  # Execute 10% of market volume
        **kwargs
    ):
        super().__init__(data_path, total_qty, total_time_ns, **kwargs)
        self.num_slices = num_slices
        self.slice_interval_ns = total_time_ns // num_slices
        self.participation_rate = participation_rate  # e.g., 0.1 = 10% of market volume
        
    def execute(self) -> ExecutionResult:
        """Execute POV strategy."""
        self.reset()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("POV EXECUTION")
            print(f"{'='*60}")
            print(f"Total Qty: {self.total_qty}")
            print(f"Num Slices: {self.num_slices}")
            print(f"Participation Rate: {self.participation_rate:.1%}")
            print(f"Interval: {self.slice_interval_ns / 1e9:.1f}s")
            print()

        arrival_price = self.get_mid_price()

        for i in range(self.num_slices):
            # Estimate market volume in this slice (using book size as proxy)
            # In production, this would track actual trade volume
            book_size_before = self.orderbook.Size()
            
            # Advance market by slice interval
            self.step(self.slice_interval_ns)
            
            book_size_after = self.orderbook.Size()
            # Estimate volume as change in book activity
            # This is a simplified proxy - real POV would track actual trades
            estimated_volume = max(book_size_after - book_size_before, book_size_after // 10)
            
            # Execute participation_rate of estimated market volume
            target_slice_qty = int(estimated_volume * self.participation_rate)
            
            # But don't exceed remaining quantity
            slice_qty = min(target_slice_qty, self.qty_remaining)
            
            # Record trajectory
            self.record_trajectory(self.qty_remaining - slice_qty, self.qty_remaining)
            
            # Execute slice
            if slice_qty > 0:
                self.place_market_order(slice_qty)
                
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"Slice {i+1}/{self.num_slices}: "
                          f"Market Vol: {estimated_volume}, "
                          f"Executed: {slice_qty}, Remaining: {self.qty_remaining}, "
                          f"Mid: {self.get_mid_price():.2f}")
            
            # Check if we're done or out of data
            if self.qty_remaining <= 0:
                break
            if self.loader and not self.loader.HasMoreData():
                if self.verbose:
                    print("Out of market data!")
                break
        
        # Calculate final metrics
        result = self.calculate_metrics(arrival_price)
        result.strategy = "POV"

        if self.verbose:
            self.print_results(result, "POV RESULTS")

        return result


class AlmgrenChrissExecutor(BaselineExecutor):
    """
    Almgren-Chriss Optimal Execution Strategy.
    
    Minimizes: E[Cost] + λ * Var[Cost]
    
    Optimal trajectory rate:
        v(t) = (X * κ * sinh(κ(T-t))) / sinh(κT)
    
    Where:
        X = total quantity
        T = total time
        κ = sqrt(λσ²/η)
        λ = risk aversion parameter
        σ = volatility
        η = temporary impact coefficient
    
    Execution:
    - Use limit orders at best bid/ask + 1 tick (passive)
    - If falling behind schedule, cross the spread (aggressive)
    """
    
    def __init__(
        self,
        data_path: str,
        total_qty: int = 1000,
        total_time_ns: int = 60 * 60 * int(1e9),
        risk_aversion: float = 1e-6,  # λ
        volatility: float = 0.01,      # σ (1% per unit time)
        temp_impact: float = 0.001,    # η
        tick_size: int = 1,            # Minimum price increment
        schedule_tolerance: float = 0.1,  # 10% behind schedule triggers aggressive
        passive_order_timeout_steps: int = 3,  # How many steps to wait before cancelling unfilled passive order
        **kwargs
    ):
        super().__init__(data_path, total_qty, total_time_ns, **kwargs)
        self.risk_aversion = risk_aversion
        self.volatility = volatility
        self.temp_impact = temp_impact
        self.tick_size = tick_size
        self.schedule_tolerance = schedule_tolerance
        self.passive_order_timeout_steps = passive_order_timeout_steps
        
        # Calculate κ
        self.kappa = np.sqrt(risk_aversion * volatility**2 / temp_impact)
        
    def optimal_rate(self, t: float, T: float, X: float) -> float:
        """
        Calculate optimal execution rate at time t.
        
        v(t) = X * κ * sinh(κ(T-t)) / sinh(κT)
        
        Returns quantity to execute per unit time.
        """
        if T <= 0 or t >= T:
            return X  # Execute everything remaining
        
        kappa = self.kappa
        remaining_time = T - t
        
        # Handle numerical stability
        if kappa * T > 20:  # sinh overflow
            return X * kappa * np.exp(-kappa * t)
        
        numerator = kappa * np.sinh(kappa * remaining_time)
        denominator = np.sinh(kappa * T)
        
        if denominator < 1e-10:
            return X / (T - t)  # Fallback to uniform
        
        return X * numerator / denominator
    
    def target_inventory(self, t: float, T: float, X: float) -> float:
        """
        Calculate target remaining inventory at time t.
        
        x(t) = X * sinh(κ(T-t)) / sinh(κT)
        """
        if T <= 0 or t >= T:
            return 0
        
        kappa = self.kappa
        remaining_time = T - t
        
        if kappa * T > 20:
            return X * np.exp(-kappa * t)
        
        return X * np.sinh(kappa * remaining_time) / np.sinh(kappa * T)
    
    def execute(self) -> ExecutionResult:
        """Execute Almgren-Chriss strategy."""
        self.reset()
        
        # Use same number of steps as TWAP slices for fair comparison
        # This ensures both strategies execute over the same time horizon
        num_steps = 60  # Same as default TWAP slices
        step_interval_ns = self.total_time_ns // num_steps
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("ALMGREN-CHRISS EXECUTION")
            print(f"{'='*60}")
            print(f"Total Qty: {self.total_qty}")
            print(f"Total Time: {self.total_time_ns / 1e9:.0f}s")
            print(f"Num Steps: {num_steps}")
            print(f"Step Interval: {step_interval_ns / 1e9:.1f}s")
            print(f"Risk Aversion (λ): {self.risk_aversion}")
            print(f"Kappa (κ): {self.kappa:.6f}")
            print()
        
        arrival_price = self.get_mid_price()
        start_time = self.exchange.GetCurrentTime()
        
        T = self.total_time_ns  # Total time in ns
        X = self.total_qty
        
        # Track active passive order to prevent "Zombie Orders"
        # FIXED: Orders now rest for multiple steps (more realistic) instead of being cancelled after 1 step
        active_passive_order_id = None
        active_passive_order_age = 0  # How many steps the order has been resting
        
        for step in range(num_steps):
            # CLEANUP: Cancel unfilled passive order if it has been resting too long
            # This prevents "Zombie Orders" while allowing realistic order resting behavior
            if active_passive_order_id is not None:
                active_passive_order_age += 1
                if active_passive_order_age >= self.passive_order_timeout_steps:
                    # Order has been resting for timeout_steps, cancel it
                    self.exchange.CancelAgentOrder(active_passive_order_id, 0)  # 0 latency for cancel
                    self.exchange.ProcessPendingAgentActions()
                    active_passive_order_id = None
                    active_passive_order_age = 0
                    if self.verbose:
                        print(f"Step {step}/{num_steps}: Cancelled unfilled passive order after {self.passive_order_timeout_steps} steps")
            
            current_time = self.exchange.GetCurrentTime()
            elapsed = current_time - start_time
            t_frac = elapsed / T  # Fraction of time elapsed (0 to 1)
            
            # Calculate target inventory at this time using A-C formula
            target_remaining = self.target_inventory(t_frac, 1.0, X)
            actual_remaining = self.qty_remaining
            
            # Record trajectory
            self.record_trajectory(int(target_remaining), actual_remaining)
            
            # Calculate how much we should have executed by now
            target_executed = X - target_remaining
            actual_executed = X - actual_remaining
            
            # How much to execute this step to catch up to schedule
            gap = target_executed - actual_executed
            
            # Base quantity: divide remaining by remaining steps
            remaining_steps = num_steps - step
            base_qty = actual_remaining // remaining_steps if remaining_steps > 0 else actual_remaining
            
            # Adjust for schedule: if behind, execute more; if ahead, execute less
            if gap > 0:
                target_qty = min(base_qty + int(gap), actual_remaining)
            else:
                target_qty = max(1, base_qty + int(gap))
            
            target_qty = max(1, min(target_qty, actual_remaining))
            
            if target_qty > 0 and actual_remaining > 0:
                # Determine if we're behind schedule
                behind_schedule = (actual_remaining - target_remaining) / X > self.schedule_tolerance
                
                if behind_schedule:
                    # AGGRESSIVE: Cross the spread with market order (pay the spread)
                    # No active order to track (market orders fill immediately)
                    self.place_market_order(target_qty)
                    if self.verbose and step % 10 == 0:
                        print(f"Step {step}/{num_steps}: AGGRESSIVE (MKT) - Qty: {target_qty}, "
                              f"Target: {target_remaining:.0f}, Actual: {actual_remaining}")
                else:
                    # PASSIVE: Place limit order at best bid (for buys) and wait for fill
                    # First, check if previous passive order was filled
                    if active_passive_order_id is not None:
                        # Check if order was filled by checking if qty_remaining decreased
                        prev_remaining = self.qty_remaining
                        self.collect_agent_fills()  # Update fills
                        # If qty_remaining decreased, order was filled
                        if self.qty_remaining < prev_remaining:
                            # Order was filled, reset tracking
                            active_passive_order_id = None
                            active_passive_order_age = 0
                    
                    # Only place new passive order if we don't have one already resting
                    if active_passive_order_id is None:
                        if self.side == ob.Side.BUY:
                            passive_price = self.get_best_bid()[0] if self.get_best_bid()[0] else self.get_mid_price()
                        else:
                            passive_price = self.get_best_ask()[0] if self.get_best_ask()[0] else self.get_mid_price()
                        
                        # Track this order ID so we can cancel if unfilled after timeout
                        active_passive_order_id = self.order_id_counter
                        active_passive_order_age = 0  # Reset age for new order
                        
                        self.place_limit_order_passive(int(passive_price), target_qty)
                        if self.verbose and step % 10 == 0:
                            print(f"Step {step}/{num_steps}: PASSIVE (LMT @ {passive_price:.0f}) - Qty: {target_qty}, "
                                  f"Target: {target_remaining:.0f}, Actual: {actual_remaining}")
                    else:
                        # Order is still resting, wait for fill
                        if self.verbose and step % 10 == 0:
                            print(f"Step {step}/{num_steps}: PASSIVE (WAITING, age={active_passive_order_age}/{self.passive_order_timeout_steps}) - "
                                  f"Target: {target_remaining:.0f}, Actual: {actual_remaining}")
            
            # Advance time by step interval (same as TWAP)
            self.step(step_interval_ns)
            
            # Check exit conditions
            if self.qty_remaining <= 0:
                break
            if self.loader and not self.loader.HasMoreData():
                if self.verbose:
                    print("Out of market data!")
                break
        
        # Final cleanup: Cancel any remaining passive order
        if active_passive_order_id is not None:
            # Check one last time if order was filled
            prev_remaining = self.qty_remaining
            self.collect_agent_fills()
            if self.qty_remaining < prev_remaining:
                # Order was filled, no need to cancel
                pass
            else:
                # Order still unfilled, cancel it
                self.exchange.CancelAgentOrder(active_passive_order_id, 0)
                self.exchange.ProcessPendingAgentActions()
        
        # Execute any remaining quantity at the end with market order
        if self.qty_remaining > 0:
            if self.verbose:
                print(f"Executing remaining {self.qty_remaining} with market order")
            self.place_market_order(self.qty_remaining)
            self.step(self.step_duration_ns)  # Process the final market order
        
        # Calculate and return metrics
        result = self.calculate_metrics(arrival_price)
        result.strategy = "Almgren-Chriss"
        
        if self.verbose:
            self.print_results(result, "ALMGREN-CHRISS RESULTS")
        
        return result


def run_comparison(
    data_path: str,
    total_qty: int = 1000,
    total_time_ns: int = 60 * 60 * int(1e9),
    num_slices: int = 60,
    risk_aversion: float = 1e-6,
    side: str = "BUY",
    verbose: bool = True,
    output_file: Optional[str] = None,
) -> Dict:
    """Run both strategies and compare results."""
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON: TWAP vs Almgren-Chriss")
    print("="*80)
    print(f"Data: {data_path}")
    print(f"Total Qty: {total_qty}")
    print(f"Total Time: {total_time_ns / 1e9:.0f} seconds")
    print()
    
    # Run TWAP
    twap = TWAPExecutor(
        data_path=data_path,
        total_qty=total_qty,
        total_time_ns=total_time_ns,
        num_slices=num_slices,
        side=side,
        verbose=verbose,
    )
    twap_result = twap.execute()
    
    # Run Almgren-Chriss
    ac = AlmgrenChrissExecutor(
        data_path=data_path,
        total_qty=total_qty,
        total_time_ns=total_time_ns,
        risk_aversion=risk_aversion,
        side=side,
        verbose=verbose,
    )
    ac_result = ac.execute()
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'TWAP':>15} {'A-C':>15} {'Diff':>15}")
    print("-" * 70)
    print(f"{'VWAP':<25} {twap_result.vwap:>15.2f} {ac_result.vwap:>15.2f} {ac_result.vwap - twap_result.vwap:>15.2f}")
    print(f"{'Impl Shortfall':<25} {twap_result.implementation_shortfall:>15.2f} {ac_result.implementation_shortfall:>15.2f} {ac_result.implementation_shortfall - twap_result.implementation_shortfall:>15.2f}")
    print(f"{'Slippage (bps)':<25} {twap_result.slippage_bps:>15.2f} {ac_result.slippage_bps:>15.2f} {ac_result.slippage_bps - twap_result.slippage_bps:>15.2f}")
    print(f"{'Num Fills':<25} {len(twap_result.fills):>15} {len(ac_result.fills):>15}")
    print("="*80)
    
    results = {
        'twap': twap_result.to_dict(),
        'almgren_chriss': ac_result.to_dict(),
        'comparison': {
            'vwap_diff': ac_result.vwap - twap_result.vwap,
            'is_diff': ac_result.implementation_shortfall - twap_result.implementation_shortfall,
            'slippage_diff_bps': ac_result.slippage_bps - twap_result.slippage_bps,
        }
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run execution algorithm baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to L3 data CSV")
    parser.add_argument("--strategy", choices=["twap", "ac", "both"], default="both",
                        help="Strategy to run")
    parser.add_argument("--qty", type=int, default=1000, help="Total quantity to execute")
    parser.add_argument("--time", type=int, default=3600, help="Total time in seconds")
    parser.add_argument("--slices", type=int, default=60, help="Number of TWAP slices")
    parser.add_argument("--risk-aversion", type=float, default=1e-6, help="A-C risk aversion")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--side", choices=["BUY", "SELL"], default="BUY", help="Order side")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    
    args = parser.parse_args()
    
    total_time_ns = args.time * int(1e9)
    verbose = not args.quiet
    
    if args.strategy == "both":
        run_comparison(
            data_path=args.data,
            total_qty=args.qty,
            total_time_ns=total_time_ns,
            num_slices=args.slices,
            risk_aversion=args.risk_aversion,
            side=args.side,
            verbose=verbose,
            output_file=args.output,
        )
    elif args.strategy == "twap":
        executor = TWAPExecutor(
            data_path=args.data,
            total_qty=args.qty,
            total_time_ns=total_time_ns,
            num_slices=args.slices,
            side=args.side,
            verbose=verbose,
        )
        result = executor.execute()
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
    else:  # ac
        executor = AlmgrenChrissExecutor(
            data_path=args.data,
            total_qty=args.qty,
            total_time_ns=total_time_ns,
            risk_aversion=args.risk_aversion,
            side=args.side,
            verbose=verbose,
        )
        result = executor.execute()
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()

