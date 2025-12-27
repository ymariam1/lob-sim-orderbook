"""
Gymnasium environment for the Limit Order Book Simulator.

This environment wraps the C++ LOB simulator for reinforcement learning.
"""

from typing import Optional, Tuple, Dict, Any, Set
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import lob_sim as ob

# Import latency model
try:
    from src.py.latency import (
        MarketConditions, AgentLatencyModel, AgentLatencyProfile,
        create_agent_latency, MarketRegime
    )
except ImportError:
    from latency import (
        MarketConditions, AgentLatencyModel, AgentLatencyProfile,
        create_agent_latency, MarketRegime
    )


class LOBEnv(gym.Env):
    """
    Limit Order Book Trading Environment.
    
    The agent can:
    - Place limit buy/sell orders at various price levels
    - Place market buy/sell orders
    - Cancel existing orders (with latency!)
    - Hold (do nothing)
    
    State observation includes:
    - Current bid/ask levels (prices and quantities)
    - Agent's current position
    - Agent's P&L
    - Current market time
    - Number of active agent orders
    
    Rewards are based on P&L changes.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        max_levels: int = 10,
        agent_latency_ns: int = None,  # Deprecated: use agent_type instead
        agent_type: str = "institutional",  # "hft", "institutional", "retail", or custom
        volume_sensitivity: float = 0.1,  # How much volume affects latency
        max_position: int = 100,
        step_duration_ns: int = 10_000_000,  # 10ms per step
        warmup_duration_ns: int = 60_000_000_000,  # 60 seconds to build initial book
        timestamp_unit_ns: int = 1_000_000_000,  # CSV timestamp unit: 1e9 for seconds, 1000 for microseconds
        render_mode: Optional[str] = None,
        target_qty: int = 100,  # Target quantity to execute per episode
        execution_side: str = "SELL",  # "BUY" or "SELL" - the side the agent executes
        latency_seed: Optional[int] = None,  # Seed for reproducible latency
    ):
        """
        Initialize the LOB environment.
        
        Args:
            data_path: Path to historical data CSV. If None, uses synthetic seeding.
            max_levels: Number of bid/ask levels to include in observation
            agent_latency_ns: DEPRECATED - use agent_type instead
            agent_type: Agent latency profile:
                - "hft": ~0.5ms base (co-located, FPGA)
                - "institutional": ~10ms base (good infrastructure)
                - "retail": ~100ms base (consumer internet)
                - "5.0:0.5": Custom format "base_ms:sigma"
            volume_sensitivity: How much market volume affects latency (η)
            max_position: Maximum absolute position the agent can hold
            step_duration_ns: How much market time advances per step (default 10ms)
            warmup_duration_ns: How much data to pump at reset to build initial book
            timestamp_unit_ns: Conversion factor from CSV timestamp to nanoseconds:
                - 1 for nanoseconds
                - 1000 for microseconds (Tardis.dev format)
                - 1000000 for milliseconds
                - 1000000000 for seconds (default, datagen.py format)
            render_mode: Rendering mode ("human", "ansi", or None)
            target_qty: Target quantity to execute per episode (for IS reward)
            execution_side: "BUY" or "SELL" - which side the agent is executing
            latency_seed: Random seed for reproducible latency sampling
        """
        super().__init__()
        
        self.max_levels = max_levels
        self.max_position = max_position
        self.step_duration_ns = step_duration_ns
        self.warmup_duration_ns = warmup_duration_ns
        self.timestamp_unit_ns = timestamp_unit_ns
        self.render_mode = render_mode
        self.data_path = data_path
        self.target_qty = target_qty
        self.execution_side = ob.Side.BUY if execution_side.upper() == "BUY" else ob.Side.SELL
        self.agent_type = agent_type
        self.volume_sensitivity = volume_sensitivity
        self.latency_seed = latency_seed
        
        # Create latency components (separated environment vs agent state)
        # MarketConditions: Updated ONCE per step (shared by all agents)
        # AgentLatencyModel: Agent-specific jitter sampling
        if agent_latency_ns is not None:

            self._market_conditions = None
            self._agent_latency = None
            self._fixed_latency_ns = agent_latency_ns
        else:

            self._market_conditions = MarketConditions(
                volume_sensitivity=volume_sensitivity,
                seed=latency_seed,
            )
            self._agent_latency = create_agent_latency(
                agent_type=agent_type,
                seed=latency_seed + 1 if latency_seed else None,
            )
            self._fixed_latency_ns = None
        
        # Initialize orderbook, exchange, and loader
        self._orderbook: Optional[ob.Orderbook] = None
        self._exchange: Optional[ob.ExchangeSimulator] = None
        self._loader: Optional[ob.DataLoader] = None
        
        # Agent state
        self._position = 0  # Current inventory
        self._cash = 0.0    # Realized P&L
        self._order_id_counter = 1_000_000  # Start agent orders at high ID
        
        # Track active agent orders: Dict[order_id, {'side': Side, 'price': int, 'qty': int}]
        # This is critical for cancel functionality and latency risk mitigation
        self._active_orders: Dict[int, Dict[str, Any]] = {}
        
        # Episode tracking
        self._total_events_at_reset = 0
        
        # Implementation Shortfall tracking
        self._arrival_price = 0.0       # Mid price at episode start
        self._total_qty = 100           # Target quantity to execute (can be configured)
        self._executed_qty = 0          # Quantity executed so far
        self._execution_cost = 0.0      # Sum of (price * qty) for all executions
        self._prev_fills_count = 0      # Track fills processed
        
        # Latency tracking
        self._last_latency_ns = 0       # Last sampled latency
        self._latency_samples = []      # History of latencies
        
        # Action space:
        # 0 = Hold
        # 1-5 = Limit Buy at best bid - (0,1,2,3,4) ticks
        # 6-10 = Limit Sell at best ask + (0,1,2,3,4) ticks  
        # 11 = Market Buy
        # 12 = Market Sell
        # 13 = Cancel all agent orders
        self.action_space = spaces.Discrete(14)
        
        # Observation space: [bid_prices, bid_qtys, ask_prices, ask_qtys, position, cash, time, active_orders, progress]
        # Normalized to reasonable ranges
        obs_dim = max_levels * 4 + 5  # prices + qtys for bids/asks + position + cash + time + active_orders + progress
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Create fresh orderbook and exchange
        self._orderbook = ob.Orderbook()
        self._exchange = ob.ExchangeSimulator(self._orderbook)
        
        # Reset agent state
        self._position = 0
        self._cash = 0.0
        self._order_id_counter = 1_000_000
        self._active_orders.clear()
        
        # Reset Implementation Shortfall tracking
        self._executed_qty = 0
        self._execution_cost = 0.0
        self._prev_fills_count = 0
        
        # Reset latency tracking
        self._last_latency_ns = 0
        self._latency_samples = []
        if self._market_conditions is not None:
            self._market_conditions.reset()
        if self._agent_latency is not None:
            self._agent_latency.reset()
        
        # Initialize data loader or use synthetic seeding
        if self.data_path:
            self._loader = ob.DataLoader(self.data_path, self.timestamp_unit_ns)
            
            # Auto-warmup: Pump until we reach the first event's timestamp + warmup duration
            # This handles CSVs that don't start at timestamp 0
            first_event_time_ns = self._loader.PeekNextTimestampNs()
            target_time_ns = first_event_time_ns + self.warmup_duration_ns
            
            if self.render_mode:
                print(f"First event at {first_event_time_ns / 1e9:.1f}s, warming up to {target_time_ns / 1e9:.1f}s")
            
            events_pumped = self._loader.PumpToExchange(self._exchange, target_time_ns)
            
            if self.render_mode:
                print(f"Warmup: Pumped {events_pumped} events, book size: {self._orderbook.Size()}")
            
            self._total_events_at_reset = self._loader.GetTotalEventsProcessed()
        else:
            self._loader = None
            # Fall back to synthetic seeding if no data path provided
            self._seed_initial_book()
        
        # Capture arrival price (mid price after warmup)
        self._arrival_price = self._get_mid_price()
        self._total_qty = self.target_qty  # Reset target quantity
        
        obs = self._get_observation()
        info = self._get_info()
        info["arrival_price"] = self._arrival_price
        
        return obs, info
    
    def _seed_initial_book(self):
        """Add synthetic initial orders to create a realistic book (fallback mode)."""
        base_price = 10000  # e.g., $100.00 in cents
        timestamp = 0
        
        # Add bid orders
        for i, (offset, qty) in enumerate([(0, 50), (1, 100), (2, 150), (3, 200), (4, 250)]):
            order = ob.Order(
                i + 1,              # orderId
                ob.Side.BUY,        # side
                base_price - offset,# price
                qty,                # quantity
                timestamp,          # timestamp
                ob.LIMIT,           # orderType
                ob.GTC              # timeInForce
            )
            self._exchange.ProcessHistoricalEvent(order)
        
        # Add ask orders
        for i, (offset, qty) in enumerate([(1, 50), (2, 100), (3, 150), (4, 200), (5, 250)]):
            order = ob.Order(
                i + 100,            # orderId
                ob.Side.SELL,       # side
                base_price + offset,# price
                qty,                # quantity
                timestamp,          # timestamp
                ob.LIMIT,           # orderType
                ob.GTC              # timeInForce
            )
            self._exchange.ProcessHistoricalEvent(order)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: The action to take (0-13)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # 1. Execute Agent Action (Place Order / Cancel)
        self._execute_action(action)
        
        # 2. VITAL: Pump the Market Data Forward
        # This advances market time and processes historical events
        events_processed = 0
        if self._loader:
            events_processed = self._loader.PumpToExchange(self._exchange, self.step_duration_ns)
        
        # 3. Update market conditions ONCE per step (affects ALL agents equally)
        # This updates regime (Calm/Stressed) and global congestion
        if self._market_conditions is not None:
            volume = events_processed * 10  # Scale events as volume proxy
            self._market_conditions.update(volume)
        
        # 3. Process pending agent actions
        self._exchange.ProcessPendingAgentActions()
        
        # 4. Compute Implementation Shortfall reward from fills
        step_reward = self._process_fills_for_reward()
        
        # 5. Check termination conditions
        terminated = False
        truncated = False
        
        # Episode ends when target quantity is fully executed
        if self._executed_qty >= self._total_qty:
            terminated = True
        
        # Episode truncated when we run out of data
        if self._loader and not self._loader.HasMoreData():
            truncated = True
        
        # 6. Add terminal penalty for incomplete execution
        if terminated or truncated:
            step_reward += self._compute_terminal_penalty()
        
        obs = self._get_observation()
        info = self._get_info()
        info["events_processed"] = events_processed
        info["executed_qty"] = self._executed_qty
        info["remaining_qty"] = self._total_qty - self._executed_qty
        
        # Compute VWAP if we have executions
        if self._executed_qty > 0:
            info["vwap"] = self._execution_cost / self._executed_qty
            info["slippage_bps"] = abs(info["vwap"] - self._arrival_price) / self._arrival_price * 10000
        
        return obs, step_reward, terminated, truncated, info
    
    def _execute_action(self, action: int):
        """Execute the given action."""
        if action == 0:
            # Hold - do nothing
            return
        
        book_state = self._orderbook.GetOrderInfos()
        bids = book_state.GetBids()
        asks = book_state.GetAsks()
        
        current_time = self._exchange.GetCurrentTime()
        
        if action <= 5:  # Limit Buy (1-5)
            if not bids:
                return  # No bids to reference
            best_bid = bids[0].price
            offset = action - 1
            price = best_bid - offset
            self._place_limit_order(ob.Side.BUY, price, 10, current_time)
            
        elif action <= 10:  # Limit Sell (6-10)
            if not asks:
                return  # No asks to reference
            best_ask = asks[0].price
            offset = action - 6
            price = best_ask + offset
            self._place_limit_order(ob.Side.SELL, price, 10, current_time)
            
        elif action == 11:  # Market Buy
            self._place_market_order(ob.Side.BUY, 10, current_time)
            
        elif action == 12:  # Market Sell
            self._place_market_order(ob.Side.SELL, 10, current_time)
            
        elif action == 13:  # Cancel all agent orders
            self._cancel_all_orders()
    
    def _place_limit_order(self, side: ob.Side, price: int, qty: int, timestamp: int):
        """Place a limit order through the exchange with latency."""
        if abs(self._position) >= self.max_position:
            return  # Position limit reached
        
        order_id = self._order_id_counter
        
        order = ob.Order(
            order_id,
            side,
            price,
            qty,
            timestamp,
            ob.LIMIT,
            ob.GTC
        )
        
        # Track the order BEFORE sending (we know it's in flight)
        self._active_orders[order_id] = {
            'side': side,
            'price': price,
            'qty': qty,
            'timestamp': timestamp,
            'status': 'pending'  # Will become 'active' after latency
        }
        
        self._order_id_counter += 1
        latency = self._get_latency()
        self._exchange.PlaceAgentOrder(order, latency)
    
    def _place_market_order(self, side: ob.Side, qty: int, timestamp: int):
        """Place a market order through the exchange with latency."""
        if abs(self._position) >= self.max_position:
            return  # Position limit reached
        
        order = ob.Order(
            self._order_id_counter,
            side,
            0,  # Price ignored for market orders
            qty,
            timestamp,
            ob.MARKET,
            ob.IOC
        )
        # Market orders are not tracked since they execute immediately (after latency)
        # and don't rest on the book
        self._order_id_counter += 1
        latency = self._get_latency()
        self._exchange.PlaceAgentOrder(order, latency)
    
    def _get_latency(self) -> int:
        """
        Get current latency in nanoseconds.
        
        Architecture:
        - MarketConditions provides regime + global congestion (updated once per step)
        - AgentLatencyModel samples jitter based on agent type and current regime
        
        Total Latency = Base + Global_Congestion + Jitter
        
        Key insight: HFT has low variance even in stress, Institutional spikes.
        """
        if self._agent_latency is not None and self._market_conditions is not None:
            # Dynamic latency: Agent samples given current market conditions
            latency = self._agent_latency.sample(self._market_conditions)
            self._last_latency_ns = latency
            self._latency_samples.append(latency)
            return latency
        else:
            # Fixed latency (legacy mode)
            return self._fixed_latency_ns
    
    def _cancel_all_orders(self):
        """Cancel all active agent orders with proper latency simulation."""
        latency = self._get_latency()
        # Cancel each active order through the exchange (with latency!)
        for order_id in list(self._active_orders.keys()):
            # Use the exchange's cancel method which applies latency
            self._exchange.CancelAgentOrder(order_id, latency)
        
        # Clear local tracking (orders are "cancel pending" now)
        # In a more sophisticated implementation, you'd track cancel status
        self._active_orders.clear()
    
    def _cancel_order(self, order_id: int):
        """Cancel a single order with latency simulation."""
        if order_id in self._active_orders:
            latency = self._get_latency()
            self._exchange.CancelAgentOrder(order_id, latency)
            del self._active_orders[order_id]
    
    def _get_observation(self) -> np.ndarray:
        """Build the observation vector."""
        book_state = self._orderbook.GetOrderInfos()
        bids = book_state.GetBids()
        asks = book_state.GetAsks()
        
        # Pad/truncate to max_levels
        bid_prices = np.zeros(self.max_levels, dtype=np.float32)
        bid_qtys = np.zeros(self.max_levels, dtype=np.float32)
        ask_prices = np.zeros(self.max_levels, dtype=np.float32)
        ask_qtys = np.zeros(self.max_levels, dtype=np.float32)
        
        for i, level in enumerate(bids[:self.max_levels]):
            bid_prices[i] = level.price
            bid_qtys[i] = level.quantity
            
        for i, level in enumerate(asks[:self.max_levels]):
            ask_prices[i] = level.price
            ask_qtys[i] = level.quantity
        
        # Normalize (simple scaling - could be improved)
        mid_price = (bid_prices[0] + ask_prices[0]) / 2 if bid_prices[0] > 0 and ask_prices[0] > 0 else 10000
        
        # Execution progress (0 = not started, 1 = complete)
        progress = self._executed_qty / self._total_qty if self._total_qty > 0 else 0
        
        obs = np.concatenate([
            (bid_prices - mid_price) / 100,  # Relative prices
            bid_qtys / 1000,                  # Scaled quantities
            (ask_prices - mid_price) / 100,
            ask_qtys / 1000,
            [self._position / self.max_position],  # Normalized position
            [self._cash / 10000],                  # Scaled cash
            [self._exchange.GetCurrentTime() / 1e9],  # Time in seconds
            [len(self._active_orders) / 10],  # Normalized active order count
            [progress],                        # Execution progress (0-1)
        ])
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for debugging."""
        info = {
            "position": self._position,
            "cash": self._cash,
            "pnl": self._calculate_pnl(),
            "book_size": self._orderbook.Size(),
            "current_time": self._exchange.GetCurrentTime(),
            "active_orders": len(self._active_orders),
            "pending_actions": self._exchange.GetPendingActionCount(),
            # Implementation Shortfall metrics
            "arrival_price": self._arrival_price,
            "target_qty": self._total_qty,
            "executed_qty": self._executed_qty,
            "mid_price": self._get_mid_price(),
            # Latency info
            "last_latency_ns": self._last_latency_ns,
            "last_latency_ms": self._last_latency_ns / 1e6,
        }
        
        # Add market conditions and latency statistics
        if self._market_conditions is not None:
            info["market_regime"] = self._market_conditions.regime.value
            info["global_congestion_ms"] = self._market_conditions.global_congestion_ns / 1e6
            info["current_volume"] = self._market_conditions.current_volume
        
        if self._agent_latency is not None and self._latency_samples:
            samples = np.array(self._latency_samples)
            info["latency_mean_ms"] = np.mean(samples) / 1e6
            info["latency_p99_ms"] = np.percentile(samples, 99) / 1e6
        
        # Add loader stats if available
        if self._loader:
            info["total_events_processed"] = self._loader.GetTotalEventsProcessed()
            info["has_more_data"] = self._loader.HasMoreData()
        
        return info
    
    def _get_mid_price(self) -> float:
        """Get current mid price."""
        book_state = self._orderbook.GetOrderInfos()
        bids = book_state.GetBids()
        asks = book_state.GetAsks()
        
        if bids and asks:
            return (bids[0].price + asks[0].price) / 2
        elif bids:
            return bids[0].price
        elif asks:
            return asks[0].price
        return 10000  # Fallback
    
    def _calculate_pnl(self) -> float:
        """Calculate current P&L (cash + mark-to-market inventory)."""
        mid_price = self._get_mid_price()
        return self._cash + self._position * mid_price
    
    def _process_fills_for_reward(self) -> float:
        """
        Process fills from the matching engine and compute step reward.
        
        Implementation Shortfall reward:
        - For sells: reward = -(arrival_price - execution_price) * qty / arrival_price
        - For buys: reward = -(execution_price - arrival_price) * qty / arrival_price
        
        Negative cost = positive reward (we want to minimize cost)
        """
        fills = self._exchange.GetAgentFills()
        step_reward = 0.0
        
        for trade in fills:
            # Get the trade info for our side
            if self.execution_side == ob.Side.BUY:
                trade_info = trade.GetBidTrade()
            else:
                trade_info = trade.GetAskTrade()
            
            price = trade_info.price
            qty = trade_info.quantity
            
            # Update tracking
            self._executed_qty += qty
            self._execution_cost += price * qty
            
            # Compute step reward (normalized implementation shortfall)
            if self._arrival_price > 0:
                if self.execution_side == ob.Side.SELL:
                    # For sells: higher price = better = positive reward
                    slippage = (self._arrival_price - price) / self._arrival_price
                else:
                    # For buys: lower price = better = positive reward
                    slippage = (price - self._arrival_price) / self._arrival_price
                
                # Negative slippage is good (we beat arrival price)
                step_reward -= slippage * qty
        
        self._exchange.ClearAgentFills()
        return step_reward
    
    def _compute_terminal_penalty(self) -> float:
        """
        Compute penalty for unexecuted quantity at episode end.
        
        This encourages the agent to complete execution rather than
        just holding to avoid slippage.
        """
        remaining = self._total_qty - self._executed_qty
        if remaining > 0:
            # Large penalty for incomplete execution
            return -10.0 * (remaining / self._total_qty)
        return 0.0
    
    def render(self):
        """Render the current state."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            book_state = self._orderbook.GetOrderInfos()
            bids = book_state.GetBids()
            asks = book_state.GetAsks()
            
            print("\n" + "=" * 60)
            print(f"Time: {self._exchange.GetCurrentTime()} ns")
            print(f"Position: {self._position} | Cash: ${self._cash:.2f}")
            print(f"P&L: ${self._calculate_pnl():.2f}")
            print(f"Active Orders: {len(self._active_orders)} | Pending: {self._exchange.GetPendingActionCount()}")
            
            if self._loader:
                print(f"Events Processed: {self._loader.GetTotalEventsProcessed()} | Has More: {self._loader.HasMoreData()}")
            
            print("-" * 60)
            print("  ASKS:")
            for level in reversed(asks[:5]):
                print(f"    {level.price:>10} | {level.quantity:>6}")
            print("  " + "-" * 20)
            print("  BIDS:")
            for level in bids[:5]:
                print(f"    {level.price:>10} | {level.quantity:>6}")
            print("=" * 60)
    
    def close(self):
        """Clean up resources."""
        self._orderbook = None
        self._exchange = None
        self._loader = None
        self._active_orders.clear()


# Register the environment with Gymnasium
try:
    gym.register(
        id="LOB-v0",
        entry_point="gym:LOBEnv",
    )
except Exception:
    pass  # Already registered


if __name__ == "__main__":
    import os
    
    script_dir = os.path.dirname(__file__)
    
    # Check for Tardis L2 data first (microseconds), then synthetic data (seconds)
    tardis_path = os.path.join(script_dir, "../../data/coinbase_btc_l2.csv")
    synth_path = os.path.join(script_dir, "../../data/simulation_data.csv")
    
    if os.path.exists(tardis_path):
        print(f"Using Tardis L2 data from: {tardis_path}")
        env = LOBEnv(
            data_path=tardis_path,
            render_mode="human",
            timestamp_unit_ns=1000,  # Tardis uses microseconds
            warmup_duration_ns=int(10 * 1e9),   # 10 seconds warmup
            step_duration_ns=int(1 * 1e9),      # 1 second per step
        )
    elif os.path.exists(synth_path):
        print(f"Using synthetic data from: {synth_path}")
        env = LOBEnv(
            data_path=synth_path,
            render_mode="human",
            timestamp_unit_ns=int(1e9),  # datagen.py uses seconds
            warmup_duration_ns=int(1000 * 1e9),  # 1000 seconds warmup
            step_duration_ns=int(100 * 1e9),     # 100 seconds per step
        )
    else:
        print("No data file found, using synthetic seeding")
        env = LOBEnv(render_mode="human")
    
    obs, info = env.reset()
    
    print("Initial observation shape:", obs.shape)
    print("Initial info:", info)
    env.render()
    
    # Take a few random actions
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"\nStep {i+1}: Action={action}, Reward={reward:.4f}")
        
        if i % 3 == 0:  # Render every 3 steps
            env.render()
        
        if trunc:
            print("Episode truncated - no more data!")
            break
    
    env.close()
