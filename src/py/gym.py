"""
Gymnasium environment for the Limit Order Book Simulator.

This environment wraps the C++ LOB simulator for reinforcement learning.
"""

from typing import Optional, Tuple, Dict, Any, Set
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("Please install gymnasium: pip install gymnasium")

try:
    import lob_sim as ob
except ImportError:
    raise ImportError(
        "lob_sim module not found. Please build it first:\n"
        "  cd /path/to/lob-sim-orderbook\n"
        "  pip install ."
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
        agent_latency_ns: int = 1_000_000,  # 1ms default
        max_position: int = 100,
        step_duration_ns: int = 10_000_000,  # 10ms per step
        warmup_duration_ns: int = 60_000_000_000,  # 60 seconds to build initial book
        timestamp_unit_ns: int = 1_000_000_000,  # CSV timestamp unit: 1e9 for seconds, 1000 for microseconds
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the LOB environment.
        
        Args:
            data_path: Path to historical data CSV. If None, uses synthetic seeding.
            max_levels: Number of bid/ask levels to include in observation
            agent_latency_ns: Simulated network latency in nanoseconds
            max_position: Maximum absolute position the agent can hold
            step_duration_ns: How much market time advances per step (default 10ms)
            warmup_duration_ns: How much data to pump at reset to build initial book
            timestamp_unit_ns: Conversion factor from CSV timestamp to nanoseconds:
                - 1 for nanoseconds
                - 1000 for microseconds (Tardis.dev format)
                - 1000000 for milliseconds
                - 1000000000 for seconds (default, datagen.py format)
            render_mode: Rendering mode ("human", "ansi", or None)
        """
        super().__init__()
        
        self.max_levels = max_levels
        self.agent_latency_ns = agent_latency_ns
        self.max_position = max_position
        self.step_duration_ns = step_duration_ns
        self.warmup_duration_ns = warmup_duration_ns
        self.timestamp_unit_ns = timestamp_unit_ns
        self.render_mode = render_mode
        self.data_path = data_path
        
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
        
        # Action space:
        # 0 = Hold
        # 1-5 = Limit Buy at best bid - (0,1,2,3,4) ticks
        # 6-10 = Limit Sell at best ask + (0,1,2,3,4) ticks  
        # 11 = Market Buy
        # 12 = Market Sell
        # 13 = Cancel all agent orders
        self.action_space = spaces.Discrete(14)
        
        # Observation space: [bid_prices, bid_qtys, ask_prices, ask_qtys, position, cash, time, active_orders]
        # Normalized to reasonable ranges
        obs_dim = max_levels * 4 + 4  # prices + qtys for bids/asks + position + cash + time + active_orders
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
        
        obs = self._get_observation()
        info = self._get_info()
        
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
        prev_pnl = self._calculate_pnl()
        
        # 1. Execute Agent Action (Place Order / Cancel)
        self._execute_action(action)
        
        # 2. VITAL: Pump the Market Data Forward
        # This advances market time and processes historical events
        events_processed = 0
        if self._loader:
            events_processed = self._loader.PumpToExchange(self._exchange, self.step_duration_ns)
        
        # 3. Check if episode is over (no more data)
        terminated = False
        truncated = False
        if self._loader and not self._loader.HasMoreData():
            truncated = True  # Episode ends when we run out of data
        
        # 4. Calculate Reward
        current_pnl = self._calculate_pnl()
        reward = current_pnl - prev_pnl
        
        obs = self._get_observation()
        info = self._get_info()
        info["events_processed"] = events_processed
        
        return obs, reward, terminated, truncated, info
    
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
        self._exchange.PlaceAgentOrder(order, self.agent_latency_ns)
    
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
        self._exchange.PlaceAgentOrder(order, self.agent_latency_ns)
    
    def _cancel_all_orders(self):
        """Cancel all active agent orders with proper latency simulation."""
        # Cancel each active order through the exchange (with latency!)
        for order_id in list(self._active_orders.keys()):
            # Use the exchange's cancel method which applies latency
            self._exchange.CancelAgentOrder(order_id, self.agent_latency_ns)
        
        # Clear local tracking (orders are "cancel pending" now)
        # In a more sophisticated implementation, you'd track cancel status
        self._active_orders.clear()
    
    def _cancel_order(self, order_id: int):
        """Cancel a single order with latency simulation."""
        if order_id in self._active_orders:
            self._exchange.CancelAgentOrder(order_id, self.agent_latency_ns)
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
        
        obs = np.concatenate([
            (bid_prices - mid_price) / 100,  # Relative prices
            bid_qtys / 1000,                  # Scaled quantities
            (ask_prices - mid_price) / 100,
            ask_qtys / 1000,
            [self._position / self.max_position],  # Normalized position
            [self._cash / 10000],                  # Scaled cash
            [self._exchange.GetCurrentTime() / 1e9],  # Time in seconds
            [len(self._active_orders) / 10],  # Normalized active order count
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
        }
        
        # Add loader stats if available
        if self._loader:
            info["total_events_processed"] = self._loader.GetTotalEventsProcessed()
            info["has_more_data"] = self._loader.HasMoreData()
        
        return info
    
    def _calculate_pnl(self) -> float:
        """Calculate current P&L (cash + mark-to-market inventory)."""
        book_state = self._orderbook.GetOrderInfos()
        bids = book_state.GetBids()
        asks = book_state.GetAsks()
        
        if bids and asks:
            mid_price = (bids[0].price + asks[0].price) / 2
        else:
            mid_price = 10000  # Fallback
        
        return self._cash + self._position * mid_price
    
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
