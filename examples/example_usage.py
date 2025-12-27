#!/usr/bin/env python3
"""
Example usage of the lob_sim Python bindings.

Demonstrates:
1. Creating orderbook and exchange simulator
2. Processing historical market data
3. Agent order placement with latency
4. Fill tracking for agent orders (aggressive + passive)
5. Cancel functionality

Build first:
    pip install .
"""

try:
    import lob_sim as ob
except ImportError:
    print("ERROR: lob_sim module not found.")
    print("Please build it first:")
    print("  cd /path/to/lob-sim-orderbook")
    print("  pip install .")
    exit(1)


def print_book(book: ob.Orderbook, title: str = "ORDER BOOK"):
    """Print order book state."""
    state = book.GetOrderInfos()
    asks = state.GetAsks()
    bids = state.GetBids()
    
    print(f"\n{title}")
    print("-" * 40)
    print("  ASKS:")
    for level in reversed(asks[:5]):
        print(f"    {level.price:>10} | {level.quantity:>6}")
    print("  " + "-" * 20)
    print("  BIDS:")
    for level in bids[:5]:
        print(f"    {level.price:>10} | {level.quantity:>6}")
    print("-" * 40)


def main():
    print("=" * 60)
    print("LOB Simulator - Python Bindings Example")
    print("=" * 60)
    
    # =========================================================================
    # 1. Create orderbook and exchange
    # =========================================================================
    print("\n1. Creating orderbook and exchange...")
    book = ob.Orderbook()
    exchange = ob.ExchangeSimulator(book)
    print(f"   Book size: {book.Size()}")
    
    # =========================================================================
    # 2. Add historical orders to build the book
    # =========================================================================
    print("\n2. Adding historical orders...")
    
    # Add bid orders
    for i in range(5):
        order = ob.Order(
            i + 1,           # orderId
            ob.Side.BUY,     # side
            10000 - i,       # price (10000, 9999, 9998, ...)
            100 * (i + 1),   # quantity
            1_000_000,       # timestamp
            ob.LIMIT,        # orderType
            ob.GTC           # timeInForce
        )
        exchange.ProcessHistoricalEvent(order)
    
    # Add ask orders
    for i in range(5):
        order = ob.Order(
            i + 100,         # orderId
            ob.Side.SELL,    # side
            10001 + i,       # price (10001, 10002, 10003, ...)
            100 * (i + 1),   # quantity
            1_000_000,       # timestamp
            ob.LIMIT,        # orderType
            ob.GTC           # timeInForce
        )
        exchange.ProcessHistoricalEvent(order)
    
    print(f"   Book size: {book.Size()}")
    print_book(book, "Initial Book State")
    
    # =========================================================================
    # 3. Agent places AGGRESSIVE order (crosses spread)
    # =========================================================================
    print("\n3. Agent places aggressive BUY (crosses spread)...")
    
    # BUY @ 10001 will match the best ask immediately
    aggressive_order = ob.Order(
        9999,               # orderId
        ob.Side.BUY,        # side  
        10001,              # price (at best ask - will match!)
        50,                 # quantity
        2_000_000,          # timestamp
        ob.LIMIT,           # orderType
        ob.GTC              # timeInForce
    )
    
    exchange.PlaceAgentOrder(aggressive_order, 1_000_000)  # 1ms latency
    print(f"   Order sent (pending: {exchange.GetPendingActionCount()})")
    
    # Advance time to process the order
    exchange.SetCurrentTime(5_000_000)  # Jump to 5ms
    exchange.ProcessPendingAgentActions()
    
    # Check fills
    print(f"   Agent fills: {exchange.GetAgentFillCount()}")
    for fill in exchange.GetAgentFills():
        bid = fill.GetBidTrade()
        ask = fill.GetAskTrade()
        print(f"   -> Matched: bid#{bid.orderId} bought {bid.quantity} @ {bid.price} from ask#{ask.orderId}")
    exchange.ClearAgentFills()
    
    print_book(book, "After Aggressive Order")
    
    # =========================================================================
    # 4. Agent places PASSIVE order (rests on book)
    # =========================================================================
    print("\n4. Agent places passive BUY (rests on book)...")
    
    # BUY @ 9990 (below best bid) - will rest on book
    passive_order = ob.Order(
        8888,               # orderId
        ob.Side.BUY,        # side
        9990,               # price (below best bid - will rest)
        200,                # quantity
        10_000_000,         # timestamp
        ob.LIMIT,           # orderType
        ob.GTC              # timeInForce
    )
    
    exchange.PlaceAgentOrder(passive_order, 1_000_000)  # 1ms latency
    exchange.SetCurrentTime(15_000_000)
    exchange.ProcessPendingAgentActions()
    
    print(f"   Agent fills (should be 0): {exchange.GetAgentFillCount()}")
    print_book(book, "After Passive Order (should see 200 @ 9990)")
    
    # =========================================================================
    # 5. Historical SELL comes in and fills our passive order
    # =========================================================================
    print("\n5. Historical SELL arrives and fills our passive order...")
    
    # A market participant sells at 9990 - matches our resting bid!
    historical_sell = ob.Order(
        7777,               # orderId
        ob.Side.SELL,       # side
        9990,               # price (matches our passive bid)
        100,                # quantity (partial fill)
        20_000_000,         # timestamp
        ob.LIMIT,           # orderType
        ob.GTC              # timeInForce
    )
    exchange.ProcessHistoricalEvent(historical_sell)
    
    # Now check for fills - our passive order got filled!
    print(f"   Agent fills: {exchange.GetAgentFillCount()}")
    for fill in exchange.GetAgentFills():
        bid = fill.GetBidTrade()
        ask = fill.GetAskTrade()
        print(f"   -> Passive fill: bid#{bid.orderId} bought {bid.quantity} @ {bid.price}")
    exchange.ClearAgentFills()
    
    print_book(book, "After Passive Fill (100 filled, 100 remaining @ 9990)")
    
    # =========================================================================
    # 6. Cancel remaining order
    # =========================================================================
    print("\n6. Canceling remaining passive order...")
    
    exchange.CancelAgentOrder(8888, 1_000_000)  # 1ms latency
    print(f"   Pending actions: {exchange.GetPendingActionCount()}")
    
    exchange.SetCurrentTime(25_000_000)
    exchange.ProcessPendingAgentActions()
    
    print_book(book, "After Cancel (9990 level should be gone)")
    
    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final book size: {book.Size()}")
    print(f"Current time: {exchange.GetCurrentTime()} ns")
    print("=" * 60)


if __name__ == "__main__":
    main()
