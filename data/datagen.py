import pandas as pd 
import numpy as np
import csv

def generate_market_data(num_events=100000, start_price=10000):
    print(f"Generating {num_events} events...")
    
    # Order types and time-in-force options
    ORDER_TYPES = ["MARKET", "LIMIT"]
    TIME_IN_FORCE = ["GTC", "FOK", "IOC", "GFD"]
    
    # 1. Simulate a Price Path (Geometric Brownian Motion)
    # This represents the "Fair Value" moving over time
    returns = np.random.normal(0, 0.0001, num_events)
    price_path = start_price * np.exp(np.cumsum(returns))
    
    events = []
    active_orders = {} # Track active orders to create valid CANCELs
    order_id_counter = 1
    timestamp = 1000 # Start time in seconds (will increment by seconds)
    
    for i in range(num_events):
        mid_price = int(price_path[i])
        timestamp += np.random.randint(1, 10) # Time moves forward 1-10 seconds
        
        # Decide Event Type: 80% ADD, 15% CANCEL, 5% MODIFY
        rand_val = np.random.random()
        
        if rand_val < 0.80: 
            # --- CREATE ADD ORDER ---
            side_str = "BUY" if np.random.random() > 0.5 else "SELL"
            
            # Order type distribution: 85% LIMIT, 15% MARKET
            order_type_rand = np.random.random()
            if order_type_rand < 0.85:
                order_type = "LIMIT"
            else:
                order_type = "MARKET"
            
            # Time-in-force distribution:
            # - LIMIT orders: 70% GTC, 15% GFD, 10% IOC, 5% FOK
            # - MARKET orders: 50% IOC, 30% FOK, 20% GTC (market GTC is unusual but possible)
            tif_rand = np.random.random()
            if order_type == "LIMIT":
                if tif_rand < 0.70:
                    time_in_force = "GTC"
                elif tif_rand < 0.85:
                    time_in_force = "GFD"
                elif tif_rand < 0.95:
                    time_in_force = "IOC"
                else:
                    time_in_force = "FOK"
            else:  # MARKET
                if tif_rand < 0.50:
                    time_in_force = "IOC"
                elif tif_rand < 0.80:
                    time_in_force = "FOK"
                else:
                    time_in_force = "GTC"
            
            # Spread logic: Place orders near the mid price
            # Market orders don't need a price (but we'll set it to mid_price for CSV)
            if order_type == "MARKET":
                price = mid_price  # Market orders use mid price as placeholder
            else:
                offset = int(np.random.exponential(scale=5)) # Most orders close to BBO
                offset = max(1, offset) # Minimum 1 tick spread
                
                if side_str == "BUY":
                    price = mid_price - offset
                else:
                    price = mid_price + offset
                
            qty = np.random.randint(1, 10) * 10 # 10, 20, ... 100
            
            events.append([timestamp, "ADD", order_id_counter, side_str, price, qty, order_type, time_in_force])
            
            # Track it so we can cancel it later
            active_orders[order_id_counter] = {
                'side': side_str, 
                'price': price, 
                'qty': qty,
                'order_type': order_type,
                'time_in_force': time_in_force
            }
            order_id_counter += 1
            
        elif rand_val < 0.95 and active_orders:
            # --- CREATE CANCEL ORDER ---
            # Pick a random active order to kill
            oid_to_cancel = np.random.choice(list(active_orders.keys()))
            order_info = active_orders.pop(oid_to_cancel)
            
            # Cancel events need order_id, side, price, qty (0s for price/qty), order_type, time_in_force
            events.append([
                timestamp, "CANCEL", oid_to_cancel, order_info['side'], 
                0, 0, order_info['order_type'], order_info['time_in_force']
            ])
            
        elif active_orders:
            # --- CREATE MODIFY ORDER ---
            # Pick order to modify
            oid_to_mod = np.random.choice(list(active_orders.keys()))
            order_info = active_orders[oid_to_mod]
            
            # New quantity (Resize down usually to keep priority, or up)
            new_qty = np.random.randint(1, 5) * 10
            
            events.append([
                timestamp, "MODIFY", oid_to_mod, order_info['side'], 
                order_info['price'], new_qty, order_info['order_type'], order_info['time_in_force']
            ])

    # Convert to DataFrame
    df = pd.DataFrame(events, columns=[
        "timestamp", "type", "order_id", "side", "price", "qty", "order_type", "time_in_force"
    ])
    
    # Save to CSV
    filename = "simulation_data.csv"
    df.to_csv(filename, index=False)
    print(f"Successfully saved {filename} with {len(df)} rows.")

if __name__ == "__main__":
    generate_market_data()