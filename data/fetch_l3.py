#!/usr/bin/env python3
"""
Fetch L3 (Market-by-Order) data from Blockchain.com via Tardis.dev HTTP API.

Blockchain.com is one of the few exchanges that provides FREE L3 data on Tardis.
L3 data includes individual order IDs, which is REQUIRED for queue position simulation.

The HTTP API returns 1 minute of data per request, so we iterate through all
1440 minutes of a full trading day.

Usage:
    python fetch_l3.py                          # Fetch full day
    python fetch_l3.py --hours 1                # Fetch 1 hour only
    python fetch_l3.py --date 2023-04-01        # Different date
    python fetch_l3.py --convert-only           # Just convert existing raw data

Output:
    - raw_l3/blockchain_l3_YYYY-MM-DD.jsonl     # Raw data
    - blockchain_l3_YYYY-MM-DD.csv              # Converted CSV
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def fetch_l3_minute(
    date: str,
    symbol: str,
    offset: int,
    max_retries: int = 3
) -> Optional[str]:
    """
    Fetch 1 minute of L3 data from Tardis HTTP API.
    
    Args:
        date: Date in YYYY-MM-DD format
        symbol: Trading symbol (e.g., 'BTC-USD')
        offset: Minute offset from midnight (0-1439)
        max_retries: Number of retries on failure
        
    Returns:
        Raw text data or None on failure
    """
    base_url = 'https://api.tardis.dev/v1/data-feeds/blockchain-com'
    filters = json.dumps([{"channel": "l3", "symbols": [symbol]}])
    url = f'{base_url}?from={date}&filters={filters}&offset={offset}'
    
    headers = {'Accept-Encoding': 'gzip'}
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 429:  # Rate limited
                wait_time = 2 ** attempt
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    Error {resp.status_code}: {resp.text[:100]}")
                return None
        except requests.exceptions.Timeout:
            print(f"    Timeout, retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    return None


def fetch_full_day(
    date: str = "2023-03-01",
    symbol: str = "BTC-USD",
    hours: int = 24,
    output_dir: str = "raw_l3",
    delay: float = 0.05
) -> str:
    """
    Fetch a full day of L3 data, one minute at a time.
    
    Args:
        date: Date to fetch (YYYY-MM-DD). First of month = free!
        symbol: Trading symbol
        hours: Number of hours to fetch (1-24)
        output_dir: Where to save raw data
        delay: Delay between requests (seconds)
        
    Returns:
        Path to output file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"blockchain_l3_{date}.jsonl")
    
    total_minutes = hours * 60
    
    print(f"Fetching {hours}h of L3 data for {symbol} on {date}")
    print(f"Total requests: {total_minutes}")
    print(f"Estimated time: {total_minutes * (delay + 0.3) / 60:.1f} minutes")
    print()
    
    # Resume from where we left off if file exists
    existing_lines = 0
    start_offset = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_lines = sum(1 for _ in f)
        # Estimate which minute we're on (rough)
        # We'll just append, duplicates are fine for raw data
        print(f"Resuming... ({existing_lines} existing lines)")
    
    total_events = existing_lines
    start_time = time.time()
    
    with open(output_file, 'a') as f:
        for offset in range(start_offset, total_minutes):
            # Progress
            if offset % 60 == 0:
                elapsed = time.time() - start_time
                remaining = (total_minutes - offset) * (elapsed / max(offset, 1))
                print(f"Hour {offset // 60}: {total_events:,} events | "
                      f"ETA: {remaining/60:.1f}min")
            
            # Fetch
            data = fetch_l3_minute(date, symbol, offset)
            if data:
                lines = [l for l in data.strip().split('\n') if l]
                for line in lines:
                    f.write(line + '\n')
                total_events += len(lines)
            
            time.sleep(delay)
    
    print(f"\nDone! Total events: {total_events:,}")
    print(f"Saved to: {output_file}")
    
    return output_file


def convert_l3_to_csv(
    input_file: str,
    output_file: Optional[str] = None,
    price_multiplier: float = 100.0,    # Convert to cents
    qty_multiplier: float = 100000.0,   # Scale up BTC quantities
) -> pd.DataFrame:
    """
    Convert raw Blockchain.com L3 data to our CSV format.
    
    Blockchain.com L3 format:
        {"event":"updated","channel":"l3","symbol":"BTC-USD",
         "bids":[{"id":"12345","px":23000.0,"qty":0.5}],
         "asks":[...]}
    
    Key insight:
        - qty > 0: ADD (new order or quantity update)
        - qty = 0: CANCEL (order removed)
        
    Our format:
        timestamp, type, order_id, side, price, qty, order_type, time_in_force
    """
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = base.replace('raw_l3/', '') + '.csv'
    
    print(f"Converting {input_file}...")
    
    rows = []
    line_count = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            line_count += 1
            if not line.strip():
                continue
            
            # Parse: "2023-03-01T00:00:00.123Z {json}"
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            
            local_ts, json_str = parts
            
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue
            
            # Only process update and snapshot events
            event = data.get('event', '')
            if event not in ['updated', 'snapshot']:
                continue
            
            # Parse timestamp to microseconds
            try:
                # Remove 'Z' and handle variable precision
                ts_str = local_ts.rstrip('Z')
                # Truncate to 6 decimal places for microseconds
                if '.' in ts_str:
                    base, frac = ts_str.split('.')
                    frac = frac[:6].ljust(6, '0')
                    ts_str = f"{base}.{frac}"
                ts_dt = datetime.fromisoformat(ts_str)
                ts_us = int(ts_dt.timestamp() * 1_000_000)
            except:
                continue
            
            # Process bids and asks
            for side_name, side_value in [('bids', 'BUY'), ('asks', 'SELL')]:
                for order in data.get(side_name, []):
                    try:
                        order_id = int(order['id'])
                        price = int(float(order['px']) * price_multiplier)
                        qty_raw = float(order['qty'])
                        qty = int(qty_raw * qty_multiplier)
                        
                        # qty=0 means CANCEL, qty>0 means ADD
                        event_type = 'CANCEL' if qty_raw == 0 else 'ADD'
                        
                        rows.append({
                            'timestamp': ts_us,
                            'type': event_type,
                            'order_id': order_id,
                            'side': side_value,
                            'price': price,
                            'qty': qty,
                            'order_type': 'LIMIT',
                            'time_in_force': 'GTC'
                        })
                    except (KeyError, ValueError):
                        continue
            
            if line_count % 10000 == 0:
                print(f"  Processed {line_count:,} lines, {len(rows):,} events...")
    
    if not rows:
        print("ERROR: No events found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Statistics
    add_count = (df['type'] == 'ADD').sum()
    cancel_count = (df['type'] == 'CANCEL').sum()
    
    print(f"\n{'='*60}")
    print("L3 DATA STATISTICS")
    print('='*60)
    print(f"Total events: {len(df):,}")
    print(f"  ADDs: {add_count:,} ({100*add_count/len(df):.1f}%)")
    print(f"  CANCELs: {cancel_count:,} ({100*cancel_count/len(df):.1f}%)")
    print(f"Unique Order IDs: {df['order_id'].nunique():,}")
    
    # Check for matching ADD/CANCEL pairs (true L3 property!)
    add_ids = set(df[df['type'] == 'ADD']['order_id'])
    cancel_ids = set(df[df['type'] == 'CANCEL']['order_id'])
    matched = add_ids & cancel_ids
    print(f"Orders with both ADD and CANCEL: {len(matched):,}")
    
    # Time range
    start_us = df['timestamp'].min()
    end_us = df['timestamp'].max()
    duration_s = (end_us - start_us) / 1e6
    print(f"\nTime range:")
    print(f"  Start: {datetime.fromtimestamp(start_us/1e6)}")
    print(f"  End: {datetime.fromtimestamp(end_us/1e6)}")
    print(f"  Duration: {duration_s:.1f} seconds ({duration_s/3600:.2f} hours)")
    
    # Save
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"\nSample:")
    print(df.head(10))
    
    print(f"\nNOTE: Timestamps are in MICROSECONDS (μs)")
    print(f"Use DataLoader with timestamp_unit_ns=1000 to convert to nanoseconds")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch L3 data from Blockchain.com via Tardis.dev",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fetch_l3.py                      # Full day of 2023-03-01
    python fetch_l3.py --hours 1            # First hour only
    python fetch_l3.py --date 2023-04-01    # Different date
    python fetch_l3.py --convert-only       # Convert existing raw data
    
Note: First day of each month is FREE on Tardis.dev!
        """
    )
    parser.add_argument("--date", default="2023-03-01",
                        help="Date to fetch (YYYY-MM-DD, first of month is free)")
    parser.add_argument("--symbol", default="BTC-USD",
                        help="Trading symbol")
    parser.add_argument("--hours", type=int, default=24,
                        help="Hours to fetch (1-24)")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Delay between requests (seconds)")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert existing raw data")
    parser.add_argument("--input", help="Input file for convert-only mode")
    parser.add_argument("--output", help="Output CSV filename")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.convert_only:
        # Just convert existing file
        if args.input:
            input_file = args.input
        else:
            input_file = f"raw_l3/blockchain_l3_{args.date}.jsonl"
        
        if not os.path.exists(input_file):
            print(f"ERROR: {input_file} not found")
            sys.exit(1)
        
        convert_l3_to_csv(input_file, args.output)
    else:
        # Fetch and convert
        raw_file = fetch_full_day(
            date=args.date,
            symbol=args.symbol,
            hours=args.hours,
            delay=args.delay
        )
        
        output_csv = args.output or f"csv/blockchain_l3_{args.date}.csv"
        convert_l3_to_csv(raw_file, output_csv)
        
        print(f"\n{'='*60}")
        print(f"SUCCESS! L3 data saved to: {output_csv}")
        print(f"{'='*60}")
        print(f"\nTo use in gym.py:")
        print(f"  env = LOBEnv(")
        print(f"      data_path='data/csv/{output_csv}',")
        print(f"      timestamp_unit_ns=1000,  # Tardis uses microseconds")
        print(f"  )")


if __name__ == "__main__":
    main()

