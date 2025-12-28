#!/usr/bin/env python3
"""
Bulk fetch L3 data for ALL available months from Blockchain.com via Tardis.dev.

Fetches the FIRST day of each month (which is FREE on Tardis.dev!)
from December 2025 going backwards as far as data is available.

Features:
- Skips dates that already have data (no duplicates)
- Automatic retry on failure
- Progress tracking
- Estimates total download time

Usage:
    python fetch_all_months.py                    # Fetch all available months
    python fetch_all_months.py --start 2024-01    # Start from Jan 2024
    python fetch_all_months.py --end 2022-06      # Go back to June 2022
    python fetch_all_months.py --dry-run          # Just show what would be fetched
    python fetch_all_months.py --check            # Check what data already exists

Output:
    - data/raw_l3/blockchain_l3_YYYY-MM-01.jsonl  # Raw data
    - data/csv/blockchain_l3_YYYY-MM-01.csv       # Converted CSV
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import the existing fetch functions
try:
    from fetch_l3 import fetch_full_day, convert_l3_to_csv, fetch_l3_minute
except ImportError:
    # Running from different directory
    sys.path.insert(0, str(Path(__file__).parent))
    from fetch_l3 import fetch_full_day, convert_l3_to_csv, fetch_l3_minute

# Thread-safe print
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def check_data_available(date: str, symbol: str = "BTC-USD") -> bool:
    """
    Check if data is available for a given date on Tardis.dev.
    
    Returns True if data exists, False otherwise.
    """
    try:
        data = fetch_l3_minute(date, symbol, offset=0, max_retries=1)
        return data is not None and len(data.strip()) > 0
    except Exception:
        return False


def get_existing_dates(raw_dir: str = "raw_l3", csv_dir: str = "csv") -> set:
    """
    Get set of dates that already have data downloaded.
    
    Returns:
        Set of dates in YYYY-MM-DD format
    """
    existing = set()
    
    # Check raw files
    if os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if f.startswith("blockchain_l3_") and f.endswith(".jsonl"):
                # Extract date from filename
                date = f.replace("blockchain_l3_", "").replace(".jsonl", "")
                # Verify file has content
                filepath = os.path.join(raw_dir, f)
                if os.path.getsize(filepath) > 1000:  # At least 1KB
                    existing.add(date)
    
    # Check CSV files
    if os.path.exists(csv_dir):
        for f in os.listdir(csv_dir):
            if f.startswith("blockchain_l3_") and f.endswith(".csv"):
                date = f.replace("blockchain_l3_", "").replace(".csv", "")
                filepath = os.path.join(csv_dir, f)
                if os.path.getsize(filepath) > 1000:
                    existing.add(date)
    
    return existing


def generate_first_of_month_dates(
    start_year: int = 2025,
    start_month: int = 12,
    end_year: int = 2019,
    end_month: int = 1,
) -> List[str]:
    """
    Generate list of first-of-month dates from start to end.
    
    Goes backwards in time (newest first).
    Skips future dates automatically.
    """
    dates = []
    
    current = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    now = datetime.now()
    
    while current >= end:
        # Skip future dates
        if current <= now:
            dates.append(current.strftime("%Y-%m-%d"))
        current -= relativedelta(months=1)
    
    return dates


def find_earliest_available_date(
    symbol: str = "BTC-USD",
    start_year: int = 2019,
) -> Optional[str]:
    """
    Search to find the earliest date with data.
    
    Blockchain.com data on Tardis.dev starts around Feb 2023.
    
    Returns:
        Earliest date with data, or None if not found
    """
    print("Searching for earliest available data...")
    print("(Testing first day of each month)")
    print()
    
    # Known range for Blockchain.com: approximately Feb 2023 onwards
    # Test each month going backwards from a known good date
    test_dates = [
        "2023-12-01", "2023-11-01", "2023-10-01", "2023-09-01",
        "2023-08-01", "2023-07-01", "2023-06-01", "2023-05-01",
        "2023-04-01", "2023-03-01", "2023-02-01", "2023-01-01",
        "2022-12-01", "2022-11-01", "2022-10-01",
    ]
    
    earliest = None
    
    for date_str in test_dates:
        print(f"  {date_str}: ", end="", flush=True)
        
        if check_data_available(date_str, symbol):
            print("✓ Available")
            earliest = date_str
        else:
            print("✗ Not available")
            if earliest:
                break  # Found the boundary
        
        time.sleep(0.3)
    
    return earliest


def get_latest_available_date() -> str:
    """
    Get the latest date that might have data (today - 1 day).
    
    Returns date in YYYY-MM-DD format.
    """
    # Use yesterday to be safe
    today = datetime.now()
    
    # Go to first of current month if it's past
    if today.day > 1:
        latest = datetime(today.year, today.month, 1)
    else:
        # Use first of previous month
        latest = datetime(today.year, today.month, 1) - relativedelta(months=1)
    
    return latest.strftime("%Y-%m-%d")


def estimate_download_time(num_dates: int, hours_per_day: int = 24, workers: int = 1) -> Tuple[float, float]:
    """
    Estimate download time.
    
    Returns:
        (minutes, hours)
    """
    # Each minute takes ~0.35 seconds (request + processing + delay)
    minutes_per_day = hours_per_day * 60
    seconds_per_day = minutes_per_day * 0.35
    
    total_seconds = seconds_per_day * num_dates / workers
    return total_seconds / 60, total_seconds / 3600


def download_single_month(
    date: str,
    symbol: str = "BTC-USD",
    hours: int = 24,
    delay: float = 0.05,
    worker_id: int = 0,
) -> Tuple[str, bool, str]:
    """
    Download a single month of L3 data.
    
    Returns:
        (date, success, message)
    """
    try:
        safe_print(f"[Worker {worker_id}] Starting {date}...")
        
        # Fetch raw data
        raw_file = fetch_full_day(
            date=date,
            symbol=symbol,
            hours=hours,
            delay=delay,
        )
        
        # Convert to CSV
        output_csv = f"csv/blockchain_l3_{date}.csv"
        os.makedirs("csv", exist_ok=True)
        convert_l3_to_csv(raw_file, output_csv)
        
        safe_print(f"[Worker {worker_id}] ✓ Completed {date}")
        return (date, True, f"Saved to {output_csv}")
        
    except Exception as e:
        safe_print(f"[Worker {worker_id}] ✗ Failed {date}: {e}")
        return (date, False, str(e))


def fetch_all_months(
    start_month: str = "2025-12",
    end_month: str = "2020-01",
    hours: int = 24,
    symbol: str = "BTC-USD",
    dry_run: bool = False,
    delay: float = 0.05,
    workers: int = 1,
) -> List[str]:
    """
    Fetch L3 data for all months from start to end.
    
    Args:
        start_month: Start month in YYYY-MM format
        end_month: End month in YYYY-MM format  
        hours: Hours per day to fetch (24 = full day)
        symbol: Trading symbol
        dry_run: If True, just show what would be fetched
        delay: Delay between API requests
        workers: Number of parallel download threads
        
    Returns:
        List of successfully downloaded dates
    """
    # Parse months
    start_parts = start_month.split("-")
    end_parts = end_month.split("-")
    
    start_year, start_mon = int(start_parts[0]), int(start_parts[1])
    end_year, end_mon = int(end_parts[0]), int(end_parts[1])
    
    # Generate all first-of-month dates
    all_dates = generate_first_of_month_dates(start_year, start_mon, end_year, end_mon)
    
    print("=" * 70)
    print("BULK L3 DATA DOWNLOAD - Blockchain.com via Tardis.dev")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Date range: {all_dates[-1]} to {all_dates[0]}")
    print(f"Total months: {len(all_dates)}")
    print(f"Hours per day: {hours}")
    print(f"Parallel workers: {workers}")
    print()
    
    # Check existing data
    existing = get_existing_dates()
    print(f"Already downloaded: {len(existing)} dates")
    
    # Filter to only missing dates
    missing_dates = [d for d in all_dates if d not in existing]
    print(f"Missing (to download): {len(missing_dates)} dates")
    
    if not missing_dates:
        print("\n✓ All data already downloaded!")
        return []
    
    # Estimate time
    est_min, est_hours = estimate_download_time(len(missing_dates), hours, workers)
    print(f"\nEstimated download time: {est_min:.0f} minutes ({est_hours:.1f} hours)")
    print(f"  (with {workers} parallel workers)")
    
    if dry_run:
        print("\n[DRY RUN] Would download:")
        for d in missing_dates[:10]:
            print(f"  - {d}")
        if len(missing_dates) > 10:
            print(f"  ... and {len(missing_dates) - 10} more")
        return []
    
    # Confirm
    print()
    response = input(f"Download {len(missing_dates)} months with {workers} workers? [y/N] ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return []
    
    # Download with threading
    successful = []
    failed = []
    start_time = time.time()
    
    print()
    print("=" * 70)
    print(f"Starting parallel download with {workers} workers...")
    print("=" * 70)
    
    if workers == 1:
        # Sequential mode (original behavior)
        for i, date in enumerate(missing_dates):
            print()
            print(f"[{i+1}/{len(missing_dates)}] Downloading {date}")
            
            try:
                raw_file = fetch_full_day(
                    date=date,
                    symbol=symbol,
                    hours=hours,
                    delay=delay,
                )
                
                output_csv = f"csv/blockchain_l3_{date}.csv"
                os.makedirs("csv", exist_ok=True)
                convert_l3_to_csv(raw_file, output_csv)
                
                successful.append(date)
                print(f"✓ Successfully downloaded {date}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Progress saved.")
                break
            except Exception as e:
                print(f"✗ Failed to download {date}: {e}")
                failed.append(date)
            
            time.sleep(1)
    else:
        # Parallel mode
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all jobs
                futures = {}
                for i, date in enumerate(missing_dates):
                    worker_id = i % workers
                    future = executor.submit(
                        download_single_month,
                        date=date,
                        symbol=symbol,
                        hours=hours,
                        delay=delay,
                        worker_id=worker_id,
                    )
                    futures[future] = date
                
                # Process results as they complete
                for future in as_completed(futures):
                    date = futures[future]
                    try:
                        result_date, success, message = future.result()
                        if success:
                            successful.append(result_date)
                        else:
                            failed.append(result_date)
                    except Exception as e:
                        safe_print(f"✗ Exception for {date}: {e}")
                        failed.append(date)
                    
                    # Progress update
                    total_done = len(successful) + len(failed)
                    elapsed = time.time() - start_time
                    rate = total_done / elapsed if elapsed > 0 else 0
                    remaining = len(missing_dates) - total_done
                    eta = remaining / rate if rate > 0 else 0
                    safe_print(f"\nProgress: {total_done}/{len(missing_dates)} | "
                               f"Success: {len(successful)} | Failed: {len(failed)} | "
                               f"ETA: {eta/60:.1f} min")
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Some downloads may still be completing...")
    
    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed dates:")
        for d in failed:
            print(f"  - {d}")
        print("\nTo retry failed dates:")
        print(f"  python fetch_all_months.py --start {failed[0][:7]} --end {failed[-1][:7]}")
    
    return successful


def list_existing_data():
    """List all existing downloaded data."""
    existing = get_existing_dates()
    
    print("=" * 70)
    print("EXISTING L3 DATA")
    print("=" * 70)
    
    if not existing:
        print("No data downloaded yet.")
        return
    
    sorted_dates = sorted(existing, reverse=True)
    
    print(f"Total months: {len(sorted_dates)}")
    print()
    
    # Group by year
    by_year = {}
    for d in sorted_dates:
        year = d[:4]
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(d)
    
    for year in sorted(by_year.keys(), reverse=True):
        dates = by_year[year]
        months = [d[5:7] for d in dates]
        print(f"{year}: {', '.join(sorted(months))}")
    
    # Total size
    total_size = 0
    for d in sorted_dates:
        csv_path = f"csv/blockchain_l3_{d}.csv"
        if os.path.exists(csv_path):
            total_size += os.path.getsize(csv_path)
    
    print(f"\nTotal CSV size: {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk download L3 data from Blockchain.com via Tardis.dev",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fetch_all_months.py                    # Fetch all (sequential)
    python fetch_all_months.py -w 4               # 4 parallel downloads (4x faster!)
    python fetch_all_months.py -w 8 --hours 1     # Fast mode: 8 workers, 1hr each
    python fetch_all_months.py --start 2024-01    # Start from Jan 2024
    python fetch_all_months.py --dry-run          # Show what would be fetched
    python fetch_all_months.py --check            # Check existing data
    
Note: Tardis.dev provides FREE data for the 1st day of each month!
      Use 2-4 workers to balance speed vs rate limiting.
        """
    )
    # Default to current month
    now = datetime.now()
    current_month = f"{now.year}-{now.month:02d}"
    
    parser.add_argument("--start", default=current_month,
                        help=f"Start month (YYYY-MM, default: {current_month})")
    parser.add_argument("--end", default="2023-03",
                        help="End month (YYYY-MM, default: 2023-03, earliest available)")
    parser.add_argument("--hours", type=int, default=24,
                        help="Hours per day to fetch (default: 24)")
    parser.add_argument("--symbol", default="BTC-USD",
                        help="Trading symbol (default: BTC-USD)")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Delay between requests in seconds")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of parallel download threads (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    parser.add_argument("--check", action="store_true",
                        help="Check existing data and exit")
    parser.add_argument("--find-earliest", action="store_true",
                        help="Find the earliest available date")
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.check:
        list_existing_data()
        return
    
    if args.find_earliest:
        earliest = find_earliest_available_date(args.symbol)
        if earliest:
            print(f"\nEarliest available: {earliest}")
        else:
            print("\nCould not find earliest date")
        return
    
    # Main download
    fetch_all_months(
        start_month=args.start,
        end_month=args.end,
        hours=args.hours,
        symbol=args.symbol,
        dry_run=args.dry_run,
        delay=args.delay,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

