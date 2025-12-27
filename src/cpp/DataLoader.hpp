#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <string>
#include <cstring>
#include <iostream>
#include "csv.h"
#include "ExchangeSimulator.hpp"

class DataLoader {
public:
    // Constructor: Opens CSV file and primes the first row
    // timestamp_unit_ns: conversion factor from CSV timestamp to nanoseconds
    //   - Use 1 if CSV is in nanoseconds
    //   - Use 1000 if CSV is in microseconds
    //   - Use 1000000 if CSV is in milliseconds
    //   - Use 1000000000 (1e9) if CSV is in seconds (DEFAULT)
    DataLoader(const std::string& filename, uint64_t timestamp_unit_ns = 1000000000ULL) 
        : reader_(filename), timestamp_unit_ns_(timestamp_unit_ns) {
        // Read header row
        reader_.read_header(io::ignore_extra_column, 
            "timestamp", "type", "order_id", "side", "price", "qty", "order_type", "time_in_force");
        
        // Prime the first row
        has_pending_row_ = ReadNextRow();
    }

    // Pump market data to the exchange until time advances by duration_ns
    // Returns the number of events processed
    std::size_t PumpToExchange(ExchangeSimulator& exchange, uint64_t duration_ns) {
        uint64_t start_time = exchange.GetCurrentTime();
        uint64_t target_time = start_time + duration_ns;
        std::size_t events_processed = 0;

        while (has_pending_row_) {
            // Convert CSV timestamp to nanoseconds for comparison
            uint64_t event_time_ns = pending_timestamp_ * timestamp_unit_ns_;
            
            // If the next event is in the future relative to our target window, STOP
            if (event_time_ns > target_time) {
                break;
            }

            // Process the event based on type
            if (pending_type_ == "ADD") {
                auto order = std::make_unique<Order>(
                    pending_id_,
                    StringToSide(pending_side_),
                    pending_price_,
                    pending_qty_,
                    event_time_ns,  // Use nanosecond timestamp
                    StringToOrderType(pending_order_type_),
                    StringToTimeInForce(pending_tif_)
                );
                exchange.ProcessHistoricalEvent(std::move(order));
            }
            else if (pending_type_ == "CANCEL") {
                // For cancels, we need to directly cancel on the book
                // (Historical cancels happen instantly, no latency)
                exchange.GetOrderbook()->CancelOrder(pending_id_);
            }
            else if (pending_type_ == "MODIFY") {
                OrderModify modify(
                    pending_id_,
                    StringToSide(pending_side_),
                    pending_price_,
                    pending_qty_,
                    event_time_ns,  // Use nanosecond timestamp
                    StringToOrderType(pending_order_type_),
                    StringToTimeInForce(pending_tif_)
                );
                exchange.GetOrderbook()->ModifyOrder(modify);
            }

            events_processed++;
            
            // Read the next row for the next iteration
            has_pending_row_ = ReadNextRow();
        }
        
        // CRITICAL FIX: If we processed all events but haven't reached target_time,
        // force the clock forward. The world doesn't stop just because trading did.
        // This ensures correct timing for sparse data or quiet market periods.
        if (exchange.GetCurrentTime() < target_time) {
            exchange.SetCurrentTime(target_time);
        }
        
        return events_processed;
    }

    // Check if there's more data to read
    bool HasMoreData() const {
        return has_pending_row_;
    }

    // Peek at the next timestamp (in original CSV units)
    uint64_t PeekNextTimestamp() const {
        return pending_timestamp_;
    }

    // Peek at the next timestamp in nanoseconds
    uint64_t PeekNextTimestampNs() const {
        return pending_timestamp_ * timestamp_unit_ns_;
    }

    // Get total events processed (for debugging/stats)
    std::size_t GetTotalEventsProcessed() const {
        return total_events_processed_;
    }

    // Get the timestamp unit multiplier
    uint64_t GetTimestampUnitNs() const {
        return timestamp_unit_ns_;
    }

private:
    bool ReadNextRow() {
        // Read all 8 columns
        bool result = reader_.read_row(
            pending_timestamp_, 
            pending_type_, 
            pending_id_, 
            pending_side_, 
            pending_price_, 
            pending_qty_,
            pending_order_type_,
            pending_tif_
        );
        
        if (result) {
            total_events_processed_++;
        }
        
        return result;
    }

    // Convert string "BUY"/"SELL" to Side enum
    static Side StringToSide(const std::string& s) {
        return (s == "BUY" || s == "b" || s == "B") ? Side::BUY : Side::SELL;
    }

    // Convert string "MARKET"/"LIMIT" to OrderType enum
    static int StringToOrderType(const std::string& s) {
        if (s == "MARKET" || s == "M") return OrderType::MARKET;
        return OrderType::LIMIT;
    }

    // Convert string "GTC"/"FOK"/"IOC"/"GFD" to TimeInForce enum
    static int StringToTimeInForce(const std::string& s) {
        if (s == "GTC") return TimeInForce::GTC;
        if (s == "FOK") return TimeInForce::FOK;
        if (s == "IOC") return TimeInForce::IOC;
        if (s == "GFD") return TimeInForce::GFD;
        return TimeInForce::GTC;
    }

    // CSV Reader with 8 columns
    io::CSVReader<8> reader_;
    
    // Timestamp conversion factor (CSV units to nanoseconds)
    uint64_t timestamp_unit_ns_;
    
    // Buffer for the pending row
    uint64_t pending_timestamp_ = 0;
    std::string pending_type_;
    uint64_t pending_id_ = 0;
    std::string pending_side_;
    int64_t pending_price_ = 0;
    int64_t pending_qty_ = 0;
    std::string pending_order_type_;
    std::string pending_tif_;
    
    bool has_pending_row_ = false;
    std::size_t total_events_processed_ = 0;
};

#endif // DATALOADER_HPP

