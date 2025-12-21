#include "OrderBook.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>

// 1. Data Structure for Input (The "Feed")
struct RawEvent {
    uint64_t timestamp; // Nanoseconds or Milliseconds
    std::string type;   // ADD, CANCEL, MODIFY
    OrderId orderId;
    Side side;
    Price price;
    Quantity quantity;
};

// 2. CSV Parsing Logic (Quick & Dirty)

std::vector<RawEvent> LoadDummyData() {
    std::vector<RawEvent> events;
    
    // Time 0: Initial Book Build
    events.push_back({ 1000, "ADD", 1, Side::SELL, 10100, 100 }); // Sell 100 @ 101.00
    events.push_back({ 1001, "ADD", 2, Side::SELL, 10200, 100 }); // Sell 100 @ 102.00
    events.push_back({ 1002, "ADD", 3, Side::BUY,   9900, 100 }); // Buy  100 @  99.00
    
    // Time 10: Aggressive Order (Trade happens)
    events.push_back({ 1010, "ADD", 4, Side::BUY,  10150, 150 }); // Buy 150 @ 101.50 (Crosses spread!)

    // Time 20: Modify Logic (Resize Down)
    events.push_back({ 1020, "ADD", 5, Side::BUY,   9900, 200 }); // Join bid at 99.00
    events.push_back({ 1025, "MODIFY", 5, Side::BUY, 9900, 50 }); // Resize down (Should keep priority)

    // Time 30: Cancel
    events.push_back({ 1030, "CANCEL", 2, Side::SELL, 0, 0 });    // Cancel the 102.00 Sell

    return events;
}

// 3. The Research Simulation Loop
int main() {
    Orderbook book;
    std::vector<RawEvent> feed = LoadDummyData();

    std::cout << "TIMESTAMP | ACTION | ID | PRICE | QTY | RESULT\n";
    std::cout << "------------------------------------------------\n";

    book.Warmup();

    for (const auto& event : feed) {
        // --- 1. Event Processing ---
        Trades trades;
        
        if (event.type == "ADD") {
            auto order = std::make_unique<Order>(
                event.orderId, event.side, event.price, event.quantity, event.timestamp
            );
            trades = book.AddOrder(std::move(order));
        }
        else if (event.type == "CANCEL") {
            book.CancelOrder(event.orderId);
        }
        else if (event.type == "MODIFY") {
            OrderModify modify(event.orderId, event.side, event.price, event.quantity, event.timestamp);
            trades = book.ModifyOrder(modify);
        }
        // Print the event itself
        std::cout << std::setw(9) << event.timestamp << " | "
                  << std::setw(6) << event.type << " | "
                  << std::setw(2) << event.orderId << " | "
                  << std::setw(5) << event.price << " | "
                  << std::setw(3) << event.quantity << " | ";

        if (trades.empty()) {
            std::cout << "Ack/Resting\n";
        } else {
            // If trades happened, list them
            std::cout << "EXECUTED " << trades.size() << " TRADE(S):\n";
            for (const auto& trade : trades) {
                std::cout << "                                      >>> MATCH: " 
                          << trade.GetBidTrade().quantity_ << " shares @ $" 
                          << trade.GetBidTrade().price_ << "\n";
            }
        }
    }

    // --- 3. Final State Check ---
    std::cout << "\n------------------------------------------------\n";
    std::cout << "FINAL BOOK STATE:\n";
    auto info = book.GetOrderInfos();
    
    std::cout << "BIDS:\n";
    for (const auto& level : info.GetBids()) {
        std::cout << "  $" << level.price << " x " << level.quantity << "\n";
    }
    
    std::cout << "ASKS:\n";
    for (const auto& level : info.GetAsks()) {
        std::cout << "  $" << level.price << " x " << level.quantity << "\n";
    }

    return 0;
}