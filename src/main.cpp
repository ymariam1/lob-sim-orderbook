#include "OrderBook.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// 1. Structure to hold CSV rows
struct MarketEvent {
    uint64_t timestamp;
    std::string type; // "ADD", "CANCEL", "TRADE"
    OrderId orderId;
    Side side;
    Price price;
    Quantity quantity;
};

int main() {
    Orderbook book;
    
    // 2. Load "Dummy" Data (Simulating a CSV load)
    std::vector<MarketEvent> events = {
        {1000, "ADD", 1, Side::SELL, 10050, 100}, // Ask @ 100.50
        {1001, "ADD", 2, Side::BUY,  10040, 100}, // Bid @ 100.40
        {1002, "ADD", 3, Side::BUY,  10045, 50},  // Bid @ 100.45 (Best Bid)
        {1005, "ADD", 4, Side::SELL, 10045, 20}   // Market Cross (Trade!)
    };

    // 3. The Simulation Loop
    for (const auto& event : events) {
        if (event.type == "ADD") {
            auto order = std::make_shared<Order>(
                event.orderId, event.side, event.price, event.quantity, event.timestamp
            );
            auto trades = book.AddOrder(order);
            
            // 4. LOGGING (Crucial for Research)
            for (const auto& trade : trades) {
                std::cout << "TRADE EXECUTION: " 
                          << trade.GetBidTrade().quantity_ << " @ " 
                          << trade.GetBidTrade().price_ << std::endl;
            }
        }
        // Handle Cancel/Modify...
    }
    
    return 0;
}