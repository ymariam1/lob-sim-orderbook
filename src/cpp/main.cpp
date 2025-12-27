#include "OrderBook.hpp"
#include "csv.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>


Side StringToSide(const std::string& s) {
    return (s == "BUY" || s == "b") ? Side::BUY : Side::SELL;
}

int StringToOrderType(const std::string& s) {
    if (s == "MARKET") return OrderType::MARKET;
    return OrderType::LIMIT; // Default to LIMIT
}

int StringToTimeInForce(const std::string& s) {
    if (s == "GTC") return TimeInForce::GTC;
    if (s == "FOK") return TimeInForce::FOK;
    if (s == "IOC") return TimeInForce::IOC;
    if (s == "GFD") return TimeInForce::GFD;
    return TimeInForce::GTC; // Default to GTC
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./lob_sim <data.csv>\n";
        return 1;
    }
    // Setup the Reader
    // <8> means we expect 8 columns. This enables compile-time optimizations.
    io::CSVReader<8> in(argv[1]);

    // 3. Configure Columns
    // This looks for these specific headers in the CSV regardless of order.
    // If CSV has no headers, use: in.set_header("timestamp", "type", ...);
    in.read_header(io::ignore_extra_column, 
        "timestamp", "type", "order_id", "side", "price", "qty", 
        "order_type", "time_in_force"
    );
    
    uint64_t timestamp;
    std::string type;
    uint64_t orderId;
    std::string sideStr;
    int64_t price;
    int64_t quantity;
    std::string orderTypeStr;
    std::string timeInForceStr;
    Orderbook book;
    book.Warmup();

    uint64_t eventCount = 0;
    std::cout << "Starting Simulation on " << argv[1] << "...\n";

    // Streaming Loop
    // read_row returns false when EOF is reached
    while(in.read_row(timestamp, type, orderId, sideStr, price, quantity, orderTypeStr, timeInForceStr)) {
        
        Trades trades;
        Side side = StringToSide(sideStr);
        int orderType = StringToOrderType(orderTypeStr);
        int timeInForce = StringToTimeInForce(timeInForceStr);

        // --- Processing Logic ---
        if (type == "ADD") {
            auto order = std::make_unique<Order>(
                orderId, side, price, quantity, timestamp, orderType, timeInForce
            );
            trades = book.AddOrder(std::move(order));
        }
        else if (type == "CANCEL") {
            book.CancelOrder(orderId);
        }
        else if (type == "MODIFY") {
             OrderModify modify(orderId, side, price, quantity, timestamp, orderType, timeInForce);
             trades = book.ModifyOrder(modify);
        }

        // --- Logging / Metrics ---
        if (++eventCount % 100000 == 0) {
            std::cout << "Processed " << eventCount << " events...\n";
        }
    }

    std::cout << "Done. Total Events: " << eventCount << "\n";
    std::cout << "Final Book Depth: " << book.Size() << " orders.\n";

    return 0;
}