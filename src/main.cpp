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


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./lob_sim <data.csv>\n";
        return 1;
    }
    // Setup the Reader
    // <6> means we expect 6 columns. This enables compile-time optimizations.
    io::CSVReader<6> in(argv[1]);

    // 3. Configure Columns
    // This looks for these specific headers in the CSV regardless of order.
    // If CSV has no headers, use: in.set_header("timestamp", "type", ...);
    in.read_header(io::ignore_extra_column, 
        "timestamp", "type", "order_id", "side", "price", "qty"
    );
    
    uint64_t timestamp;
    std::string type;
    uint64_t orderId;
    std::string sideStr;
    int64_t price;
    int64_t quantity;

    Orderbook book;
    book.Warmup();

    uint64_t eventCount = 0;
    std::cout << "Starting Simulation on " << argv[1] << "...\n";

    // Streaming Loop
    // read_row returns false when EOF is reached
    while(in.read_row(timestamp, type, orderId, sideStr, price, quantity)) {
        
        Trades trades;
        Side side = StringToSide(sideStr);

        // --- Processing Logic ---
        if (type == "ADD") {
            auto order = std::make_unique<Order>(
                orderId, side, price, quantity, timestamp
            );
            trades = book.AddOrder(std::move(order));
        }
        else if (type == "CANCEL") {
            book.CancelOrder(orderId);
        }
        else if (type == "MODIFY") {
             OrderModify modify(orderId, side, price, quantity, timestamp);
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