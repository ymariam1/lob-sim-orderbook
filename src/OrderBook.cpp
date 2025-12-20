#include "OrderBook.hpp"
#include <iostream>

void OrderBook::add_order(const Order& order) {
    if (order.is_buy) {
        bids[order.price].push_back(order);
    } else {
        asks[order.price].push_back(order);
    }
}

void OrderBook::print_book() {
    std::cout << "Bids:" << std::endl;
    for (const auto& [price, orders] : bids) {
        std::cout << "Price: " << price << ", Quantity: " << orders.size() << std::endl;
    }
    std::cout << "Asks:" << std::endl;
    for (const auto& [price, orders] : asks) {
        std::cout << "Price: " << price << ", Quantity: " << orders.size() << std::endl;
    }
}
