#ifndef ORDER_BOOK_HPP
#define ORDER_BOOK_HPP
#include <cstdint>
#include <map>
#include <vector>

// Order struct
struct Order {
    uint64_t id;
    int price;
    int quantity;
    bool is_buy;
};

// OrderBook class
class OrderBook {
public:
    void add_order(const Order& order);
    void print_book();

private:
    // We use std::map because it keeps keys (prices) sorted automatically!
    // Bids: High to Low (greater<int>)
    std::map<int, std::vector<Order>, std::greater<int>> bids;
    
    // Asks: Low to High (less<int>)
    std::map<int, std::vector<Order>, std::less<int>> asks;
};
#endif