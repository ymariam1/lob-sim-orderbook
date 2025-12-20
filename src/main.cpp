#include "OrderBook.hpp"

int main() {
    OrderBook book;
    book.add_order({1, 100, 10, true}); 
    book.add_order({2, 101, 5, false});  
    
    book.print_book();
    return 0;
}