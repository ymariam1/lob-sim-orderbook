#ifndef ORDER_BOOK_HPP
#define ORDER_BOOK_HPP
#include <cstdint>
#include <map>
#include <vector>
#include <numeric>
#include <format>
#include <deque>
#include <unordered_map>
#include <memory>
#include <stdexcept>

// Order struct
using Price = std::int64_t;
using Quantity = std::int64_t;
using OrderId = std::uint64_t;
enum class Side {
    BUY,
    SELL
};

enum OrderType {
    MARKET,
    LIMIT
};

enum TimeInForce {
    GTC,
    FOK,
    IOC,
    GFD
};

struct LevelInfo {
    Price price;
    Quantity quantity;
};
using LevelInfos = std::vector<LevelInfo>;

class OrderbookLevelInfos {
public:
        OrderbookLevelInfos(const LevelInfos& bids, const LevelInfos& asks)
        : bids_{ bids }, asks_{ asks } {}
    
        const LevelInfos& GetBids() const { return bids_; }
        const LevelInfos& GetAsks() const { return asks_; }

private:
        LevelInfos bids_;
        LevelInfos asks_;
};

class Order {
public:
        Order(OrderId id, Side side, Price price, Quantity quantity, uint64_t timestamp, int orderType, int timeInForce) 
        : orderId_{ id }, side_{ side }, price_{ price }, initialQuantity_{ quantity }, remainingQuantity_{ quantity }, timestamp_{ timestamp }, orderType_{ orderType }, timeInForce_{ timeInForce } {}

        OrderId GetOrderId() const { return orderId_; }
        Side GetSide() const { return side_; }
        Price GetPrice() const { return price_; }
        Quantity GetInitialQuantity() const { return initialQuantity_; }
        Quantity GetRemainingQuantity() const { return remainingQuantity_; }
        Quantity GetFilledQuantity() const { return filledQuantity_; }
        uint64_t GetTimestamp() const { return timestamp_; }
        int GetOrderType() const { return orderType_; }
        int GetTimeInForce() const { return timeInForce_; }
        bool IsFilled() const { return GetRemainingQuantity() == 0; }
        void Fill(Quantity quantity) {
            if (quantity > GetRemainingQuantity()) {
                throw std::logic_error(std::format("Cannot fill more than the remaining quantity for order {}", GetOrderId()));
            }
            filledQuantity_ += quantity;
            remainingQuantity_ -= quantity;
        }
        void ResizeQuantity(Quantity newQuantity) {
            // Calculate New Remaining
            // We use signed integers temporarily to check for negative values
            long long newRemaining = (long long)newQuantity - (long long)filledQuantity_;

            if (newRemaining <= 0) {
                // Case 3: We already filled more than the new size, so we can cancel the rest.
                initialQuantity_ = newQuantity;
                remainingQuantity_ = 0; 
            } else {
                // Case 1 & 2: Standard downsize
                initialQuantity_ = newQuantity;
                remainingQuantity_ = (Quantity)newRemaining;
            }
        }

private:
        OrderId orderId_;
        Side side_;
        Price price_;
        Quantity initialQuantity_;
        Quantity remainingQuantity_;
        uint64_t timestamp_;
        Quantity filledQuantity_{0};
        int orderType_;
        int timeInForce_;
};

using OrderPointer = std::unique_ptr<Order>;
using OrderPointers = std::deque<OrderPointer>;

class OrderModify {
public:
    OrderModify(OrderId orderId, Side side, Price price, Quantity quantity, uint64_t timestamp, int orderType, int timeInForce)
    : orderId_{ orderId }, price_{ price }, side_{ side }, quantity_{ quantity }, timestamp_{ timestamp }, orderType_{ orderType }, timeInForce_{ timeInForce } {}

    OrderId GetOrderId() const { return orderId_; }
    Side GetSide() const { return side_; }
    Price GetPrice() const { return price_; }
    Quantity GetQuantity() const { return quantity_; }
    uint64_t GetTimestamp() const { return timestamp_; }
    int GetOrderType() const { return orderType_; }
    int GetTimeInForce() const { return timeInForce_; }

    OrderPointer toOrderPointer() const {
        return std::make_unique<Order>(GetOrderId(), GetSide(), GetPrice(), GetQuantity(), GetTimestamp(), orderType_, timeInForce_);
    }
private:
    OrderId orderId_;
    Side side_;
    Price price_;
    Quantity quantity_;
    uint64_t timestamp_;
    int orderType_;
    int timeInForce_;
};

struct TradeInfo {
    OrderId orderId_;
    Price price_;
    Quantity quantity_;
};

class Trade {
public:
    Trade(const TradeInfo& bidTrade, const TradeInfo& askTrade)
    : bidTrade_{ bidTrade }, askTrade_{ askTrade } {}

    const TradeInfo& GetBidTrade() const { return bidTrade_; }
    const TradeInfo& GetAskTrade() const { return askTrade_; }

private:
    TradeInfo bidTrade_;
    TradeInfo askTrade_;
};

using Trades = std::vector<Trade>;

class Orderbook {
private:
    struct OrderEntry {
        std::size_t index_;
        Price price_;
        Side side_;
    };

    std::map<Price, OrderPointers, std::greater<Price>> bids_;
    std::map<Price, OrderPointers, std::less<Price>> asks_;
    std::unordered_map<OrderId, OrderEntry> orders_;

    void UpdateIndicesAfterFrontRemoval(Price price, Side side) {
        OrderPointers* orders = GetOrdersPointer(price, side);
        if (!orders) return;

        // After removing front element (index 0), all remaining indices shift down by 1
        for (std::size_t i = 0; i < orders->size(); ++i) {
            auto orderId = (*orders)[i]->GetOrderId();
            auto it = orders_.find(orderId);
            if (it != orders_.end()) {
                it->second.index_ = i;
            }
        }
    }

    void UpdateIndicesAfterRemoval(Price price, Side side, std::size_t removedIndex) {
        OrderPointers* orders = GetOrdersPointer(price, side);
        if (!orders) return;

        // After removing element at removedIndex, all indices after it shift down by 1
        for (std::size_t i = removedIndex; i < orders->size(); ++i) {
            auto orderId = (*orders)[i]->GetOrderId();
            auto it = orders_.find(orderId);
            if (it != orders_.end()) {
                it->second.index_ = i;
            }
        }
    }

    // Check if a GFD order is expired based on a reference timestamp
    // Assumes timestamps are in seconds since epoch
    static bool IsGFDExpired(uint64_t orderTimestamp, uint64_t referenceTimestamp) {
        constexpr uint64_t SECONDS_PER_DAY = 86400;
        uint64_t orderDay = orderTimestamp / SECONDS_PER_DAY;
        uint64_t referenceDay = referenceTimestamp / SECONDS_PER_DAY;
        return orderDay < referenceDay; // Expired if order is from a previous day
    }

    // Get orders pointer by side and price
    OrderPointers* GetOrdersPointer(Price price, Side side) {
        if (side == Side::SELL) {
            auto it = asks_.find(price);
            return (it != asks_.end()) ? &it->second : nullptr;
        } else {
            auto it = bids_.find(price);
            return (it != bids_.end()) ? &it->second : nullptr;
        }
    }

    // Remove expired GFD order from front of queue
    bool RemoveExpiredGFDFromFront(OrderPointers* orders, Price price, Side side, uint64_t referenceTimestamp) {
        if (!orders || orders->empty()) {
            return false;
        }
        
        auto& frontOrder = orders->front();
        if (frontOrder->GetTimeInForce() == TimeInForce::GFD && 
            IsGFDExpired(frontOrder->GetTimestamp(), referenceTimestamp)) {
            OrderId expiredOrderId = frontOrder->GetOrderId();
            orders->pop_front();
            orders_.erase(expiredOrderId);
            UpdateIndicesAfterFrontRemoval(price, side);
            
            if (orders->empty()) {
                if (side == Side::SELL) {
                    asks_.erase(price);
                } else {
                    bids_.erase(price);
                }
            }
            return true;
        }
        return false;
    }

    // Check if FOK order can be fully filled (pre-calculation)
    bool CanFillFOK(Side side, Price price, Quantity quantity, int orderType) const {
        Quantity availableQuantity = 0;
        
        if (side == Side::BUY) {
            if (asks_.empty()) {
                return false;
            }
            // For market orders, sum all available asks
            // For limit orders, sum asks at or below the limit price
            for (const auto& [askPrice, asks] : asks_) {
                if (orderType == OrderType::LIMIT && askPrice > price) {
                    break; // Limit order: can't match above limit price
                }
                for (const auto& ask : asks) {
                    availableQuantity += ask->GetRemainingQuantity();
                    if (availableQuantity >= quantity) {
                        return true;
                    }
                }
            }
        } else { // SELL
            if (bids_.empty()) {
                return false;
            }
            // For market orders, sum all available bids
            // For limit orders, sum bids at or above the limit price
            for (const auto& [bidPrice, bids] : bids_) {
                if (orderType == OrderType::LIMIT && bidPrice < price) {
                    break; // Limit order: can't match below limit price
                }
                for (const auto& bid : bids) {
                    availableQuantity += bid->GetRemainingQuantity();
                    if (availableQuantity >= quantity) {
                        return true;
                    }
                }
            }
        }
        
        return availableQuantity >= quantity;
    }

    // Aggressively match an incoming order against the book
    Trades MatchAggressively(Order* incomingOrder) {
        Trades trades;
        Side side = incomingOrder->GetSide();
        Side oppositeSide = side == Side::BUY ? Side::SELL : Side::BUY;
        Price price = incomingOrder->GetPrice();
        int orderType = incomingOrder->GetOrderType();
        OrderId incomingOrderId = incomingOrder->GetOrderId();
        
        while (!incomingOrder->IsFilled()) {
            // Check if we can match
            bool canMatch = false;
            Price matchPrice = 0;
            OrderPointers* oppositeOrders = nullptr;
            Price oppositePrice = 0;
            
            if (side == Side::BUY) {
                if (asks_.empty()) {
                    break;
                }
                auto askIt = asks_.begin();
                matchPrice = askIt->first;
                // Market orders match any ask, limit orders only if price >= ask price
                if (orderType == OrderType::MARKET || price >= matchPrice) {
                    canMatch = true;
                    oppositeOrders = &askIt->second;
                    oppositePrice = askIt->first;
                }
            } else { // SELL
                if (bids_.empty()) {
                    break;
                }
                auto bidIt = bids_.begin();
                matchPrice = bidIt->first;
                // Market orders match any bid, limit orders only if price <= bid price
                if (orderType == OrderType::MARKET || price <= matchPrice) {
                    canMatch = true;
                    oppositeOrders = &bidIt->second;
                    oppositePrice = bidIt->first;
                }
            }
            
            if (!canMatch || oppositeOrders->empty()) {
                break;
            }
            
            // Skip expired GFD orders
            while (RemoveExpiredGFDFromFront(oppositeOrders, oppositePrice, oppositeSide, incomingOrder->GetTimestamp())) {
                // Continue removing expired orders until we find a valid one or run out
            }
            
            if (oppositeOrders->empty()) {
                break; // No valid orders to match
            }
            
            // Match against the best opposite order
            auto& oppositeOrder = oppositeOrders->front();
            Quantity matchQuantity = std::min(
                incomingOrder->GetRemainingQuantity(),
                oppositeOrder->GetRemainingQuantity()
            );
            
            // Execution price: price-time priority (older order's price)
            // For market orders, use the resting order's price (market orders don't have meaningful price)
            Price executionPrice;
            if (orderType == OrderType::MARKET) {
                executionPrice = oppositeOrder->GetPrice();
            } else {
                executionPrice = (incomingOrder->GetTimestamp() < oppositeOrder->GetTimestamp()) 
                    ? incomingOrder->GetPrice() 
                    : oppositeOrder->GetPrice();
            }
            
            // Fill both orders
            incomingOrder->Fill(matchQuantity);
            oppositeOrder->Fill(matchQuantity);
            
            OrderId oppositeOrderId = oppositeOrder->GetOrderId();
            
            // Remove filled opposite order from book
            if (oppositeOrder->IsFilled()) {
                oppositeOrders->pop_front();
                orders_.erase(oppositeOrderId);
                UpdateIndicesAfterFrontRemoval(oppositePrice, oppositeSide);
                
                // Remove price level if empty
                if (oppositeOrders->empty()) {
                    if (oppositeSide == Side::SELL) {
                        asks_.erase(oppositePrice);
                    } else {
                        bids_.erase(oppositePrice);
                    }
                }
            }
            
            // Record trade
            if (side == Side::BUY) {
                trades.push_back(Trade{
                    TradeInfo{ incomingOrderId, executionPrice, matchQuantity },
                    TradeInfo{ oppositeOrderId, executionPrice, matchQuantity }
                });
            } else {
                trades.push_back(Trade{
                    TradeInfo{ oppositeOrderId, executionPrice, matchQuantity },
                    TradeInfo{ incomingOrderId, executionPrice, matchQuantity }
                });
            }
        }
        
        return trades;
    }

    Trades MatchOrders() {
        Trades trades;
        trades.reserve(orders_.size());
        while (true) {
            if (bids_.empty() || asks_.empty()) {
                break;
            }

            auto& [bidPrice, bids] = *bids_.begin();
            auto& [askPrice, asks] = *asks_.begin();

            if (bidPrice < askPrice) {
                break;
            }

            while (bids.size() && asks.size()) {
                auto& bid = bids.front();
                auto& ask = asks.front();

                // Skip expired GFD orders
                if (RemoveExpiredGFDFromFront(&bids, bidPrice, Side::BUY, ask->GetTimestamp())) {
                    if (bids.empty()) {
                        bids_.erase(bidPrice);
                    }
                    continue;
                }
                
                if (RemoveExpiredGFDFromFront(&asks, askPrice, Side::SELL, bid->GetTimestamp())) {
                    if (asks.empty()) {
                        asks_.erase(askPrice);
                    }
                    continue;
                }

                Quantity quantity = std::min(bid->GetRemainingQuantity(), ask->GetRemainingQuantity());
                Price executionPrice = (bid->GetTimestamp() < ask->GetTimestamp()) ? bid->GetPrice() : ask->GetPrice();
                OrderId bidOrderId = bid->GetOrderId();
                OrderId askOrderId = ask->GetOrderId();
                
                bid->Fill(quantity);
                ask->Fill(quantity);

                if (bid->IsFilled()) {
                    bids.pop_front();
                    orders_.erase(bidOrderId);
                    UpdateIndicesAfterFrontRemoval(bidPrice, Side::BUY);
                    if (bids.empty()) {
                        bids_.erase(bidPrice);
                    }
                }
                if (ask->IsFilled()) {
                    asks.pop_front();
                    orders_.erase(askOrderId);
                    UpdateIndicesAfterFrontRemoval(askPrice, Side::SELL);
                    if (asks.empty()) {
                        asks_.erase(askPrice);
                    }
                }
                
                trades.push_back(Trade{ 
                    TradeInfo{ bidOrderId, executionPrice, quantity }, 
                    TradeInfo{ askOrderId, executionPrice, quantity } 
                });
            }
        }
        return trades;
    }
public:

    Trades AddOrder(OrderPointer&& order) {
        OrderId orderId = order->GetOrderId();
        Price price = order->GetPrice();
        Side side = order->GetSide();
        int orderType = order->GetOrderType();
        int timeInForce = order->GetTimeInForce();
        Quantity initialQuantity = order->GetInitialQuantity();
        
        if (orders_.contains(orderId)) {
            return { };
        }

        // Step 1: Check FOK - can we fill the whole thing right now?
        if (timeInForce == TimeInForce::FOK) {
            if (!CanFillFOK(side, price, initialQuantity, orderType)) {
                // FOK order cannot be fully filled, reject it
                return { };
            }
        }

        // Step 2: Match aggressively - try to match the incoming order against the book
        Trades trades = MatchAggressively(order.get());

        // Step 3: Decide on remainder
        bool shouldAddToBook = false;
        if (orderType == OrderType::LIMIT && 
            (timeInForce == TimeInForce::GTC || timeInForce == TimeInForce::GFD)) {
            // Limit GTC/GFD: Add remainder to book
            if (!order->IsFilled()) {
                shouldAddToBook = true;
            }
        }
        // Market/IOC/FOK: Throw away any unfilled remainder (don't add to book)

        if (shouldAddToBook) {
            // Add remainder to book
            std::size_t index;
            if (side == Side::BUY) {
                auto& orders = bids_[price];
                orders.push_back(std::move(order));
                index = orders.size() - 1;
            }
            else {
                auto& orders = asks_[price];
                orders.push_back(std::move(order));
                index = orders.size() - 1;
            }
            orders_.insert({ orderId, OrderEntry{ index, price, side } });
        }
        // Otherwise, order is discarded (fully filled or Market/IOC/FOK with remainder)

        return trades;
    }

    void CancelOrder(OrderId orderId) {
        if (!orders_.contains(orderId)) {
            return;
        }
        const auto& entry = orders_.at(orderId);
        auto price = entry.price_;
        auto side = entry.side_;
        auto orderIndex = entry.index_;
        orders_.erase(orderId);

        if (side == Side::SELL) {
            auto& orders = asks_.at(price);
            orders.erase(orders.begin() + orderIndex);
            UpdateIndicesAfterRemoval(price, Side::SELL, orderIndex);
            if (orders.empty()) {
                asks_.erase(price);
            }
        }
        else {
            auto& orders = bids_.at(price);
            orders.erase(orders.begin() + orderIndex);
            UpdateIndicesAfterRemoval(price, Side::BUY, orderIndex);
            if (orders.empty()) {
                bids_.erase(price);
            }
        }

    }

    Trades ModifyOrder(OrderModify order) {
        OrderId orderId = order.GetOrderId();
        if (!orders_.contains(orderId)) {
            return { };
        }
        const auto& entry = orders_.at(orderId);
        auto price = entry.price_;
        auto side = entry.side_;
        auto index = entry.index_;
        
        // Get the existing order from the deque
        OrderPointer* existingOrderPtr = nullptr;
        if (side == Side::SELL) {
            auto& orders = asks_.at(price);
            existingOrderPtr = &orders[index];
        } else {
            auto& orders = bids_.at(price);
            existingOrderPtr = &orders[index];
        }
        auto& existingOrder = *existingOrderPtr;

        // If Price changes OR Quantity increases, we lose priority (Cancel + New).
        if (order.GetPrice() != existingOrder->GetPrice() || 
            order.GetQuantity() > existingOrder->GetInitialQuantity()) {
            
            CancelOrder(orderId);
            return AddOrder(std::move(order.toOrderPointer()));
        }

        // Resizing Down maintains priority
        if (order.GetQuantity() < existingOrder->GetInitialQuantity()) {
            existingOrder->ResizeQuantity(order.GetQuantity());
        }
        return { };
    }

    std::size_t Size() const { return orders_.size(); }

    OrderbookLevelInfos GetOrderInfos() const {
        LevelInfos bidInfos, askInfos;
        bidInfos.reserve(orders_.size());
        askInfos.reserve(orders_.size());

        auto CreateLevelInfos = [](Price price, const OrderPointers& orders) {
            return LevelInfo{ price, std::accumulate(orders.begin(), orders.end(), (Quantity)0, 
                [](Quantity runningSum, const OrderPointer& order)
                { return runningSum + order->GetRemainingQuantity();}) };
        };
        for (const auto& [price, orders] : bids_) {
            bidInfos.push_back(CreateLevelInfos(price, orders));
        }
        for (const auto& [price, orders] : asks_) {
            askInfos.push_back(CreateLevelInfos(price, orders));
        }
        return OrderbookLevelInfos{ bidInfos, askInfos };
    }

    void Warmup() {
        // Run dummy orders through the Hot Path to prime the Instruction Cache (I-Cache)
        for(int i=0; i<1000; ++i) {
            // Place a Buy and immediately Sell into it to trigger matching logic
            AddOrder(std::make_unique<Order>(1000+i, Side::BUY, 1000, 10, 0, OrderType::MARKET, TimeInForce::GTC));
            AddOrder(std::make_unique<Order>(2000+i, Side::SELL, 1000, 10, 0, OrderType::MARKET, TimeInForce::GTC));
        }
        // Reset book state after warmup
        orders_.clear();
        bids_.clear();
        asks_.clear();
    }

};

#endif