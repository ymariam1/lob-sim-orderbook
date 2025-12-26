#ifndef ORDER_BOOK_HPP
#define ORDER_BOOK_HPP
#include <cstdint>
#include <map>
#include <vector>
#include <numeric>
#include <format>
#include <list>
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
using OrderPointers = std::list<OrderPointer>;

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
    using OrderIterator = OrderPointers::iterator;
    
    struct OrderEntry {
        OrderIterator iterator_;
        Price price_;
        Side side_;
    };

    std::map<Price, OrderPointers, std::greater<Price>> bids_;
    std::map<Price, OrderPointers, std::less<Price>> asks_;
    std::unordered_map<OrderId, OrderEntry> orders_;
    
    // Incremental volume tracking for O(1) GetOrderInfos
    std::map<Price, Quantity, std::greater<Price>> bid_volumes_;
    std::map<Price, Quantity, std::less<Price>> ask_volumes_;

    // Helper to update volume tracking (incremental updates for O(1) GetOrderInfos)
void UpdateVolume(Price price, Side side, Quantity delta) {
        if (side == Side::BUY) {
            bid_volumes_[price] += delta;
            if (bid_volumes_[price] <= 0) bid_volumes_.erase(price);
        } else {
            ask_volumes_[price] += delta;
            if (ask_volumes_[price] <= 0) ask_volumes_.erase(price);
        }
    }

    // Check if a GFD order is expired based on a reference timestamp
    // IMPORTANT: Assumes timestamps are in SECONDS since epoch
    // If your timestamps are in nanoseconds, use: 86400000000000ULL
    // If your timestamps are in microseconds, use: 86400000000ULL
    // If your timestamps are in milliseconds, use: 86400000ULL
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
    // Returns true if an order was removed, false otherwise
    // NOTE: Does NOT erase the map entry - caller must handle that
    bool RemoveExpiredGFDFromFront(OrderPointers* orders, Price price, Side side, uint64_t referenceTimestamp) {
        if (!orders || orders->empty()) {
            return false;
        }
        
        auto& frontOrder = orders->front();
        if (frontOrder->GetTimeInForce() == TimeInForce::GFD && 
            IsGFDExpired(frontOrder->GetTimestamp(), referenceTimestamp)) {
            OrderId expiredOrderId = frontOrder->GetOrderId();
            Quantity expiredQty = frontOrder->GetRemainingQuantity();
            orders->pop_front();
            orders_.erase(expiredOrderId);
            // Update volume tracking
            UpdateVolume(price, side, -expiredQty);
            return true;
        }
        return false;
    }

    // Check if FOK order can be fully filled (pre-calculation)
    bool CanFillFOK(Side side, Price price, Quantity quantity, int orderType) const {
        Quantity availableQuantity = 0;
        
        if (side == Side::BUY) {
            if (ask_volumes_.empty()) {
                return false;
            }
            // For market orders, sum all available ask volumes
            // For limit orders, sum ask volumes at or below the limit price
            for (const auto& [askPrice, volume] : ask_volumes_) {
                if (orderType == OrderType::LIMIT && askPrice > price) {
                    break; // Limit order: can't match above limit price
                }
                availableQuantity += volume;
                if (availableQuantity >= quantity) {
                    return true;
                }
            }
        } else { // SELL
            if (bid_volumes_.empty()) {
                return false;
            }
            // For market orders, sum all available bid volumes
            // For limit orders, sum bid volumes at or above the limit price
            for (const auto& [bidPrice, volume] : bid_volumes_) {
                if (orderType == OrderType::LIMIT && bidPrice < price) {
                    break; // Limit order: can't match below limit price
                }
                availableQuantity += volume;
                if (availableQuantity >= quantity) {
                    return true;
                }
            }
        }
        
        return availableQuantity >= quantity;
    }

    // Aggressively match an incoming order against the book
    Trades Match(Order* incomingOrder) {
        Trades trades;
        Side side = incomingOrder->GetSide();
        Side oppositeSide = side == Side::BUY ? Side::SELL : Side::BUY;
        Price price = incomingOrder->GetPrice();
        int orderType = incomingOrder->GetOrderType();
        OrderId incomingOrderId = incomingOrder->GetOrderId();
        
        while (!incomingOrder->IsFilled()) {
            // 1. Check if book is empty
            if (side == Side::BUY && asks_.empty()) break;
            if (side == Side::SELL && bids_.empty()) break;

            // 2. Get best level safely using iterator
            auto bestIt = (side == Side::BUY) ? asks_.begin() : bids_.begin();
            Price levelPrice = bestIt->first;
            OrderPointers& levelOrders = bestIt->second;

            // 3. Check price limits for limit orders
            if (orderType == OrderType::LIMIT) {
                if (side == Side::BUY && levelPrice > price) break;
                if (side == Side::SELL && levelPrice < price) break;
            }

            // 4. Match loop for this level
            while (!levelOrders.empty() && !incomingOrder->IsFilled()) {
                // Skip expired GFD orders
                while (RemoveExpiredGFDFromFront(&levelOrders, levelPrice, oppositeSide, incomingOrder->GetTimestamp())) {
                    // Continue removing expired orders until we find a valid one or run out
                }
                
                if (levelOrders.empty()) {
                    break; // No valid orders at this level
                }
                
                auto& bookOrder = levelOrders.front();
                Quantity matchQuantity = std::min(
                    incomingOrder->GetRemainingQuantity(),
                    bookOrder->GetRemainingQuantity()
                );
                
                // Execution price: Always use the maker's (resting order's) price
                Price executionPrice = bookOrder->GetPrice();
                
                // Fill both orders
                incomingOrder->Fill(matchQuantity);
                bookOrder->Fill(matchQuantity);
                
                OrderId bookOrderId = bookOrder->GetOrderId();
                
                // Record trade
                if (side == Side::BUY) {
                    trades.push_back(Trade{
                        TradeInfo{ incomingOrderId, executionPrice, matchQuantity },
                        TradeInfo{ bookOrderId, executionPrice, matchQuantity }
                    });
                } else {
                    trades.push_back(Trade{
                        TradeInfo{ bookOrderId, executionPrice, matchQuantity },
                        TradeInfo{ incomingOrderId, executionPrice, matchQuantity }
                    });
                }
                
                // Update volume tracking (subtract the quantity that was filled)
                UpdateVolume(levelPrice, oppositeSide, -matchQuantity);
                
                // Remove filled order from book
                if (bookOrder->IsFilled()) {
                    orders_.erase(bookOrderId);
                    levelOrders.pop_front();
                }
            }

            // 5. Clean up level if empty (safe because we break loop and re-evaluate next iter)
            if (levelOrders.empty()) {
                if (side == Side::BUY) {
                    asks_.erase(bestIt);
                } else {
                    bids_.erase(bestIt);
                }
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

        // Check FOK - can we fill the whole thing right now?
        if (timeInForce == TimeInForce::FOK) {
            if (!CanFillFOK(side, price, initialQuantity, orderType)) {
                // FOK order cannot be fully filled, reject it
                return { };
            }
        }

        // Match incoming order with the book
        Trades trades = Match(order.get());

        // Decide on remainder
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
            OrderIterator it;
            Quantity remainingQty = order->GetRemainingQuantity();
            
            if (side == Side::BUY) {
                auto& orders = bids_[price];
                orders.push_back(std::move(order));
                it = std::prev(orders.end());
            }
            else {
                auto& orders = asks_[price];
                orders.push_back(std::move(order));
                it = std::prev(orders.end());
            }
            orders_.insert({ orderId, OrderEntry{ it, price, side } });
            
            // Update volume tracking
            UpdateVolume(price, side, remainingQty);
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
        auto orderIt = entry.iterator_;
        
        // Get quantity for volume tracking before erasing
        Quantity cancelledQty = (*orderIt)->GetRemainingQuantity();
        
        orders_.erase(orderId);

        if (side == Side::SELL) {
            auto& orders = asks_.at(price);
            orders.erase(orderIt);
            if (orders.empty()) {
                asks_.erase(price);
            }
        }
        else {
            auto& orders = bids_.at(price);
            orders.erase(orderIt);
            if (orders.empty()) {
                bids_.erase(price);
            }
        }
        
        // Update volume tracking
        UpdateVolume(price, side, -cancelledQty);
    }

    Trades ModifyOrder(OrderModify order) {
        OrderId orderId = order.GetOrderId();
        if (!orders_.contains(orderId)) {
            return { };
        }
        const auto& entry = orders_.at(orderId);
        auto price = entry.price_;
        auto side = entry.side_;
        auto orderIt = entry.iterator_;
        
        // Get the existing order from the list
        auto& existingOrder = *orderIt;

        // If Price changes OR Quantity increases, we lose priority (Cancel + New).
        if (order.GetPrice() != existingOrder->GetPrice() || 
            order.GetQuantity() > existingOrder->GetInitialQuantity()) {
            
            CancelOrder(orderId);
            return AddOrder(std::move(order.toOrderPointer()));
        }

        // Resizing Down maintains priority
        if (order.GetQuantity() < existingOrder->GetInitialQuantity()) {
            Quantity oldQty = existingOrder->GetRemainingQuantity();
            existingOrder->ResizeQuantity(order.GetQuantity());
            Quantity newQty = existingOrder->GetRemainingQuantity();
            // Update volume tracking for the difference
            UpdateVolume(price, side, newQty - oldQty);
        }
        return { };
    }

    std::size_t Size() const { return orders_.size(); }

    OrderbookLevelInfos GetOrderInfos() const {
        LevelInfos bidInfos, askInfos;
        bidInfos.reserve(bid_volumes_.size());
        askInfos.reserve(ask_volumes_.size());

        // Use incremental volume tracking for O(1) lookup instead of O(N) calculation
        for (const auto& [price, volume] : bid_volumes_) {
            bidInfos.push_back(LevelInfo{ price, volume });
        }
        for (const auto& [price, volume] : ask_volumes_) {
            askInfos.push_back(LevelInfo{ price, volume });
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
        bid_volumes_.clear();
        ask_volumes_.clear();
    }

};

#endif