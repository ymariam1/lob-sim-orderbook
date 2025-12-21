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
        Order(OrderId id, Side side, Price price, Quantity quantity, uint64_t timestamp) 
        : orderId_{ id }, side_{ side }, price_{ price }, initialQuantity_{ quantity }, remainingQuantity_{ quantity }, timestamp_{ timestamp } {}

        OrderId GetOrderId() const { return orderId_; }
        Side GetSide() const { return side_; }
        Price GetPrice() const { return price_; }
        Quantity GetInitialQuantity() const { return initialQuantity_; }
        Quantity GetRemainingQuantity() const { return remainingQuantity_; }
        Quantity GetFilledQuantity() const { return filledQuantity_; }
        uint64_t GetTimestamp() const { return timestamp_; }
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
};

using OrderPointer = std::unique_ptr<Order>;
using OrderPointers = std::deque<OrderPointer>;

class OrderModify {
public:
    OrderModify(OrderId orderId, Side side, Price price, Quantity quantity, uint64_t timestamp)
    : orderId_{ orderId }, price_{ price }, side_{ side }, quantity_{ quantity }, timestamp_{ timestamp } {}

    OrderId GetOrderId() const { return orderId_; }
    Side GetSide() const { return side_; }
    Price GetPrice() const { return price_; }
    Quantity GetQuantity() const { return quantity_; }
    uint64_t GetTimestamp() const { return timestamp_; }

    OrderPointer toOrderPointer() const {
        return std::make_unique<Order>(GetOrderId(), GetSide(), GetPrice(), GetQuantity(), GetTimestamp());
    }
private:
    OrderId orderId_;
    Side side_;
    Price price_;
    Quantity quantity_;
    uint64_t timestamp_;
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
        OrderPointers* orders = nullptr;
        if (side == Side::SELL) {
            if (!asks_.contains(price)) return;
            orders = &asks_.at(price);
        } else {
            if (!bids_.contains(price)) return;
            orders = &bids_.at(price);
        }

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
        OrderPointers* orders = nullptr;
        if (side == Side::SELL) {
            if (!asks_.contains(price)) return;
            orders = &asks_.at(price);
        } else {
            if (!bids_.contains(price)) return;
            orders = &bids_.at(price);
        }

        // After removing element at removedIndex, all indices after it shift down by 1
        for (std::size_t i = removedIndex; i < orders->size(); ++i) {
            auto orderId = (*orders)[i]->GetOrderId();
            auto it = orders_.find(orderId);
            if (it != orders_.end()) {
                it->second.index_ = i;
            }
        }
    }

    bool CanMatch(Side side, Price price) const {
        if (side == Side::BUY) {
            if (asks_.empty()) {
                return false;
            }
            return price >= asks_.begin()->first; // checks to see if the price is greater than the best ask
        } else {
            if (bids_.empty()) {
                return false;
            }
            return price <= bids_.begin()->first; // checks to see if the price is less than the best bid
        }
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
                }
                if (ask->IsFilled()) {
                    asks.pop_front();
                    orders_.erase(askOrderId);
                    UpdateIndicesAfterFrontRemoval(askPrice, Side::SELL);
                }

                if (bids.empty()) {
                    bids_.erase(bidPrice);
                }
                if (asks.empty()) {
                    asks_.erase(askPrice);
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
        
        if (orders_.contains(orderId)) {
            return { };
        }

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
        return MatchOrders();
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
            AddOrder(std::make_unique<Order>(1000+i, Side::BUY, 1000, 10, 0));
            AddOrder(std::make_unique<Order>(2000+i, Side::SELL, 1000, 10, 0));
        }
        // Reset book state after warmup
        orders_.clear();
        bids_.clear();
        asks_.clear();
    }

};

#endif