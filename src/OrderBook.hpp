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

using OrderPointer = std::shared_ptr<Order>;
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
        return std::make_shared<Order>(GetOrderId(), GetSide(), GetPrice(), GetQuantity(), GetTimestamp());
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
        OrderPointer order_{ nullptr };
        std::size_t index_;
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
                bid->Fill(quantity);
                ask->Fill(quantity);

                if (bid->IsFilled()) {
                    bids.pop_front();
                    orders_.erase(bid->GetOrderId());
                    UpdateIndicesAfterFrontRemoval(bidPrice, Side::BUY);
                }
                if (ask->IsFilled()) {
                    asks.pop_front();
                    orders_.erase(ask->GetOrderId());
                    UpdateIndicesAfterFrontRemoval(askPrice, Side::SELL);
                }

                if (bids.empty()) {
                    bids_.erase(bidPrice);
                }
                if (asks.empty()) {
                    asks_.erase(askPrice);
                }
                
                trades.push_back(Trade{ 
                    TradeInfo{ bid->GetOrderId(), executionPrice, quantity }, 
                    TradeInfo{ ask->GetOrderId(), executionPrice, quantity } 
                });
            }
        }
        return trades;
    }
public:

    Trades AddOrder(OrderPointer order) {
        if (orders_.contains(order->GetOrderId())) {
            return { };
        }

        std::size_t index;
        if (order->GetSide() == Side::BUY) {
            auto& orders = bids_[order->GetPrice()];
            orders.push_back(order);
            index = orders.size() - 1;
        }
        else {
            auto& orders = asks_[order->GetPrice()];
            orders.push_back(order);
            index = orders.size() - 1;
        }

        orders_.insert({ order->GetOrderId(), OrderEntry{ order, index} });
        return MatchOrders();
    }

    void CancelOrder(OrderId orderId) {
        if (!orders_.contains(orderId)) {
            return;
        }
        const auto& [order, orderIndex] = orders_.at(orderId);
        auto price = order->GetPrice();
        auto side = order->GetSide();
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
        if (!orders_.contains(order.GetOrderId())) {
            return { };
        }
        const auto& [existingOrder, existingIndex] = orders_.at(order.GetOrderId());

        // If Price changes OR Quantity increases, we lose priority (Cancel + New).
        if (order.GetPrice() != existingOrder->GetPrice() || 
            order.GetQuantity() > existingOrder->GetInitialQuantity()) {
            
            CancelOrder(order.GetOrderId());
            return AddOrder(order.toOrderPointer());
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
};

#endif