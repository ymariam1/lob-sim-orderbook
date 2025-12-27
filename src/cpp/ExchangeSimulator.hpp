#ifndef EXCHANGE_SIMULATOR_HPP
#define EXCHANGE_SIMULATOR_HPP

#include <vector>
#include <set>
#include <algorithm>
#include "OrderBook.hpp"

// Pending action types for the latency queue
enum class PendingActionType {
    ADD_ORDER,
    CANCEL_ORDER
};

// A wrapper to simulate network and computation delay
class ExchangeSimulator {
public:
    ExchangeSimulator(std::shared_ptr<Orderbook> book) : book_(std::move(book)) {}
    
    // Track which order IDs belong to the agent (for fill detection)
    void RegisterAgentOrderId(OrderId id) {
        agentOrderIds_.insert(id);
    }
    
    // Get and clear accumulated agent fills
    const std::vector<Trade>& GetAgentFills() const { return agentFills_; }
    void ClearAgentFills() { agentFills_.clear(); }
    std::size_t GetAgentFillCount() const { return agentFills_.size(); }

    // 1. Ingest Historical Data (The "World" moves instantly)
    void ProcessHistoricalEvent(OrderPointer order) {
        currentTime_ = order->GetTimestamp();
        
        // Before processing the market, check if any Agent orders have "arrived"
        ProcessPendingAgentActions();
        
        // Process the historical order and capture any trades
        Trades trades = book_->AddOrder(std::move(order));
        
        // Check if any of these trades involve our agent's resting orders
        for (const auto& trade : trades) {
            // Check both bid and ask side of the trade for agent order IDs
            if (agentOrderIds_.count(trade.GetBidTrade().orderId_) > 0 ||
                agentOrderIds_.count(trade.GetAskTrade().orderId_) > 0) {
                agentFills_.push_back(trade);
            }
        }
    }

    // 2. The Agent places an order (BUT it gets delayed)
    void PlaceAgentOrder(OrderPointer order, uint64_t inferenceLatencyNs) {
        uint64_t arrivalTime = currentTime_ + inferenceLatencyNs;
        
        // Register the order ID so we can track fills
        RegisterAgentOrderId(order->GetOrderId());
        
        // Add to heap using vector + heap algorithms
        pendingActions_.push_back({arrivalTime, PendingActionType::ADD_ORDER, std::move(order), 0});
        std::push_heap(pendingActions_.begin(), pendingActions_.end(), ActionComparator{});
    }

    // 3. The Agent cancels an order (BUT it gets delayed too!)
    void CancelAgentOrder(OrderId orderId, uint64_t inferenceLatencyNs) {
        uint64_t arrivalTime = currentTime_ + inferenceLatencyNs;
        
        // Add cancel request to heap
        pendingActions_.push_back({arrivalTime, PendingActionType::CANCEL_ORDER, nullptr, orderId});
        std::push_heap(pendingActions_.begin(), pendingActions_.end(), ActionComparator{});
    }

    // 4. Check if agent actions are ready to execute
    void ProcessPendingAgentActions() {
        while (!pendingActions_.empty()) {
            // Check the front of the heap (min element)
            if (pendingActions_.front().arrivalTime <= currentTime_) {
                // 1. Move the top to the back (standard heap removal)
                std::pop_heap(pendingActions_.begin(), pendingActions_.end(), ActionComparator{});
                
                // 2. Now the element is at the back and is MUTABLE - we can safely move it
                PendingAction& action = pendingActions_.back();
                
                if (action.actionType == PendingActionType::ADD_ORDER) {
                    // Process the order and capture any immediate fills (aggressive)
                    Trades trades = book_->AddOrder(std::move(action.order));
                    
                    // All trades from agent orders are agent fills
                    for (const auto& trade : trades) {
                        agentFills_.push_back(trade);
                    }
                } else if (action.actionType == PendingActionType::CANCEL_ORDER) {
                    book_->CancelOrder(action.cancelOrderId);
                }
                
                // 3. Remove from vector
                pendingActions_.pop_back();
            } else {
                break; // Next action is still "in transit"
            }
        }
    }
    
    // Legacy alias for backwards compatibility
    void ProcessPendingAgentOrders() {
        ProcessPendingAgentActions();
    }

    uint64_t GetCurrentTime() const { return currentTime_; }
    
    // Force the clock forward (for when data is sparse or market is quiet)
    // This is CRITICAL for correct baseline execution timing
    void SetCurrentTime(uint64_t t) {
        if (t > currentTime_) {
            currentTime_ = t;
            // Also process any pending agent actions that have now "arrived"
            ProcessPendingAgentActions();
        }
    }
    
    // Get access to the orderbook for state queries
    std::shared_ptr<Orderbook> GetOrderbook() const { return book_; }
    
    // Get number of pending agent actions
    std::size_t GetPendingActionCount() const { return pendingActions_.size(); }

private:
    struct PendingAction {
        uint64_t arrivalTime;
        PendingActionType actionType;
        OrderPointer order;          // For ADD_ORDER
        OrderId cancelOrderId;       // For CANCEL_ORDER
    };
    
    // Comparator for min-heap (earliest arrival first)
    struct ActionComparator {
        bool operator()(const PendingAction& a, const PendingAction& b) const {
            return a.arrivalTime > b.arrivalTime;  // Greater-than for min-heap
        }
    };

    std::shared_ptr<Orderbook> book_;
    uint64_t currentTime_ = 0;
    std::vector<PendingAction> pendingActions_;  // Min-heap using vector + heap algorithms
    
    // Agent order tracking for fill detection
    std::set<OrderId> agentOrderIds_;       // Which orders belong to the agent
    std::vector<Trade> agentFills_;          // Accumulated fills on agent orders
};

#endif // EXCHANGE_SIMULATOR_HPP
