#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Automatically converts std::vector, std::map to Python lists/dicts
#include "OrderBook.hpp"
#include "ExchangeSimulator.hpp"
#include "DataLoader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lob_sim, m) {
    m.doc() = "High-frequency Orderbook and Exchange Simulator - Python bindings";

    // -----------------------------------------------------------------------
    // 1. Enums (So you can use Side.BUY in Python)
    // -----------------------------------------------------------------------
    py::enum_<Side>(m, "Side")
        .value("BUY", Side::BUY)
        .value("SELL", Side::SELL)
        .export_values();

    py::enum_<OrderType>(m, "OrderType")
        .value("MARKET", OrderType::MARKET)
        .value("LIMIT", OrderType::LIMIT)
        .export_values();

    py::enum_<TimeInForce>(m, "TimeInForce")
        .value("GTC", TimeInForce::GTC)
        .value("FOK", TimeInForce::FOK)
        .value("IOC", TimeInForce::IOC)
        .value("GFD", TimeInForce::GFD)
        .export_values();

    // -----------------------------------------------------------------------
    // 2. Data Structures (LevelInfo, Trades)
    // -----------------------------------------------------------------------
    py::class_<LevelInfo>(m, "LevelInfo")
        .def_readonly("price", &LevelInfo::price)
        .def_readonly("quantity", &LevelInfo::quantity)
        .def("__repr__", [](const LevelInfo &a) {
            return "<LevelInfo price=" + std::to_string(a.price) + 
                   " quantity=" + std::to_string(a.quantity) + ">";
        });

    // Note: pybind11/stl.h handles std::vector<LevelInfo> automatically!

    py::class_<OrderbookLevelInfos>(m, "OrderbookLevelInfos")
        .def("GetBids", &OrderbookLevelInfos::GetBids)
        .def("GetAsks", &OrderbookLevelInfos::GetAsks)
        .def("__repr__", [](const OrderbookLevelInfos &a) {
            return "<OrderbookLevelInfos bids=" + std::to_string(a.GetBids().size()) + 
                   " asks=" + std::to_string(a.GetAsks().size()) + ">";
        });

    py::class_<TradeInfo>(m, "TradeInfo")
        .def_readonly("orderId", &TradeInfo::orderId_)
        .def_readonly("price", &TradeInfo::price_)
        .def_readonly("quantity", &TradeInfo::quantity_)
        .def("__repr__", [](const TradeInfo &a) {
            return "<TradeInfo orderId=" + std::to_string(a.orderId_) +
                   " price=" + std::to_string(a.price_) +
                   " quantity=" + std::to_string(a.quantity_) + ">";
        });

    py::class_<Trade>(m, "Trade")
        .def("GetBidTrade", &Trade::GetBidTrade)
        .def("GetAskTrade", &Trade::GetAskTrade)
        .def("__repr__", [](const Trade &t) {
            return "<Trade bid_price=" + std::to_string(t.GetBidTrade().price_) +
                   " ask_price=" + std::to_string(t.GetAskTrade().price_) +
                   " quantity=" + std::to_string(t.GetBidTrade().quantity_) + ">";
        });

    // -----------------------------------------------------------------------
    // 3. The Order Class
    // Note: For pybind11 2.x compatibility, we avoid smart_holder and unique_ptr
    // in the Python-facing API. Instead, we use shared_ptr for the Order class.
    // -----------------------------------------------------------------------
    py::class_<Order, std::shared_ptr<Order>>(m, "Order")
        .def(py::init<OrderId, Side, Price, Quantity, uint64_t, int, int>(),
             py::arg("orderId"),
             py::arg("side"),
             py::arg("price"),
             py::arg("quantity"),
             py::arg("timestamp"),
             py::arg("orderType"),
             py::arg("timeInForce"))
        .def("GetOrderId", &Order::GetOrderId)
        .def("GetSide", &Order::GetSide)
        .def("GetPrice", &Order::GetPrice)
        .def("GetInitialQuantity", &Order::GetInitialQuantity)
        .def("GetRemainingQuantity", &Order::GetRemainingQuantity)
        .def("GetFilledQuantity", &Order::GetFilledQuantity)
        .def("GetTimestamp", &Order::GetTimestamp)
        .def("GetOrderType", &Order::GetOrderType)
        .def("GetTimeInForce", &Order::GetTimeInForce)
        .def("IsFilled", &Order::IsFilled)
        .def("__repr__", [](const Order &o) {
            std::string sideStr = (o.GetSide() == Side::BUY) ? "BUY" : "SELL";
            return "<Order id=" + std::to_string(o.GetOrderId()) +
                   " side=" + sideStr +
                   " price=" + std::to_string(o.GetPrice()) +
                   " qty=" + std::to_string(o.GetRemainingQuantity()) + "/" +
                   std::to_string(o.GetInitialQuantity()) + ">";
        });

    py::class_<OrderModify>(m, "OrderModify")
        .def(py::init<OrderId, Side, Price, Quantity, uint64_t, int, int>(),
             py::arg("orderId"),
             py::arg("side"),
             py::arg("price"),
             py::arg("quantity"),
             py::arg("timestamp"),
             py::arg("orderType"),
             py::arg("timeInForce"));

    // -----------------------------------------------------------------------
    // 4. The Orderbook Class
    // -----------------------------------------------------------------------
    py::class_<Orderbook, std::shared_ptr<Orderbook>>(m, "Orderbook")
        .def(py::init<>())
        
        // For pybind11 2.x: Accept shared_ptr and convert to unique_ptr internally
        // We create a copy of the Order to transfer ownership to C++
        .def("AddOrder", [](Orderbook& self, std::shared_ptr<Order> order) {
            // Create a new unique_ptr with a copy of the order
            auto orderCopy = std::make_unique<Order>(
                order->GetOrderId(),
                order->GetSide(),
                order->GetPrice(),
                order->GetInitialQuantity(),
                order->GetTimestamp(),
                order->GetOrderType(),
                order->GetTimeInForce()
            );
            return self.AddOrder(std::move(orderCopy));
        }, py::arg("order"),
           "Add an order to the book. Note: A copy of the order is made internally.")
        
        .def("CancelOrder", &Orderbook::CancelOrder, py::arg("orderId"))
        .def("ModifyOrder", &Orderbook::ModifyOrder, py::arg("order"))
        .def("GetOrderInfos", &Orderbook::GetOrderInfos)
        .def("Size", &Orderbook::Size)
        .def("Warmup", &Orderbook::Warmup)
        .def("__repr__", [](const Orderbook &b) {
            return "<Orderbook orders=" + std::to_string(b.Size()) + ">";
        });

    // -----------------------------------------------------------------------
    // 5. The Exchange Simulator
    // -----------------------------------------------------------------------
    py::class_<ExchangeSimulator>(m, "ExchangeSimulator")
        .def(py::init<std::shared_ptr<Orderbook>>(), py::arg("orderbook"))
        
        // For pybind11 2.x: Accept shared_ptr and convert to unique_ptr internally
        .def("ProcessHistoricalEvent", [](ExchangeSimulator& self, std::shared_ptr<Order> order) {
            auto orderCopy = std::make_unique<Order>(
                order->GetOrderId(),
                order->GetSide(),
                order->GetPrice(),
                order->GetInitialQuantity(),
                order->GetTimestamp(),
                order->GetOrderType(),
                order->GetTimeInForce()
            );
            self.ProcessHistoricalEvent(std::move(orderCopy));
        }, py::arg("order"),
           "Process a historical market event. Note: A copy of the order is made internally.")
        
        .def("PlaceAgentOrder", [](ExchangeSimulator& self, std::shared_ptr<Order> order, uint64_t latency) {
            auto orderCopy = std::make_unique<Order>(
                order->GetOrderId(),
                order->GetSide(),
                order->GetPrice(),
                order->GetInitialQuantity(),
                order->GetTimestamp(),
                order->GetOrderType(),
                order->GetTimeInForce()
            );
            self.PlaceAgentOrder(std::move(orderCopy), latency);
        }, py::arg("order"), py::arg("latencyNs"),
           "Place an agent order with network latency simulation. Note: A copy of the order is made internally.")
        
        // Cancel with latency simulation
        .def("CancelAgentOrder", &ExchangeSimulator::CancelAgentOrder,
             py::arg("orderId"), py::arg("latencyNs"),
             "Cancel an agent order with network latency simulation.")
        
        .def("ProcessPendingAgentOrders", &ExchangeSimulator::ProcessPendingAgentOrders)
        .def("ProcessPendingAgentActions", &ExchangeSimulator::ProcessPendingAgentActions)
        .def("GetCurrentTime", &ExchangeSimulator::GetCurrentTime)
        .def("SetCurrentTime", &ExchangeSimulator::SetCurrentTime,
             py::arg("timestamp"),
             "Force the clock forward to a specific time (for sparse data or quiet markets).\n"
             "CRITICAL: This ensures time flows even when no market events occur.")
        .def("GetOrderbook", &ExchangeSimulator::GetOrderbook)
        .def("GetPendingActionCount", &ExchangeSimulator::GetPendingActionCount)
        // Agent fill tracking - for proper passive order fill detection
        .def("GetAgentFills", &ExchangeSimulator::GetAgentFills,
             "Get list of trades that filled agent orders (both aggressive and passive fills)")
        .def("ClearAgentFills", &ExchangeSimulator::ClearAgentFills,
             "Clear the agent fills buffer after reading")
        .def("GetAgentFillCount", &ExchangeSimulator::GetAgentFillCount,
             "Get the number of pending agent fills")
        .def("__repr__", [](const ExchangeSimulator &e) {
            return "<ExchangeSimulator time=" + std::to_string(e.GetCurrentTime()) + 
                   " pending=" + std::to_string(e.GetPendingActionCount()) + ">";
        });

    // -----------------------------------------------------------------------
    // 6. Helper Factory Functions (Optional but convenient)
    // For pybind11 2.x: Return shared_ptr instead of unique_ptr
    // -----------------------------------------------------------------------
    m.def("create_limit_order", [](OrderId id, Side side, Price price, Quantity qty, 
                                    uint64_t timestamp, TimeInForce tif) {
        return std::make_shared<Order>(id, side, price, qty, timestamp, OrderType::LIMIT, tif);
    }, py::arg("orderId"), py::arg("side"), py::arg("price"), py::arg("quantity"),
       py::arg("timestamp"), py::arg("timeInForce") = TimeInForce::GTC,
       "Create a LIMIT order");

    m.def("create_market_order", [](OrderId id, Side side, Quantity qty, 
                                     uint64_t timestamp, TimeInForce tif) {
        return std::make_shared<Order>(id, side, 0, qty, timestamp, OrderType::MARKET, tif);
    }, py::arg("orderId"), py::arg("side"), py::arg("quantity"),
       py::arg("timestamp"), py::arg("timeInForce") = TimeInForce::IOC,
       "Create a MARKET order (price is ignored)");

    // -----------------------------------------------------------------------
    // 7. Data Loader - CSV to Exchange Bridge
    // -----------------------------------------------------------------------
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<const std::string&, uint64_t>(), 
             py::arg("filename"),
             py::arg("timestamp_unit_ns") = 1000000000ULL,
             "Create a DataLoader from a CSV file.\n\n"
             "Args:\n"
             "  filename: Path to CSV file\n"
             "  timestamp_unit_ns: Conversion factor from CSV timestamp to nanoseconds.\n"
             "    - 1 if CSV is in nanoseconds\n"
             "    - 1000 if CSV is in microseconds\n"
             "    - 1000000 if CSV is in milliseconds\n"
             "    - 1000000000 if CSV is in seconds (default)")
        .def("PumpToExchange", &DataLoader::PumpToExchange,
             py::arg("exchange"), py::arg("duration_ns"),
             "Reads CSV rows and pushes them to the exchange until time advances by duration_ns. "
             "Returns the number of events processed.")
        .def("HasMoreData", &DataLoader::HasMoreData,
             "Check if there is more data to read from the CSV")
        .def("PeekNextTimestamp", &DataLoader::PeekNextTimestamp,
             "Peek at the timestamp of the next event (in original CSV units)")
        .def("PeekNextTimestampNs", &DataLoader::PeekNextTimestampNs,
             "Peek at the timestamp of the next event in nanoseconds")
        .def("GetTotalEventsProcessed", &DataLoader::GetTotalEventsProcessed,
             "Get the total number of events processed so far")
        .def("GetTimestampUnitNs", &DataLoader::GetTimestampUnitNs,
             "Get the timestamp conversion factor (CSV units to nanoseconds)")
        .def("__repr__", [](const DataLoader &d) {
            return "<DataLoader events_processed=" + std::to_string(d.GetTotalEventsProcessed()) + 
                   " has_more=" + (d.HasMoreData() ? "True" : "False") + ">";
        });
}
