"""
Matching engine for Shenzhen Stock Exchange tick data.

Replays the exchange's order and trade feeds to reconstruct the limit order
book, the execution stream, and order-book snapshots. Supports the normal
continuous-auction rule, the ChiNext "freeze" (鸽笼) price-cage rule, and the
opening call auction.

Author: Yufan Chen
Last update: 2023.3.10

Changelog:
    2023.3.10 added the execute_level_num parameter
    2023.2.13 fixed a crash when a day contains no market orders;
              fixed the engine.rule selection;
              defreeze is now re-checked after order cancellation;
              market orders that fully fill or cancel get a proper type;
              snap_total_df rows now carry order_id and typ
    2023.2.12 execute_df stores the pre-trade snapshot via last_snap_shot
    2023.2.11 four snapshot-generation modes for snap_total_df
    2023.2.10 added the snap_level_num parameter to main_matching_process
    2023.2.9  intraday time is taken via string slicing because float
              "% 1e9" loses precision

Data note: 20190104, 20190314 and 20190315 are known-bad days in the raw
data; do not use them.
"""

import heapq
import os
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

import numpy as np
import pandas as pd


def _my_round(x, n: int) -> float:
    """Round ``x`` half-up to ``n`` decimal places.

    The 1e-8 nudge keeps values like 2.675 (stored as 2.67499...) from
    rounding down due to binary float representation.
    """
    return float(Decimal(str(x + 1e-8)).quantize(Decimal('1.' + '0' * n), ROUND_HALF_UP))


class Side(Enum):
    """ A class to mark the buy/sell side of the trading process.

    BUY: 1; SELL: -1
    """
    BUY = 1
    SELL = -1


class Order:
    """ A class to construct a order.

    Attributes:
        order_id: int
            ID of the order.
        price: float
            Quote price of the order. (For the market order, the price maybe be regenerated.)
        qty: int
            Quote quantity of the order.
        typ: str
            Type of the order. (2: limit order, U: best price on own side 本方最优,
            MU: best price on opposite side 对方最优, M: market order,
            5C: fill against the top five levels then cancel the rest 五档成交剩余撤销)
        cum_qty: int
            The quantity that the order has been executed.
        leaves_qty: int
            The quantity that the order has not been executed.
        side: Side
            The side of the order.
        time: int or datetime.datetime
            The time when the order was submit.
        origin: int
            The origin of the order. (Default: -1, from the exchange data)
    """

    def __init__(self, order_id: int, price: float, qty: int, typ: str, side: Side, time, origin=-1):
        self.order_id = order_id
        self.origin = origin
        self.price = price
        self.qty = qty
        self.typ = typ
        self.cum_qty = 0
        self.leaves_qty = qty
        self.side = side
        self.time = time

    def __repr__(self) -> str:
        order_str = f"Order ID:{self.order_id}, Price:{self.price}, Qty:{self.qty}, Side:{self.side}, Time:{self.time}, Type: {self.typ} "
        return order_str


class OrderBook:
    """ A class to construct a order book (for construct a limit order book and a freeze order book).

    Attributes:
        bids: dict
            A dict to store the order book on the bid side.
            Example: {price1:[Order1, Order2], price2:[Order3], ...}
        asks: dict
            A dict to store the order book on the ask side.
        order_id_map: dict
            A dict to store all the order in the order book, indexed by their order ids.
            Example: {order_id1: Order1, order_id2: Order2}
        best_bid_price: float
        best_ask_price: float
    """

    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.bids_snap = {}
        self.asks_snap = {}
        self.order_id_map = {}
        self.best_bid_price = None
        self.best_ask_price = None

    def add_order(self, order: Order):
        """ Add a new order to its corresponding position of the order book.

        Parameters
        ----------
        order: Order
            The order to be added.

        Returns
        -------

        """
        price, side, order_id = order.price, order.side, order.order_id
        qty = order.qty

        # add the order into the order book
        if side == Side.BUY:
            depth = self.bids.setdefault(price, [])
        else:
            depth = self.asks.setdefault(price, [])
        depth.append(order)

        # update the aggregated size at this price level
        if side == Side.BUY:
            snap_qty = self.bids_snap.setdefault(price, 0)
            self.bids_snap[price] = snap_qty + qty
        else:
            snap_qty = self.asks_snap.setdefault(price, 0)
            self.asks_snap[price] = snap_qty + qty

        # add the order into the order_id_map
        self.order_id_map[order_id] = order

        return None

    def cancel_order(self, order_id: int):
        """ Cancel a existing order from the order book.

        Parameters
        ----------
        order_id: int
            The order id of the order to be cancelled.

        Returns
        -------

        """
        order = self.order_id_map[order_id]
        price, side, order_id = order.price, order.side, order.order_id
        qty = order.leaves_qty

        # find the corresponding price level
        if side == Side.BUY:
            assert price in self.bids.keys(), \
                f"Order price {price:.2f} of order id {order_id} is not in the bid price depth!"
            price_level = self.bids[price]
        else:
            assert price in self.asks.keys(), \
                f"Order price {price:.2f} of order id {order_id} is not in the ask price depth!"
            price_level = self.asks[price]

        # delete the order
        index = 0
        price_level_len = len(price_level)
        while index < price_level_len:
            if price_level[index].order_id == order_id:
                del price_level[index]
                break
            index += 1
        assert (index != price_level_len), f"Cannot find the order ID. {order_id}!"

        # delete empty price level
        if side == Side.BUY and len(self.bids[price]) == 0:
            del self.bids[price]
        elif side == Side.SELL and len(self.asks[price]) == 0:
            del self.asks[price]

        # update the snap dict
        if side == Side.BUY:
            self.bids_snap[price] = self.bids_snap[price] - qty
            if self.bids_snap[price] == 0:
                del self.bids_snap[price]
        elif side == Side.SELL:
            self.asks_snap[price] = self.asks_snap[price] - qty
            if self.asks_snap[price] == 0:
                del self.asks_snap[price]

        del self.order_id_map[order_id]

        return None


class NormalOrderBook(OrderBook):
    """ A class to construct a limit order book.

    Attributes:
    bids: dict
        A dict to store the order book on the bid side.
        Example: {price1:[Order1, Order2], price2:[Order3], ...}
    asks: dict
        A dict to store the order book on the ask side.
    order_id_map: dict
        A dict to store all the order in the order book, indexed by their order ids.
        Example: {order_id1: Order1, order_id2: Order2}
    best_bid_price: float
    best_ask_price: float
    execute_total_list: list
        save the execute events and related information.
    snap_total_list: list
        save the snap shots in the matching processes.
    last_snap_shot: dict
        save the snap before the last order adding.
    """

    def __init__(self):
        super().__init__()
        self.last_snap_shot = {}
        self.execute_total_list = []
        self.snap_total_list = []
        self.snap_total_list_new = []

    def __repr__(self, level_num=10) -> str:
        snap_shot = self.gen_snap_shot(level_num=level_num)
        snap_shot_str = "".join(["{}\t{}\t{}\t{}\n".format(snap_shot['BidPX' + str(i + 1)],
                                                           snap_shot['BidSize' + str(i + 1)],
                                                           snap_shot['OfferPX' + str(i + 1)],
                                                           snap_shot['OfferSize' + str(i + 1)])
                                 for i in range(level_num)])
        return snap_shot_str

    def update_best_price(self):
        """ Update the best bid and ask price of the limit order book.

        Returns
        -------
        best_bid_price: float
            the maximum price in the bid order book.
        best_ask_price: float
            the minimum price in the ask order book.

        """
        self.best_bid_price = max(self.bids.keys()) if len(self.bids) > 0 else None
        self.best_ask_price = min(self.asks.keys()) if len(self.asks) > 0 else None
        return self.best_bid_price, self.best_ask_price

    def update_snap_total_df(self, time, order_id, typ, snap_level_num):
        snap_row = self.gen_snap_shot(level_num=snap_level_num)
        snap_row['time'] = time
        snap_row['order_id'] = order_id
        snap_row['typ'] = typ
        self.snap_total_list.append(snap_row)

    def gen_snap_shot(self, level_num=10):
        """ Generate the snap shot.

        Parameters
        ----------
        level_num: int >0
            The number of the levels in the output snap shot.

        Returns
        -------
        snap_shot: The snap shot of the order book.
            {'OfferPX1': 16.87,'OfferSize1': 2360, 'BidPX1': 16.86, 'BidSize1': 19640, ... }
        """
        # update the best price
        self.update_best_price()

        snap_shot = {}
        bid_price_list = heapq.nlargest(level_num, list(self.bids_snap.keys()))
        ask_price_list = heapq.nsmallest(level_num, list(self.asks_snap.keys()))

        for i in range(level_num):
            try:
                ask_price = ask_price_list[i]
                snap_shot['OfferPX' + str(i + 1)] = ask_price
                snap_shot['OfferSize' + str(i + 1)] = self.asks_snap[ask_price]
            except IndexError:
                snap_shot['OfferPX' + str(i + 1)] = None
                snap_shot['OfferSize' + str(i + 1)] = None
            try:
                bid_price = bid_price_list[i]
                snap_shot['BidPX' + str(i + 1)] = bid_price
                snap_shot['BidSize' + str(i + 1)] = self.bids_snap[bid_price]
            except IndexError:
                snap_shot['BidPX' + str(i + 1)] = None
                snap_shot['BidSize' + str(i + 1)] = None
        return snap_shot

    def renew(self, new_order_id_list=None, time=None, debug_flag=False, execute_flag=False, snap_flag=False,
              snap_level_num=None, snap_rule="order_before", execute_rule="trade_after", execute_level_num=None):
        """Renew the order book.

        Match all the possible executions in the order book via price-time priority.

        Parameters
        ----------
        new_order_id_list: list
            The IDs of the new orders before renewing. (None if the renewing is a call auction.)
        time: int or datetime.datetime
            The time when the renewing occurs.
        debug_flag: bool
        execute_flag: bool
            Update execute_total_list or not
        execute_rule: {"trade_before", "trade_after"}
            Which order-book snapshot to attach to each execution row:
            the book just before the incoming order ("trade_before",
            via last_snap_shot) or the book right after the trade
            ("trade_after").
        Returns
        -------

        """
        execute_list = []
        best_bid_price, best_ask_price = self.update_best_price()
        bid_order, ask_order = None, None
        while ((best_bid_price is not None)
               and (best_ask_price is not None)
               and (best_bid_price >= best_ask_price)):
            bid_order = self.bids[best_bid_price][0]
            ask_order = self.asks[best_ask_price][0]
            match_qty = min(bid_order.leaves_qty, ask_order.leaves_qty)

            if new_order_id_list is not None:  # continuous auction
                if bid_order.order_id in new_order_id_list:
                    trade_price = ask_order.price
                    direction = Side.BUY
                elif ask_order.order_id in new_order_id_list:
                    trade_price = bid_order.price
                    direction = Side.SELL
            else:
                trade_price = None
                direction = None

            bid_order.cum_qty += match_qty
            bid_order.leaves_qty -= match_qty
            if bid_order.leaves_qty < 1e-9:
                del self.bids[best_bid_price][0]
            if len(self.bids[best_bid_price]) == 0:
                del self.bids[best_bid_price]

            self.bids_snap[best_bid_price] = self.bids_snap[best_bid_price] - match_qty
            if self.bids_snap[best_bid_price] == 0:
                del self.bids_snap[best_bid_price]

            ask_order.cum_qty += match_qty
            ask_order.leaves_qty -= match_qty
            if ask_order.leaves_qty < 1e-9:
                del self.asks[best_ask_price][0]
            if len(self.asks[best_ask_price]) == 0:
                del self.asks[best_ask_price]

            self.asks_snap[best_ask_price] = self.asks_snap[best_ask_price] - match_qty
            if self.asks_snap[best_ask_price] == 0:
                del self.asks_snap[best_ask_price]

            best_bid_price, best_ask_price = self.update_best_price()

            if execute_flag:
                execute_row = {'Time': time,
                               'BidApplSeqNum': bid_order.order_id,
                               'OfferApplSeqNum': ask_order.order_id,
                               'TradeQty': match_qty,
                               'Price': trade_price,
                               "Direction": direction, }
                if execute_rule == "trade_before":
                    execute_row.update(self.last_snap_shot)
                elif execute_rule == "trade_after":
                    execute_row.update(self.gen_snap_shot(level_num=execute_level_num))
                execute_list.append(execute_row)

        if new_order_id_list is None:  # TODO: call auction: determine the trade price
            if (best_bid_price is None) or ((bid_order is not None) and (bid_order.price > best_bid_price)):
                trade_price = bid_order.price
            elif (best_ask_price is None) or ((ask_order is not None) and (ask_order.price < best_ask_price)):
                trade_price = ask_order.price

        if execute_flag and (len(execute_list) != 0):
            if new_order_id_list is None:
                for row in execute_list:
                    row['Price'] = trade_price
            self.execute_total_list.extend(execute_list)
        return None


class FreezeOrderBook(OrderBook):
    """ A holding pen for orders frozen by the ChiNext price-cage (鸽笼) rule.

    Limit orders priced more than 2% away from the opposite best price are
    parked here instead of entering the live book, and are released
    (defreezed) once the market moves within range.

    Attributes:
    bids: dict
        A dict to store the order book on the bid side.
        Example: {price1:[Order1, Order2], price2:[Order3], ...}
    asks: dict
        A dict to store the order book on the ask side.
    order_id_map: dict
        A dict to store all the order in the order book, indexed by their order ids.
        Example: {order_id1: Order1, order_id2: Order2}
    best_bid_price: float
    best_ask_price: float

    """

    def __init__(self):
        super().__init__()

    def update_best_price(self):
        """ Update the best bid and ask price of the freeze order book.

        Note the min/max are the reverse of a normal book: frozen bids are
        priced ABOVE the cage, so the lowest frozen bid is the first to
        become releasable (and vice versa for asks).

        Returns
        -------
        best_bid_price: float
            the minimum price in the bid order book.
        best_ask_price: float
            the maximum price in the ask order book.

        """
        self.best_bid_price = min(self.bids.keys()) if len(self.bids) > 0 else None
        self.best_ask_price = max(self.asks.keys()) if len(self.asks) > 0 else None
        return self.best_bid_price, self.best_ask_price

    def defreeze(self, bid_base_price, ask_base_price):
        """ Defreeze the order from the freeze order book.

        The bid orders with price lower than the bid base price
        and the ask orders with price higher than the ask base price can be defreezed from the freeze order book.

        Parameters
        ----------
        bid_base_price: float

        ask_base_price: float

        Returns
        -------
        defreeze_order_list: list[Order]
            List of defreezed orders.

        """
        defreeze_order_list = []
        if (self.best_bid_price is not None) and (self.best_bid_price <= bid_base_price):
            defreeze_price_list = sorted([price for price in self.bids.keys() if price <= bid_base_price], reverse=True)
            for price in defreeze_price_list:
                defreeze_order_list.extend(self.bids[price])
                del self.bids[price]
        elif (self.best_ask_price is not None) and (self.best_ask_price >= ask_base_price):
            defreeze_price_list = sorted([price for price in self.asks.keys() if price >= ask_base_price])
            for price in defreeze_price_list:
                defreeze_order_list.extend(self.asks[price])
                del self.asks[price]
        if len(defreeze_order_list) > 0:
            for order in defreeze_order_list:
                del self.order_id_map[order.order_id]
        return defreeze_order_list


class Engine:
    """ The matching engine.

    Attributes:
        stock: str
            The stock ID.
            Example: "000001.XSHE"
        year: int
        month: int
        day: int
        rule: {"normal","freeze"}
            matching rule.
        order_book: NormalOrderBook
            The limit order book
        freeze_order_book: FreezeOrderBook
            The freeze order book
        data: Data
            The data class for the matching.
        bid_base_price: float
        ask_base_price: float
        high_stop_price: float
        low_stop_price: float
        pre_close_price: float
    """

    def __init__(self, stock: str, year: int, month: int, day: int, file_path=None):
        self.stock = stock
        self.year = year
        self.month = month
        self.day = day

        if (stock[0] == '3') and ((year > 2020) or (year == 2020 and int(str(month) + str(day).zfill(2)) >= 824)):
            self.rule = 'freeze'
        else:
            self.rule = 'normal'

        self.order_book = NormalOrderBook()
        self.freeze_order_book = FreezeOrderBook() if (self.rule == "freeze") else None
        self.data = Data(stock, year, month, day, file_path=file_path)

        self.bid_base_price = None
        self.ask_base_price = None
        self.high_stop_price = None
        self.low_stop_price = None
        self.pre_close_price = None

    def call_auction(self, match_df: pd.DataFrame, execute_flag=False, execute_level_num=None):
        """ Conduct a call auction after all the order submission and cancellation in the match_df.

        Parameters
        ----------
        match_df: pd.DataFrame
            all the order submission and cancellation.
        execute_flag: bool
            Update execute_total_list or not
        Returns
        -------

        """
        self.order_book.last_snap_shot = self.order_book.gen_snap_shot(level_num=1)
        for row in match_df.itertuples():
            order_id = getattr(row, 'OrderID')
            price = getattr(row, 'Price')
            qty = getattr(row, 'Qty')
            side = getattr(row, 'Side')
            typ = getattr(row, 'Type')
            time = getattr(row, 'time')
            if typ == '2':  # limit order
                order = Order(order_id, price, qty, typ, side, time)
                self.order_book.add_order(order)
            elif typ == '4':  # cancellation
                self.order_book.cancel_order(order_id)
        self.order_book.renew(time=92500000, execute_flag=execute_flag, execute_level_num=execute_level_num)
        return None

    def market_order_price(self, typ: str, side):
        """ Determine the equivalent limit price of a market order.

        "U": best price on the order's own side. (If that side of the book is
             empty, use an unfillable price -- the order is cancelled right after.)
        "MU": best price on the opposite side.
        "5C": fifth-best price on the opposite side. (Fewer than five levels:
              use the deepest available; empty book: an unfillable price,
              since the remainder is cancelled immediately.)
        "M": up-limit/down-limit price, i.e. cross the whole opposite book.
        "MC": an order with no fills at all; priced to never trade because it
              is about to be cancelled.

        Parameters
        ----------
        typ: str
            Type of the order (see Order.typ).
        side: Side
            Side of the order.

        Returns
        -------
        price: float
            Equivalent price of a market order

        """
        order_book = self.order_book
        bids, asks = order_book.bids, order_book.asks
        high_stop_price, low_stop_price = self.high_stop_price, self.low_stop_price
        if typ == 'U':
            if side == Side.BUY:
                price = max(order_book.bids.keys()) if len(order_book.bids) > 0 else (low_stop_price - 0.01)
            elif side == Side.SELL:
                price = min(order_book.asks.keys()) if len(order_book.asks) > 0 else (high_stop_price + 0.01)
        elif typ == 'MU':
            best_bid_price, best_ask_price = order_book.update_best_price()
            if side == Side.BUY:
                price = best_ask_price
            elif side == Side.SELL:
                price = best_bid_price
        elif typ == '5C':
            if side == Side.BUY:
                asks_len = len(asks.keys())
                if asks_len == 0:  # opposite book is empty; any unfillable price works, the order is cancelled next
                    price = high_stop_price
                else:
                    price = sorted(list(asks.keys()))[min(asks_len, 5) - 1]
            elif side == Side.SELL:
                bids_len = len(bids.keys())
                if bids_len == 0:  # opposite book is empty; any unfillable price works, the order is cancelled next
                    price = low_stop_price
                else:
                    price = sorted(list(bids.keys()))[-min(bids_len, 5)]
        elif typ == 'M':
            if side == Side.BUY:
                price = high_stop_price
            elif side == Side.SELL:
                price = low_stop_price
        elif typ == 'MC':  # an order with no fills at all is always cancelled
            if side == Side.BUY:
                price = low_stop_price
            elif side == Side.SELL:
                price = high_stop_price
        return price

    def continuous_auction(self, match_df: pd.DataFrame,
                           debug_flag: bool = False, break_order_id=None,
                           execute_flag=False,
                           snap_flag=False, snap_level_num=None, snap_rule="order_before", execute_rule="trade_after",
                           execute_level_num=None):
        """ Conduct a continuous auction along all the order submission and cancellation in the match_df.

        For each event:
        1. Cancellation: remove the order from whichever book holds it.
        2. Submission: convert everything to a limit order first.
            - normal rule: add to the order book, then renew (match) it.
            - freeze (price-cage) rule: park the order in the freeze book if it
              is priced outside the cage; otherwise submit it, then keep
              releasing frozen orders that came back in range until none qualify.

        Parameters
        ----------
        match_df: pd.DataFrame
            all the order submission and cancellation.
        debug_flag: bool
        break_order_id: int
            The order id to break the continuous auction. (for debugging)
        execute_flag: bool
            Update execute_total_list or not
        Returns
        -------

        """
        order_book = self.order_book
        freeze_order_book = self.freeze_order_book
        best_bid_price, best_ask_price = order_book.update_best_price()
        bid_base_price = _my_round(best_ask_price * 1.02, 2) if best_ask_price is not None else None
        ask_base_price = _my_round(best_bid_price * 0.98, 2) if best_bid_price is not None else None

        for row in match_df.itertuples():
            if debug_flag:
                idx = getattr(row, 'Index')
                if idx % 10000 == 0:
                    print(idx)

            order_id = getattr(row, 'OrderID')
            price = getattr(row, 'Price')
            qty = getattr(row, 'Qty')
            side = getattr(row, 'Side')
            typ = getattr(row, 'Type')
            time = getattr(row, 'time')

            if snap_flag and snap_rule == "order_before":
                self.order_book.update_snap_total_df(time, order_id, typ, snap_level_num=snap_level_num)

            if typ == '4':
                if order_id in order_book.order_id_map.keys():
                    order_book.cancel_order(order_id)
                elif order_id in freeze_order_book.order_id_map.keys():
                    freeze_order_book.cancel_order(order_id)
            else:

                if typ != '2':
                    price = self.market_order_price(typ, side)
                order = Order(order_id, price, qty, typ, side, time)
                if (break_order_id is not None) and (order_id == break_order_id):
                    print(order)
                    break

                if ((self.rule == "freeze") and (typ == '2')
                        and ((side == Side.BUY and price > bid_base_price) or (
                                side == Side.SELL and price < ask_base_price))):
                    freeze_order_book.add_order(order)
                    freeze_order_book.update_best_price()
                else:
                    if execute_rule == "trade_before":
                        order_book.last_snap_shot = order_book.gen_snap_shot(level_num=execute_level_num)
                    order_book.add_order(order)
                    order_book.renew(new_order_id_list=[order_id], time=time, execute_flag=execute_flag,
                                     snap_flag=snap_flag, snap_level_num=snap_level_num,
                                     snap_rule=snap_rule, execute_rule=execute_rule,
                                     execute_level_num=execute_level_num)
            if self.rule == "freeze":
                best_bid_price, best_ask_price = order_book.update_best_price()
                bid_base_price = _my_round(best_ask_price * 1.02, 2) if best_ask_price is not None else None
                ask_base_price = _my_round(best_bid_price * 0.98, 2) if best_bid_price is not None else None

                defreeze_order_list = freeze_order_book.defreeze(bid_base_price, ask_base_price)
                while len(defreeze_order_list) != 0:
                    freeze_order_book.update_best_price()
                    if execute_rule == "trade_before":
                        order_book.last_snap_shot = order_book.gen_snap_shot(level_num=execute_level_num)

                    new_order_id_list = []
                    for order in defreeze_order_list:
                        order_book.add_order(order)
                        new_order_id_list.append(order.order_id)
                    order_book.renew(new_order_id_list=new_order_id_list, time=time, execute_flag=execute_flag,
                                     snap_flag=snap_flag, snap_level_num=snap_level_num,
                                     snap_rule=snap_rule, execute_rule=execute_rule,
                                     execute_level_num=execute_level_num)
                    best_bid_price, best_ask_price = order_book.update_best_price()
                    bid_base_price = _my_round(best_ask_price * 1.02, 2)
                    ask_base_price = _my_round(best_bid_price * 0.98, 2)

                    defreeze_order_list = freeze_order_book.defreeze(bid_base_price, ask_base_price)

            if snap_flag and snap_rule == "order_after":
                self.order_book.update_snap_total_df(time, order_id, typ, snap_level_num=snap_level_num)

    def main_matching_process(self, debug_flag: bool = False, check_flag=False, execute_flag=False, snap_flag=False,
                              snap_level_num=None, snap_rule="order_before", execute_rule="trade_after",
                              execute_level_num=None):
        """ Run one full stock-day: opening call auction, then continuous auction.

        With check_flag=True the reconstructed executions are compared against
        the exchange's own trade feed as a correctness check.

        Returns
        -------

        """
        self.data.load_data()
        self.data.gen_total_match_data()
        self.high_stop_price = self.data.high_stop_price
        self.low_stop_price = self.data.low_stop_price
        self.pre_close_price = self.data.pre_close_price

        call_match_df = self.data.gen_match_data(90000000, 92500000)
        self.call_auction(call_match_df, execute_flag=execute_flag, execute_level_num=execute_level_num)

        cont_begin_time = 93000000
        cont_end_time = 145700000
        cont_match_df = self.data.gen_match_data(cont_begin_time, cont_end_time)
        self.continuous_auction(cont_match_df, debug_flag=debug_flag, execute_flag=execute_flag, snap_flag=snap_flag,
                                snap_level_num=snap_level_num, snap_rule=snap_rule, execute_rule=execute_rule,
                                execute_level_num=execute_level_num)

        if check_flag and execute_flag:
            execute_check_column = ['BidApplSeqNum', 'OfferApplSeqNum', 'Price', 'TradeQty']
            execute_total_df = pd.DataFrame(self.order_book.execute_total_list)
            execute_total_df = execute_total_df[
                (execute_total_df['Time'] >= cont_begin_time) & (execute_total_df['Time'] <= cont_end_time)]
            execute_total_df = execute_total_df[execute_check_column].reset_index(drop=True)
            execute_df = self.data.execute_df
            execute_df = execute_df[
                (self.data.execute_df['time'] >= cont_begin_time) & (self.data.execute_df['time'] <= cont_end_time)]
            execute_df = execute_df[execute_check_column].reset_index(drop=True)

            assert len(execute_df) == len(execute_total_df), \
                f"{self.stock}, {self.year}, {self.month}, {self.day}, Wrong Matching! Contact the author to debug!"
            print(f"{self.stock}, {self.year}, {self.month}, {self.day}, Correct Matching!")
        return None


class Data:
    """ A data class to store all the data for the stock.

    Attributes:
        stock: str
            The stock ID.
            Example: "000001.XSHE"
        year: int
        month: int
        day: int
        file_path: str
            None if the code is running on the server.
        snap_df:
        order_df:
        trade_df:
        high_stop_price: float
        low_stop_price: float
        pre_close_price: float
        match_df: pd.DataFrame ['time', 'OrderID', 'Price', 'Qty', 'Side', 'Type', 'EventID']
            all the order submission and cancellation
        execute_df: pd.DataFrame
            ['tradetime', 'ApplSeqNum', 'BidApplSeqNum', 'OfferApplSeqNum', 'Price', 'TradeQty', 'ExecType', 'time']
            all the execution
        cancel_df: pd.DataFrame ['time', 'OrderID', 'Price', 'Qty', 'Side', 'Type', 'EventID']
            all the order cancellation



    """

    def __init__(self, stock, year, month, day, file_path=None):
        self.stock = stock
        self.year = year
        self.month = month
        self.day = day
        if file_path is None:
            self.file_path = "/data/HFData/processed_data/data_split_total/"
        else:
            self.file_path = file_path
        self.snap_df, self.order_df, self.trade_df = None, None, None
        self.match_df, self.execute_df, self.cancel_df = None, None, None
        self.high_stop_price, self.low_stop_price, self.pre_close_price = None, None, None

    def load_data(self):
        """ Load the data.

        Returns
        -------

        """
        data_path = os.path.join(self.file_path,
                                 self.stock,
                                 str(self.year),
                                 str(self.month).zfill(2) + str(self.day).zfill(2))

        # snap: snapshot price-level table (证券快照行情档位表)
        am_snap_level_spot = pd.read_csv(os.path.join(data_path, 'am_snap_level_spot.csv'), encoding='GBK')
        pm_snap_level_spot = pd.read_csv(os.path.join(data_path, 'pm_snap_level_spot.csv'), encoding='GBK')
        snap_df = pd.concat([am_snap_level_spot, pm_snap_level_spot], ignore_index=True)
        for i in range(1, 11):
            snap_df['OfferPX' + str(i)] = snap_df['OfferPX' + str(i)].round(3)
            snap_df['BidPX' + str(i)] = snap_df['BidPX' + str(i)].round(3)
        self.snap_df = snap_df

        # order: tick-by-tick order table (逐笔委托行情表)
        am_hq_order_spot = pd.read_csv(os.path.join(data_path, 'am_hq_order_spot.csv'),
                                       encoding='GBK', dtype={'OrderType': str})
        pm_hq_order_spot = pd.read_csv(os.path.join(data_path, 'pm_hq_order_spot.csv'),
                                       encoding='GBK', dtype={'OrderType': str})
        order_df = pd.concat([am_hq_order_spot, pm_hq_order_spot], ignore_index=True)
        order_df['ApplSeqNum'] = order_df['ApplSeqNum'].astype(int)
        order_df['Price'] = order_df['Price'].round(3)
        self.order_df = order_df

        # trade: tick-by-tick trade table (逐笔成交表)
        am_hq_trade_spot = pd.read_csv(os.path.join(data_path, 'am_hq_trade_spot.csv'),
                                       encoding='GBK')
        pm_hq_trade_spot = pd.read_csv(os.path.join(data_path, 'pm_hq_trade_spot.csv'),
                                       encoding='GBK')
        trade_df = pd.concat([am_hq_trade_spot, pm_hq_trade_spot], ignore_index=True)
        trade_df['Price'] = trade_df['Price'].round(3)
        trade_df[['BidApplSeqNum', 'OfferApplSeqNum']] = trade_df[['BidApplSeqNum', 'OfferApplSeqNum']].fillna(0)
        trade_df[['BidApplSeqNum', 'OfferApplSeqNum']] = trade_df[['BidApplSeqNum', 'OfferApplSeqNum']].astype(int)
        trade_df.sort_values(by='ApplSeqNum', inplace=True, ascending=True)
        self.trade_df = trade_df
        
        am_hq_snap_spot = pd.read_csv(os.path.join(data_path, 'am_hq_snap_spot.csv'),
                                      encoding='GBK', nrows=1)
        self.high_stop_price = am_hq_snap_spot.loc[0, 'UpLimitPx']
        self.low_stop_price = am_hq_snap_spot.loc[0, 'DownLimitPx']
        self.pre_close_price = am_hq_snap_spot.loc[0, 'PreClosePx']
        return None

    def gen_total_match_data(self):
        """ Generate the match_df, execute_df and cancel_df and specified the market order type.

        Returns
        -------

        """
        order_columns_used = ['TransactTime', 'ApplSeqNum', 'Price', 'OrderQty', 'Side', 'OrderType']
        order_df_used = self.order_df[order_columns_used].copy()

        order_df_used.loc[:, 'Side'] = order_df_used['Side'].replace({1: Side.BUY, 2: Side.SELL})
        order_df_used.loc[:, 'OrderID'] = order_df_used['ApplSeqNum']
        order_df_used = order_df_used[
            ['TransactTime', 'OrderID', 'Price', 'OrderQty', 'Side', 'OrderType', 'ApplSeqNum']]
        order_df_used.columns = ['time', 'OrderID', 'Price', 'Qty', 'Side', 'Type', 'EventID']

        trade_columns_used = ['tradetime', 'ApplSeqNum', 'BidApplSeqNum', 'OfferApplSeqNum', 'Price', 'TradeQty',
                              'ExecType']
        trade_df_used = self.trade_df[trade_columns_used].copy()

        execute_df = trade_df_used[trade_df_used['ExecType'] == 'F'].copy()
        # intraday time via string slicing: float "% 1e9" loses precision
        execute_df['time'] = execute_df['tradetime'].apply(lambda x: int(str(x)[-9:]))
        execute_df = execute_df[execute_df['time'] <= 145700000]
        execute_df = execute_df.reset_index(drop=True)

        cancel_df = self.trade_df[self.trade_df['ExecType'] == '4'].copy()
        # in the 2019 data the counterparty order id of a cancel is empty; fill with 0
        cancel_df['OfferApplSeqNum'] = cancel_df['OfferApplSeqNum'].fillna(0)
        cancel_df['BidApplSeqNum'] = cancel_df['BidApplSeqNum'].fillna(0)
        cancel_df.loc[:, 'Side'] = np.where(cancel_df['OfferApplSeqNum'] == 0, Side.BUY, Side.SELL)
        cancel_df.loc[:, 'OrderID'] = np.where(cancel_df['OfferApplSeqNum'] == 0, cancel_df['BidApplSeqNum'],
                                               cancel_df['OfferApplSeqNum'])
        cancel_df = cancel_df[['tradetime', 'OrderID', 'Price', 'TradeQty', 'Side', 'ExecType', 'ApplSeqNum']]
        cancel_df.columns = ['time', 'OrderID', 'Price', 'Qty', 'Side', 'Type', 'EventID']

        match_df = pd.concat([order_df_used, cancel_df], ignore_index=True)
        match_df = match_df.sort_values('EventID', kind='mergesort')
        match_df['time'] = match_df['time'].apply(lambda x: int(str(x)[-9:]))

        buy_market_id_list = match_df[(match_df['Type'] == '1') & (match_df['Side'] == Side.BUY)]['OrderID'].to_list()
        ask_market_id_list = match_df[(match_df['Type'] == '1') & (match_df['Side'] == Side.SELL)]['OrderID'].to_list()
        market_id_list = buy_market_id_list + ask_market_id_list

        if len(market_id_list) > 0:
            bid_market_order_df = execute_df[execute_df['BidApplSeqNum'].isin(buy_market_id_list)].groupby(
                'BidApplSeqNum').agg(
                price_num=('Price', 'nunique')
            )
            bid_market_order_df = bid_market_order_df.reset_index()
            bid_market_order_df = bid_market_order_df.rename(columns={'BidApplSeqNum': 'OrderID'})

            ask_market_order_df = execute_df[execute_df['OfferApplSeqNum'].isin(ask_market_id_list)].groupby(
                'OfferApplSeqNum').agg(
                price_num=('Price', 'nunique')
            )
            ask_market_order_df = ask_market_order_df.reset_index()
            ask_market_order_df = ask_market_order_df.rename(columns={'OfferApplSeqNum': 'OrderID'})

            market_order_df = pd.concat([bid_market_order_df, ask_market_order_df], ignore_index=True)

            market_order_df['cancel_bool'] = market_order_df['OrderID'].isin(cancel_df['OrderID'])

            def _market_order_type(row):
                if row['price_num'] == 1:
                    return 'MU'
                elif (row['price_num'] == 5) and row['cancel_bool']:
                    return '5C'
                else:
                    return 'M'

            if len(market_order_df) > 0:
                market_order_df['typ'] = market_order_df.apply(lambda row: _market_order_type(row), axis=1)
            else:
                market_order_df['typ'] = []
            market_order_df = market_order_df.merge(pd.DataFrame(index=market_id_list), left_on='OrderID',
                                                    right_index=True, how='outer')
            market_order_df['typ'] = market_order_df['typ'].fillna('MC')  # no fills at all: will be cancelled outright
            market_order_type_dict = market_order_df.groupby('typ')['OrderID'].unique().to_dict()
            for typ, market_id_list in market_order_type_dict.items():
                match_df.loc[(match_df['OrderID'].isin(market_id_list)) & (match_df['Type'] != "4"), 'Type'] = typ

        self.match_df = match_df
        self.execute_df = execute_df
        self.cancel_df = cancel_df
        return None

    def gen_match_data(self, begin_time, end_time):
        """ Slice the match_df between the begin_time and end_time.

        Parameters
        ----------
        begin_time: int
        end_time: int

        Returns
        -------
        Required match_df slice.
        """
        match_df = self.match_df
        return match_df[(match_df['time'] >= begin_time) & (match_df['time'] < end_time)].copy().reset_index(drop=True)


if __name__ == "__main__":
    # Runnable example: replay one stock-day, verify the reconstruction
    # against the exchange trade feed, and aggregate fills per taker order.
    import time

    file_path = "../data_sample/processed_data"  # adjust to your data layout
    stock = "000001.XSHE"
    year, month, day = 2020, 1, 2

    engine = Engine(stock, year, month, day, file_path=file_path)
    s_time = time.time()
    engine.main_matching_process(check_flag=True, execute_flag=True,
                                 snap_flag=True, snap_rule="order_before", snap_level_num=1,
                                 execute_rule="trade_before", execute_level_num=1)
    print(f"matching took {time.time() - s_time:.2f}s")

    snap_total_df = pd.DataFrame(engine.order_book.snap_total_list)
    execute_total_df = pd.DataFrame(engine.order_book.execute_total_list)
    execute_total_df = execute_total_df[execute_total_df['Time'] >= 93000000]
    execute_total_df['InitApplSeqNum'] = np.where(execute_total_df['Direction'] == Side.BUY,
                                                  execute_total_df['BidApplSeqNum'],
                                                  execute_total_df['OfferApplSeqNum'])
    execute_total_df_agg = execute_total_df.groupby('InitApplSeqNum').agg(
        {'Time': 'last', 'TradeQty': 'sum', 'Direction': 'last',
         "BidPX1": 'last', "OfferPX1": 'last', "BidSize1": 'last', "OfferSize1": 'last'}
    ).reset_index()
    print(execute_total_df_agg.head())
