import time
from typing import Optional
from enum import Enum
from datetime import datetime

from koapys.koa_py_simple import KoaPySimple, KoaPySimpleEvent, OrderType, OrderBookType
from koapys.logger.logger import koapysLogger
from lib.event_emitter import EventEmitter
from lib.thread_safe_dict import ThreadSafeDict
from lib.type import VoidListener
from lib.utils import is_empty_str


class OrderState(Enum):
    NONE = 0
    ORDER_RECEIPTED = 1
    CONTRACTED = 2
    CANCELED = 3
    REFUSED = 100


class Order:
    _buy_order_map = ThreadSafeDict()
    _sell_order_map = ThreadSafeDict()
    _liquidated_orders = []
    _canceled_orders = []

    def __init__(self, koapys: KoaPySimple,
                 account_no: str,
                 code: str,
                 name: str,
                 price: int,
                 quantity: int,
                 type: OrderType,
                 org_order: Optional['Order'] = None):
        self.koapys = koapys
        self.state = OrderState.NONE

        self.request_name = f"{time.time()}-{name}-{type.name}"
        if org_order:
            self.request_name += f"-{org_order.no}"

        self.account_no = account_no
        self.code = code
        self.name = name
        self.price = price
        self.contracted_price = 0
        self.quantity = quantity
        self.type = type
        self.org_order = org_order

        self.no = ""
        self.contracted_time = None
        self.filled = 0
        self.left = 0

        self._emitter = EventEmitter()

    @property
    def is_buy_order(self):
        return self.type == OrderType.BUY

    @property
    def is_sell_order(self):
        return self.type == OrderType.SELL

    @property
    def is_receipted(self):
        return self.state == OrderState.ORDER_RECEIPTED

    @property
    def is_receipted_and_filled(self):
        """
        IOC 나 FOK 로 하면 접수는 되지만 매매가 안되면 바로 취소 됨
        :return:
        """
        return self.is_receipted and self.filled > 0

    @property
    def is_refused(self):
        return self.state == OrderState.REFUSED

    @property
    def is_contracted(self):
        return self.state == OrderState.CONTRACTED

    @property
    def is_contracted_and_filled_all(self):
        return self.is_contracted and self.filled == self.quantity

    def add_receipted_event_listener(self, listener: VoidListener):
        self._emitter.on(KoaPySimpleEvent.ORDER_RECEIPTED, listener)

    def add_contracted_event_listener(self, listener: VoidListener):
        self._emitter.on(KoaPySimpleEvent.ORDER_CONTRACTED, listener)

    def add_refused_event_listener(self, listener: VoidListener):
        self._emitter.on(KoaPySimpleEvent.ORDER_REFUSED, listener)

    def _handle_receipted_event(self, data: dict):
        order_no = data["주문번호"]
        if not is_empty_str(self.no) and self.no != order_no:
            return
        if Order.find(self.code, order_no):
            return

        #**del
        # # check identity
        # order_type_map = {
        #     '매수': OrderType.BUY,
        #     '매도': OrderType.SELL,
        #     '매수취소': OrderType.BUY_CANCEL,
        #     '매도취소': OrderType.SELL_CANCEL,
        #     '매수정정': OrderType.BUY_CORRECT,
        #     '매도정정': OrderType.SELL_CORRECT
        # }
        # for suffix, order_type in order_type_map.items():
        #     if data['주문구분'].endswith(suffix) and self.type != order_type:
        #         return

        self._receipted(order_no)

    def _handle_contracted_event(self, data: dict):
        order_no = data["주문번호"]
        if self.no != order_no:
            return

        contracted_price = data["체결가"]
        filled = data["체결량"]
        left = data["미체결수량"]
        self._contracted(contracted_price, filled, left)

    def _handle_filled_event(self, data: dict):
        # 잔고
        code = data["종목코드,업종코드"][1:]
        if self.code != code:
            return

        name = data["종목명"]
        amount = data["보유수량"]
        print(f'종목명: {name}, 체결량: {self.filled}, 보유수량: {amount}')

    def _handle_refused_event(self):
        self._refused()

    def _receipted(self, no: str):
        #**del 같은 종목에서 매수, 매도, 취소, 정정 등 여러번 주문하면 여러번 올 수 있음
        # assert self.state == OrderState.NONE
        if self.state != OrderState.NONE:
            return
        koapysLogger.debug(f"Order._receipted: {[self.code, self.name, self.state]}")
        self.state = OrderState.ORDER_RECEIPTED
        self.no = no
        self._emitter.emit(KoaPySimpleEvent.ORDER_RECEIPTED)

    def _contracted(self, contracted_price, filled, left):
        #**del 같은 종목에서 매수, 매도, 취소, 정정 등 여러번 주문하면 여러번 올 수 있음
        # assert self.state == OrderState.ORDER_RECEIPTED
        # if self.state != OrderState.ORDER_RECEIPTED:
        #     koapysLogger.debug(f"Order._contracted: {[self.code, self.name, self.state]}")
        #     return
        koapysLogger.debug(f"Order._contracted: {[self.code, self.name, self.state]}")
        # TODO: 부분 체결되서 체결가가 서로 다른 경우 처리.
        self.state = OrderState.CONTRACTED
        self.contracted_time = datetime.now()
        self.contracted_price = int(contracted_price)
        #**del: '단위체결량' 은 filled 를 누적 해야 하고 '체결량'은 그냥 넣으면 된다.
        # self.filled += int(filled)
        self.filled = int(filled)
        self.left = int(left)
        self._emitter.emit(KoaPySimpleEvent.ORDER_CONTRACTED)

    def _refused(self):
        koapysLogger.debug(f"Order._refused: {[self.code, self.name, self.state]}")
        self.state = OrderState.REFUSED
        self._emitter.emit(KoaPySimpleEvent.ORDER_REFUSED)

    @staticmethod
    def _liquidated(buy_order: Optional['Order'], sell_order: Optional['Order']):
        Order._liquidated_orders.append({
            'buy_order': buy_order,
            'sell_order': sell_order,
        })

    @staticmethod
    def _canceled(org_order: Optional['Order'], new_order: Optional['Order']):
        Order._canceled_orders.append({
            'org_order': org_order,
            'new_order': new_order,
        })

    @staticmethod
    def get_equity_balance():
        lst = []
        for code, buy_orders in Order._buy_order_map.items():
            # 매수 주문 중 체결된 수량 합계 계산
            total_buy_quantity = sum(
                order.filled
                for order in buy_orders
                if order.is_contracted
            )

            # 매도 주문이 있는 경우 체결된 매도 수량 차감
            sell_orders = Order._sell_order_map.get(code, [])
            total_sell_quantity = sum(
                order.filled
                for order in sell_orders
                if order.is_contracted
            )

            # 실제 보유 수량 계산
            remaining_quantity = total_buy_quantity - total_sell_quantity

            # 보유 수량이 있는 경우만 리스트에 추가
            if remaining_quantity > 0:
                # 가장 최근 매수 주문 정보 사용
                last_buy_order = next(
                    (order for order in reversed(buy_orders) if order.is_contracted),
                    None
                )
                if last_buy_order:
                    # 새로운 Order 객체 생성하여 정보 복사
                    balance_order = Order(
                        koapys=last_buy_order.koapys,
                        account_no=last_buy_order.account_no,
                        code=last_buy_order.code,
                        name=last_buy_order.name,
                        price=last_buy_order.price,
                        quantity=remaining_quantity,  # 실제 보유 수량 사용
                        type=last_buy_order.type
                    )
                    balance_order.state = last_buy_order.state
                    balance_order.no = last_buy_order.no
                    balance_order.contracted_time = last_buy_order.contracted_time
                    balance_order.contracted_price = last_buy_order.contracted_price
                    balance_order.filled = remaining_quantity  # 실제 보유 수량 사용
                    balance_order.left = 0

                    lst.append(balance_order)

        return lst

    @staticmethod
    def find(code, no: str) -> Optional['Order']:
        lst = Order._buy_order_map.get(code, [])
        if len(lst) > 0:
            for order in lst:
                if order.no == no:
                    return order
        return None

    @staticmethod
    def get_last_buy_order(code: str) -> Optional['Order']:
        lst = Order._buy_order_map.get(code, [])
        if len(lst) > 0:
            return lst[-1]
        return None

    @staticmethod
    def get_last_sell_order(code: str) -> Optional['Order']:
        lst = Order._sell_order_map.get(code, [])
        if len(lst) > 0:
            return lst[-1]
        return None


class Broker:
    @staticmethod
    def buy(koapys: KoaPySimple, account_no: str,
            code: str, name: str, price: int, quantity: int,
            orderbook_type: OrderBookType = OrderBookType.MARKET_IOC,
            simulation: bool = False) -> Order:

        order = Order(koapys, account_no, code, name, price, quantity, OrderType.BUY)
        order.koapys.add_order_receipted_event_listener(order.code, order._handle_receipted_event)
        order.koapys.add_order_contracted_event_listener(order.code, order._handle_contracted_event)
        order.koapys.add_order_filled_event_listener(order.code, order._handle_filled_event)
        order.koapys.add_order_refused_event_listener(order.request_name, order._handle_refused_event)

        if simulation:
            order._receipted("0")
            order._contracted(price, quantity, 0)
        else:
            order.koapys.send_order(order.request_name, order.account_no,
                                    order.code, order.quantity, order.price,
                                    order.type, orderbook_type)

        return order

    @staticmethod
    def sell(koapys: KoaPySimple, account_no: str,
             code: str, name: str, price: int, quantity: int,
             orderbook_type: OrderBookType = OrderBookType.MARKET_IOC,
             simulation: bool = False) -> Order:

        order = Order(koapys, account_no, code, name, price, quantity, OrderType.SELL)
        order.koapys.add_order_receipted_event_listener(order.code, order._handle_receipted_event)
        order.koapys.add_order_contracted_event_listener(order.code, order._handle_contracted_event)
        order.koapys.add_order_filled_event_listener(order.code, order._handle_filled_event)
        order.koapys.add_order_refused_event_listener(order.request_name, order._handle_refused_event)

        if simulation:
            order._receipted("0")
            order._contracted(price, quantity, 0)
        else:
            order.koapys.send_order(order.request_name, order.account_no,
                                    order.code, order.quantity, order.price,
                                    order.type, orderbook_type)

        return order

    @staticmethod
    def correct(org_order: Order, new_price: int, new_quantity: int,
                orderbook_type: OrderBookType = OrderBookType.MARKET_IOC) -> Order:
        assert org_order.type == OrderType.BUY or org_order.type == OrderType.SELL

        order = Order(org_order.koapys,
                      org_order.account_no, org_order.code, org_order.name, 0, org_order.filled,
                      OrderType.SELL_CORRECT if org_order.type == OrderType.SELL else OrderType.BUY_CORRECT,
                      org_order=org_order)

        order.koapys.add_order_receipted_event_listener(order.code, order._handle_receipted_event)
        order.koapys.add_order_contracted_event_listener(order.code, order._handle_contracted_event)
        order.koapys.add_order_filled_event_listener(order.code, order._handle_filled_event)
        order.koapys.add_order_refused_event_listener(order.request_name, order._handle_refused_event)

        org_order.koapys.send_order(order.request_name, order.account_no,
                                    order.code, new_quantity, new_price,
                                    order.type, orderbook_type,
                                    original_order_no=org_order.no)
        return order

    @staticmethod
    def cancel(org_order: Order) -> Order:
        assert org_order.type == OrderType.BUY or org_order.type == OrderType.SELL

        order = Order(org_order.koapys,
                      org_order.account_no, org_order.code, org_order.name, 0, org_order.filled,
                      OrderType.SELL_CANCEL if org_order.type == OrderType.SELL else OrderType.BUY_CANCEL,
                      org_order=org_order)

        order.koapys.add_order_receipted_event_listener(order.code, order._handle_receipted_event)
        order.koapys.add_order_contracted_event_listener(order.code, order._handle_contracted_event)
        order.koapys.add_order_filled_event_listener(order.code, order._handle_filled_event)
        order.koapys.add_order_refused_event_listener(order.request_name, order._handle_refused_event)

        org_order.koapys.send_order(order.request_name, order.account_no,
                                    order.code, order.quantity, order.price,
                                    order.type, OrderBookType.LIMIT,
                                    original_order_no=org_order.no)
        return order

    @staticmethod
    def sell_all():
        for orders in Order._buy_order_map.values():
            for order in orders:
                if order.is_contracted:
                    Broker.sell(order.koapys, order.account_no,
                                order.code, order.name, 0, order.filled)

    @staticmethod
    def cancel_all():
        for orders in Order._buy_order_map.values():
            for order in orders:
                if order.is_contracted:
                    Broker.cancel(order)
