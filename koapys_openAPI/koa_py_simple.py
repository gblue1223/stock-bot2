import logging
import threading
from datetime import datetime

from enum import Enum
from typing import Any, Tuple, Union, Callable
from PyQt5.QtCore import *

from koapys.fid_list import FidList
from koapys.koa_proxy import KoaProxy
from koapys.logger.logger import koapysLogger
from koapys.screen_no import ScreenNo
from koapys.simulator import Simulator
from lib.event_emitter import EventEmitter
from lib.thread_safe_dict import ThreadSafeDict
from lib.type import DataListener, StringListener
from lib.utils import drop_keys


OrderingListener = Callable[[str, any, any], None]


def _to_readable_number(data, display_type=0) -> str:
    if display_type == 1:
        f = int(data) / 100
        value = '{:-,.2f}'.format(f)

    elif display_type == 2:
        f = float(data)
        value = '{:-,.2f}'.format(f)

    else:
        d = int(data)
        value = '{:-,d}'.format(d)

    return value


class OrderType(Enum):
    """
    주문유형(1:신규매수, 2:신규매도 3:매수취소, 4:매도취소, 5:매수정정,
           6:매도정정, 7:프로그램매매 매수, 8:프로그램매매 매도)
    """
    NONE = 0
    BUY = 1
    SELL = 2
    BUY_CANCEL = 3
    SELL_CANCEL = 4
    BUY_CORRECT = 5
    SELL_CORRECT = 6


class KoaPySimpleEvent:
    ORDER_ORDERING = ":ORDER_ORDERING"
    ORDER_RECEIPTED = ":ORDER_RECEIPTED"
    ORDER_CONTRACTED = ":ORDER_CONTRACTED"
    ORDER_FILLED = ":ORDER_FILLED"
    ORDER_REFUSED = ":ORDER_REFUSED"
    CONDITION_LIST = ":CONDITION_LIST"
    CONDITION_ADDED = ":CONDITION_ADDED"
    CONDITION_REMOVED = ":CONDITION_REMOVED"
    REAL_DATA = ":REAL_DATA"


class OrderBookType(Enum):
    """
    00: 지정가, 03: 시장가,
    05: 조건부지정가, 06: 최유리지정가, 07: 최우선지정가,
    10: 지정가IOC, 13: 시장가IOC, 16: 최유리IOC,
    20: 지정가FOK, 23: 시장가FOK, 26: 최유리FOK,
    61: 장전시간외종가, 62: 시간외단일가, 81: 장후시간외종가
    """
    LIMIT = "00"
    MARKET = "03"
    CONDITIONAL_LIMIT = "05"
    BEST_LIMIT = "06"
    PRIORITY_LIMIT = "07"
    LIMIT_IOC = "10"
    MARKET_IOC = "13"
    BEST_IOC = "16"
    LIMIT_FOK = "20"
    MARKET_FOK = "23"
    BEST_FOK = "26"
    PRE_MARKET_CLOSE = "61"
    AFTER_HOURS_SINGLE = "62"
    POST_MARKET_CLOSE = "81"


class KoaPySimple:
    def __init__(self, simulation: bool = False, logger: logging.Logger = None):
        if logger is None:
            logger = koapysLogger
        self._logger = logger

        self._is_connected = False
        self._login_event_loop = QEventLoop()

        self._condition_event_loop = QEventLoop()
        self._condition_search_result_map = {}

        self._emitter = EventEmitter()
        self._registered_realtime_data = ThreadSafeDict()

        self._lock = threading.RLock()
        self._lock_4_tr = threading.RLock()
        self._lock_4_tr_condition = threading.RLock()
        self._lock_4_real = threading.RLock()
        self._lock_4_order = threading.RLock()

        self._simulation = simulation
        if not simulation:
            KoaProxy.Init()
            self._set_signal_slots()

    #
    # Event handlers
    #

    _handler_emitter = None

    def _set_signal_slots(self):
        if KoaPySimple._handler_emitter is None:
            KoaPySimple._handler_emitter = emitter = EventEmitter()
            KoaProxy.ocx.OnEventConnect.connect(lambda *args: emitter.emit("_event_connect", *args))
            KoaProxy.ocx.OnReceiveTrData.connect(lambda *args: emitter.emit("_receive_tr_data", *args))
            KoaProxy.ocx.OnReceiveRealData.connect(lambda *args: emitter.emit("_receive_real_data", *args))
            KoaProxy.ocx.OnReceiveChejanData.connect(lambda *args: emitter.emit("_receive_chejan_data", *args))
            KoaProxy.ocx.OnReceiveMsg.connect(lambda *args: emitter.emit("_receive_msg", *args))
            KoaProxy.ocx.OnReceiveConditionVer.connect(lambda *args: emitter.emit("_receive_condition_ver", *args))
            KoaProxy.ocx.OnReceiveTrCondition.connect(lambda *args: emitter.emit("_receive_tr_condition", *args))
            KoaProxy.ocx.OnReceiveRealCondition.connect(lambda *args: emitter.emit("_receive_real_condition", *args))

        KoaPySimple._handler_emitter.on("_event_connect", self._event_connect)
        KoaPySimple._handler_emitter.on("_receive_tr_data", self._receive_tr_data)
        KoaPySimple._handler_emitter.on("_receive_real_data", self._receive_real_data)
        KoaPySimple._handler_emitter.on("_receive_chejan_data", self._receive_chejan_data)
        KoaPySimple._handler_emitter.on("_receive_msg", self._receive_msg)
        KoaPySimple._handler_emitter.on("_receive_condition_ver", self._receive_condition_ver)
        KoaPySimple._handler_emitter.on("_receive_tr_condition", self._receive_tr_condition)
        KoaPySimple._handler_emitter.on("_receive_real_condition", self._receive_real_condition)

    def _receive_tr_data(self, screen_no, rqname, trcode, record_name, next, unused1, unused2, unused3, unused4):
        # trcode
        # - KOA_NORMAL_BUY_KP_ORD  : 코스피 매수
        # - KOA_NORMAL_SELL_KP_ORD : 코스피 매도
        # - KOA_NORMAL_KP_CANCEL   : 코스피 주문 취소
        # - KOA_NORMAL_KP_MODIFY   : 코스피 주문 변경
        # - KOA_NORMAL_BUY_KQ_ORD  : 코스닥 매수
        # - KOA_NORMAL_SELL_KQ_ORD : 코스닥 매도
        # - KOA_NORMAL_KQ_CANCEL   : 코스닥 주문 취소
        # - KOA_NORMAL_KQ_MODIFY   : 코스닥 주문 변경
        #
        if trcode in ('KOA_NORMAL_BUY_KP_ORD', 'KOA_NORMAL_SELL_KP_ORD',
                      'KOA_NORMAL_KP_CANCEL', 'KOA_NORMAL_KP_MODIFY',
                      'KOA_NORMAL_BUY_KQ_ORD', 'KOA_NORMAL_SELL_KQ_ORD',
                      'KOA_NORMAL_KQ_CANCEL', 'KOA_NORMAL_KQ_MODIFY'):
            self._logger.debug(f"주문이 성공적으로 접수되었습니다: {trcode}")
            return None

        rqname_without_code = rqname.split(":")[0]
        single, multi = FidList.get_all_tr_fields(rqname_without_code)
        if len(single) == 0 and len(multi) == 0:
            self._logger.debug(f"'{rqname}'는 지원하지 않는 데이터입니다: trcode={trcode}, record_name={record_name}")
            return

        single_data = {}
        for key in single:
            cell = KoaProxy.GetCommData(trcode, rqname, 0, key)
            single_data[key] = cell

        multi_data_list = []
        row_len = KoaProxy.GetRepeatCnt(trcode, rqname)
        if row_len == 0:
            row_len = 1

        for row_idx in range(row_len):
            row = {}
            for key in multi:
                cell = KoaProxy.GetCommData(trcode, rqname, row_idx, key)
                row[key] = cell
            multi_data_list.append(row)

        QTimer.singleShot(0, lambda: self._emitter.emit(
            rqname, single_data, multi_data_list))

        data = {"single_data": single_data, "multi_data_list": multi_data_list}
        Simulator.log_tr_data(rqname, str(data))

    def _receive_real_data(self, code, real_type, real_data):
        if code not in self._registered_realtime_data:
            return

        if real_type == "ECN주식호가잔량":  # 시간외 잔량
            real_type = "주식호가잔량"
        if real_type == "ECN주식체결":  # 시간외 주식체결
            real_type = "주식체결"

        fid_list = FidList.get_all_values(real_type)
        columns = FidList.get_all_fields(real_type)
        if len(fid_list) == 0:
            if real_type not in ["주식예상체결", "주식우선호가"]:
                koapysLogger.debug(f"'{real_type}'는 지원하지 않는 데이터입니다: code={code} real_data={real_data}")
            return
        row = [code]
        columns.insert(0, "주식코드")

        for fid in fid_list:
            val = KoaProxy.GetCommRealData(code, fid)
            row.append(val)
        data = dict(zip(columns, row))

        QTimer.singleShot(0, lambda: self._emitter.emit(
            code + KoaPySimpleEvent.REAL_DATA, code, real_type, data))

        Simulator.log_real_data(real_type, str(data))

    def _receive_chejan_data(self, gubun, item_cnt, fid_list):
        """주문접수, 체결, 잔고 변경시 이벤트가 발생

        Args:
            gubun (str): '0': 접수, 체결, '1': 잔고 변경
            item_cnt (int): 아이템 갯수
            fid_list (str): fid list
        """
        self._logger.debug(f"주문체결 데이터 수신: gubun={gubun}")

        if gubun == "0":  # 주문 접수, 체결
            rqname = "주문체결"
            fid_list = FidList.get_all_values(rqname)
            columns = FidList.get_all_fields(rqname)
            row = []
            for fid in fid_list:
                cell = KoaProxy.GetChejanData(fid)
                row.append(cell)
            data = dict(zip(columns, row))

            code = data["종목코드,업종코드"][1:]
            status = data["주문상태"]
            order_no = data["주문번호"]

            if status == "접수":
                QTimer.singleShot(0, lambda: self._emitter.emit(
                    code + KoaPySimpleEvent.ORDER_RECEIPTED, data))
            elif status == "체결":
                QTimer.singleShot(0, lambda: self._emitter.emit(
                    code + KoaPySimpleEvent.ORDER_CONTRACTED, data))
            elif status == "확인":
                org_order_no = data["원주문번호"]
                self._logger.debug(f'주문번호: {order_no}, 원주문번호: {org_order_no}')

            Simulator.log_order_data(status, str(data))

        elif gubun in ["1", "4"]:
            rqname = "잔고"
            fid_list = FidList.get_all_values(rqname)
            columns = FidList.get_all_fields(rqname)
            row = []
            for fid in fid_list:
                cell = KoaProxy.GetChejanData(fid)
                row.append(cell)
            data = dict(zip(columns, row))

            code = data["종목코드,업종코드"][1:]
            QTimer.singleShot(0, lambda: self._emitter.emit(
                code + KoaPySimpleEvent.ORDER_FILLED, data))

            Simulator.log_order_data("잔고", str(data))

    def _receive_msg(self, screen, rqname, trcode, msg):
        self._logger.debug(f"서버 메시지 screen: {ScreenNo.to_string(screen)}, rqname: {rqname}, trcode: {trcode}, msg: {msg}")
        if trcode == "KOA_NORMAL_BUY_KP_ORD":
            if ("주문이 거부되었습니다" in msg or
                    "장개시전입니다" in msg or
                    "장종료되었습니다" in msg or
                    "IOC/FOK 불가종목입니다" in msg or
                    "정정단가를 입력하십시요" in msg):
                QTimer.singleShot(0, lambda: self._emitter.emit(
                    rqname + KoaPySimpleEvent.ORDER_REFUSED))
            pass

    def _receive_condition_ver(self, ret, msg):
        if ret == 1:
            # 성공
            self._logger.debug(msg)
        else:
            # 실패
            self._logger.debug(msg)
        self._condition_event_loop.exit()

    def _receive_tr_condition(self, screen_no, code_list, condition_name, condition_index, next):
        if next == '2':
            self._logger.debug("중목이 더 있습니다.")
            KoaProxy.SendCondition(screen_no, condition_name, int(condition_index), 2)  # 2: 연속조회
        else:
            self._logger.debug("받은 종목 리스트의 마지막 입니다.")

        result = self._condition_search_result_map.get(condition_name, [])
        code_list = code_list.split(';')[:-1]
        for code in code_list:
            if code not in result:
                result.append(code)

        QTimer.singleShot(0, lambda: self._emitter.emit(
            KoaPySimpleEvent.CONDITION_LIST, result))

        Simulator.log_condition_data(condition_name, str(code_list))

    def _receive_real_condition(self, code, event_type, condition_name, condition_index):
        result = self._condition_search_result_map.get(condition_name, [])
        if event_type == "I":
            self._logger.debug(f"종목진입: {code}")
            if code not in result:
                result.append(code)
                QTimer.singleShot(0, lambda: self._emitter.emit(
                    KoaPySimpleEvent.CONDITION_ADDED, code))

                Simulator.log_condition_added_data(condition_name, str(code))
        elif event_type == "D":
            self._logger.debug(f"종목이탈: {code}")
            if code in result:
                result.remove(code)
                QTimer.singleShot(0, lambda: self._emitter.emit(
                    KoaPySimpleEvent.CONDITION_REMOVED, code))

                Simulator.log_condition_removed_data(condition_name, str(code))

    def _event_connect(self, err_code):
        if err_code == 0:
            ret = KoaProxy.GetLoginInfo("GetServerGubun")
            if ret == 1 or ret == "1":
                type_ = f"모의투자 {ret}"
            else:
                type_ = f"실투자 {ret}"

            self._logger.debug(f"로그인 성공: {type_}")
            self._is_connected = True
        else:
            self._logger.debug(f"로그인 실패: {err_code}")
            self._is_connected = False
        self._login_event_loop.exit()

    #
    # Event Listener
    #

    def add_ordering_event_listener(self, listener: OrderingListener):
        """
        주문을 증권사에 보낼 때 발생하는 이벤트
        :param listener: code 와 OrderType, OrderBookType 을 argument 로 넘겨준다.
        """
        self._emitter.on(KoaPySimpleEvent.ORDER_ORDERING, listener)

    def add_order_receipted_event_listener(self, code: str, listener: DataListener):
        """
        주문이 접수되었을 때 발생하는 이벤트
        :param code:
        :param listener:
        :return:
        """
        self._emitter.on(code + KoaPySimpleEvent.ORDER_RECEIPTED, listener)

    def add_order_contracted_event_listener(self, code: str, listener: DataListener):
        self._emitter.on(code + KoaPySimpleEvent.ORDER_CONTRACTED, listener)

    def add_order_filled_event_listener(self, code: str, listener: DataListener):
        self._emitter.on(code + KoaPySimpleEvent.ORDER_FILLED, listener)

    def add_order_refused_event_listener(self, request_name: str, listener: DataListener):
        self._emitter.on(request_name + KoaPySimpleEvent.ORDER_REFUSED, listener)

    def add_condition_added_event_listener(self, listener: StringListener):
        """
        ex)
        koapys.add_condition_added_event_listener(lambda code: self.koapys.start_real_data(code))

        :param listener: 이벤트 리스너
        """
        self._emitter.on(KoaPySimpleEvent.CONDITION_ADDED, listener)

    def add_condition_removed_event_listener(self, listener: StringListener):
        """
        ex)
        koapys.add_condition_added_event_listener(lambda code: self.koapys.stop_real_data(code))

        :param listener: 이벤트 리스너
        """
        self._emitter.on(KoaPySimpleEvent.CONDITION_REMOVED, listener)

    def add_real_data_event_listener(self, code, listener: DataListener):
        """
        ex)
        koapys.add_real_data_event_listener(code, lambda type, data: print(type, data))

        :param code: 주식코드
        :param listener: 이벤트 리스너
        """
        self._emitter.on(code + KoaPySimpleEvent.REAL_DATA, listener)

    def remove_real_data_event_listener(self, code, listener: DataListener):
        """
        ex)
        koapys.remove_real_data_event_listener(code, listener)

        :param code: 주식코드
        :param listener: 이벤트 리스너
        """
        self._emitter.off(code + KoaPySimpleEvent.REAL_DATA, listener)

    #
    # Basic Methods
    #

    @property
    def is_connected(self):
        return self._is_connected

    def ensure_connected(self):
        """
        현재접속 상태를 확인하여 연결이 안되어 있으면 연결하는 메소드
        """
        if self.is_simulation_mode:
            return

        ret = KoaProxy.GetConnectState()
        if ret == 0:
            KoaProxy.CommConnect()
            self._login_event_loop.exec_()

            if not self.is_connected:
                raise ConnectionError

    def close(self):
        if self.is_simulation_mode:
            return

        KoaPySimple._handler_emitter.off("_event_connect", self._event_connect)
        KoaPySimple._handler_emitter.off("_receive_tr_data", self._receive_tr_data)
        KoaPySimple._handler_emitter.off("_receive_real_data", self._receive_real_data)
        KoaPySimple._handler_emitter.off("_receive_chejan_data", self._receive_chejan_data)
        KoaPySimple._handler_emitter.off("_receive_msg", self._receive_msg)
        KoaPySimple._handler_emitter.off("_receive_condition_ver", self._receive_condition_ver)
        KoaPySimple._handler_emitter.off("_receive_tr_condition", self._receive_tr_condition)
        KoaPySimple._handler_emitter.off("_receive_real_condition", self._receive_real_condition)

    def get_premarket_start_time(self) -> datetime:
        now = datetime.now()
        if now.month == 11 and now.day == 14:
            now = now.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            now = now.replace(hour=8, minute=30, second=0, microsecond=0)
        return now

    def get_market_start_time(self) -> datetime:
        now = datetime.now()
        if now.month == 11 and now.day == 14:
            now = now.replace(hour=10, minute=0, second=0, microsecond=0)
        else:
            now = now.replace(hour=9, minute=0, second=0, microsecond=0)
        return now

    def get_market_end_time(self) -> datetime:
        now = datetime.now()
        if now.month == 11 and now.day == 14:
            now = now.replace(hour=16, minute=20, second=0, microsecond=0)
        else:
            now = now.replace(hour=15, minute=20, second=0, microsecond=0)
        return now

    def get_account_list(self):
        if self.is_simulation_mode:
            return ['0000000000']

        accounts = KoaProxy.GetLoginInfo("ACCNO")

        self._logger.debug(f"보유 계좌 수: {len(accounts)}")
        self._logger.debug("계좌 목록:")
        for account in accounts:
            self._logger.debug(account)

        return accounts

    def _block_tr_until_event_fired(self, event_id) -> Union[Any, Tuple[Any, ...]]:
        """
        event_loop 를 이용해 동기 방식으로 값을 가져오는 메소드
        :param event_id:
        :return:
            # 예: fn(data) 처럼 data 인자가 하나만 전달될 때
            result = self._block_tr_until_event_fired("event_id")  # result에는 data 값이 들어감

            # 예: fn(data, error_code, message) 처럼 여러 인자가 전달될 때
            result = self._block_tr_until_event_fired("event_id")  # result에는 (data, error_code, message) 튜플이 들어감
            data, error_code, message = result  # 필요시 언패킹하여 사용 가능
        """
        result = {'args': (None, None)}  # 모든 인자를 저장할 수 있도록 변경

        with self._lock_4_tr:
            event_loop = QEventLoop()
            def fn(*args):  # 가변 인자로 변경
                # 단일 인자인 경우 첫 번째 인자만 저장, 그렇지 않으면 전체 인자 튜플 저장
                result['args'] = args[0] if len(args) == 1 else args
                event_loop.exit()
            self._emitter.once(event_id, fn)
            event_loop.exec_()

        #**del
        # def fn(*args):  # 가변 인자로 변경
        #     # 단일 인자인 경우 첫 번째 인자만 저장, 그렇지 않으면 전체 인자 튜플 저장
        #     result['args'] = args[0] if len(args) == 1 else args
        # self._emitter.once(event_id, fn)
        #
        # while result['args'] is None:
        #     pythoncom.PumpWaitingMessages()

        return result['args']  # 저장된 결과 반환

    def _block_tr_condition_until_event_fired(self) -> Union[Any, Tuple[Any, ...]]:
        """
        event_loop 를 이용해 동기 방식으로 값을 가져오는 메소드
        :param event_id:
        :return:
            # 예: fn(data) 처럼 data 인자가 하나만 전달될 때
            result = self._block_tr_condition_until_event_fired("event_id")  # result에는 data 값이 들어감

            # 예: fn(data, error_code, message) 처럼 여러 인자가 전달될 때
            result = self._block_tr_condition_until_event_fired("event_id")  # result에는 (data, error_code, message) 튜플이 들어감
            data, error_code, message = result  # 필요시 언패킹하여 사용 가능
        """
        result = {'args': (None, None)}  # 모든 인자를 저장할 수 있도록 변경

        with self._lock_4_tr_condition:
            event_loop = QEventLoop()
            def fn(*args):  # 가변 인자로 변경
                # 단일 인자인 경우 첫 번째 인자만 저장, 그렇지 않으면 전체 인자 튜플 저장
                result['args'] = args[0] if len(args) == 1 else args
                event_loop.exit()
            self._emitter.once(KoaPySimpleEvent.CONDITION_LIST, fn)
            event_loop.exec_()

        #**del
        # def fn(*args):  # 가변 인자로 변경
        #     # 단일 인자인 경우 첫 번째 인자만 저장, 그렇지 않으면 전체 인자 튜플 저장
        #     result['args'] = args[0] if len(args) == 1 else args
        # self._emitter.once(event_id, fn)
        #
        # while result['args'] is None:
        #     pythoncom.PumpWaitingMessages()

        return result['args']  # 저장된 결과 반환

    def get_deposit(self, account_no: str) -> dict:
        rqname = "예수금상세현황요청"
        if self.is_simulation_mode:
            single_data = Simulator.get_tr_message(rqname)["single_data"]
            #**test
            # return {
            #     "예수금": "100000000",
            #     "d+1추정예수금": "0",
            #     "d+2추정예수금": "0",
            #     "출금가능금액": "100000000",
            #     "d+1출금가능금액": "0",
            #     "d+2출금가능금액": "0",
            # }
        else:
            with self._lock_4_tr:
                KoaProxy.SetInputValue("계좌번호", account_no)
                KoaProxy.SetInputValue("비밀번호", "")  # 계좌비밀번호 입력창을 통해 조회에 사용한 계좌번호의 비밀번호를 입력하십시오. (44)
                                                                # https://autostock.tistory.com/9?category=718663
                KoaProxy.SetInputValue("비밀번호입력매체구분", "00")
                KoaProxy.SetInputValue("조회구분", "3")  # 3: 추정조회, 2: 일반조회

                KoaProxy.CommRqData(rqname, "opw00001", 0, ScreenNo.tr_data())

            single_data, _ = self._block_tr_until_event_fired(rqname)

        if single_data:
            exclude_keys = []
            row = drop_keys(single_data, exclude_keys)
            for key, value in row.items():
                if value == '':
                    single_data[key] = 0
                    continue
                single_data[key] = _to_readable_number(value)

        return single_data

    def get_equity_balance(self, account_no: str) -> dict:
        rqname = "계좌평가잔고내역요청"

        if self.is_simulation_mode:
            return {
                "total": {
                    "총매입금액": "0", "총평가금액": "0", "총평가손익금액": "0",
                    "총수익률(%)": "0", "추정예탁자산": "0",
                },
                "stocks": []
            }
        else:
            with self._lock_4_tr:
                fid_list = ";".join(FidList.get_all_values("잔고"))
                KoaProxy.SetRealReg(ScreenNo.tr_data(), account_no, fid_list, "1")

                KoaProxy.SetInputValue("계좌번호", account_no)
                KoaProxy.SetInputValue("비밀번호", "")  # 계좌비밀번호 입력창을 통해 조회에 사용한 계좌번호의 비밀번호를 입력하십시오. (44)
                                                                # https://autostock.tistory.com/9?category=718663
                KoaProxy.SetInputValue("비밀번호입력매체구분", "00")
                KoaProxy.SetInputValue("조회구분", "2")  # 1: 추정조회, 2: 일반조회

                KoaProxy.CommRqData(rqname, "opw00018", 0, ScreenNo.tr_data())

            single_data, multi_data_list = self._block_tr_until_event_fired(rqname)

        if single_data:
            for key, value in single_data.items():
                if value == '':
                    single_data[key] = "0"
                    continue
                if key.startswith("총수익률"):
                    single_data[key] = _to_readable_number(value, 1)
                else:
                    single_data[key] = _to_readable_number(value)

        if multi_data_list:
            exclude_keys = ['종목번호', '종목명', '신용구분', '신용구분명', '대출일']
            for multi_data in multi_data_list:
                row = drop_keys(multi_data, exclude_keys)
                for key, value in row.items():
                    if value == '':
                        multi_data[key] = "0"
                        continue
                    if key.startswith("수익률"):
                        multi_data[key] = _to_readable_number(value, 1)
                    else:
                        multi_data[key] = _to_readable_number(value)

        return {
            "total": single_data,
            "stocks": multi_data_list,
        }

    def get_equity_balance2(self, account_no: str) -> dict:
        rqname = "계좌평가현황요청"

        if self.is_simulation_mode:
            return {}
        else:
            with self._lock_4_tr:
                KoaProxy.SetInputValue("계좌번호", account_no)
                KoaProxy.SetInputValue("비밀번호", "")  # 계좌비밀번호 입력창을 통해 조회에 사용한 계좌번호의 비밀번호를 입력하십시오. (44)
                                                                # https://autostock.tistory.com/9?category=718663
                KoaProxy.SetInputValue("상장폐지조회구분", "0")  # 0:전체, 1:상장폐지종목제외
                KoaProxy.SetInputValue("비밀번호입력매체구분", "00")

                KoaProxy.CommRqData(rqname, "opw00004", 0, ScreenNo.tr_data())

            single_data, multi_data_list = self._block_tr_until_event_fired(rqname)

        if single_data:
            for key, value in single_data.items():
                if value == '':
                    single_data[key] = 0
                    continue
                if key.startswith("총수익률"):
                    single_data[key] = _to_readable_number(value, 1)
                else:
                    single_data[key] = _to_readable_number(value)

        if multi_data_list:
            exclude_keys = ['종목번호', '종목명', '신용구분', '신용구분명', '대출일']
            for multi_data in multi_data_list:
                row = drop_keys(multi_data, exclude_keys)
                for key, value in row.items():
                    if value == '':
                        multi_data[key] = 0
                        continue
                    if key.startswith("수익률"):
                        multi_data[key] = _to_readable_number(value, 1)
                    else:
                        multi_data[key] = _to_readable_number(value)

        return {
            "total": single_data,
            "stocks": multi_data_list,
        }

    def get_stock_basic_info(self, code, listener=None):
        """
        주식 기본 정보를 요청하는 메소드
        :param code: 종목코드
        :param listener: 데이터를 처리할 리스너 None 이라면 동기식(블러킹)으로 처리
        """
        rqname = f"주식기본정보요청:{code}"

        if self.is_simulation_mode:
            return Simulator.get_tr_message(rqname)["single_data"]
            #**test
            # return {
            #     "종목명": "시뮬레이션",
            #     "기준가": "1000",
            #     "현재가": "1020",
            #     "시가": "1000",
            #     "고가": "1100",
            #     "저가": "990",
            #     "상한가": "1300",
            #     "하한가": "700",
            #     "영업이익": "200",
            #     "신용비율": "1.35",  # %
            #     "유통주식": "20600",  # 천단위
            #     "유통비율": "56.1",  # %
            # }
        else:
            with self._lock_4_tr:
                KoaProxy.SetInputValue("종목코드", code)
                KoaProxy.CommRqData(rqname, "opt10001", 0, ScreenNo.tr_data())

            if listener is None:
                single_data, _ = self._block_tr_until_event_fired(rqname)
                return single_data
            else:
                self._emitter.once(rqname, lambda single_data_, _: listener(single_data_))
                return None

    def get_upper_limit_price(self, code, listener=None):
        """
        상한가를 숫자 형식으로 가져오는 메소드
        :param code: 종목코드
        :param listener: 데이터를 처리할 리스너 None 이라면 동기식(블러킹)으로 처리
        :return:
        """
        data = self.get_stock_basic_info(code, listener)
        if not data:
            return -1
        price = data['상한가']
        return int(price)

    def get_current_price(self, code, listener=None):
        """
        현재가를 숫자 형식으로 가져오는 메소드
        :param code: 종목코드
        :param listener: 데이터를 처리할 리스너 None 이라면 동기식(블러킹)으로 처리
        :return:
        """
        data = self.get_stock_basic_info(code, listener)
        if not data:
            return -1
        price = data['현재가']
        return int(price)

    def get_stocks_of_upper_limit_reached(self, listener=None):
        """
        전일 상한가 존목들
        :param listener: 데이터를 처리할 리스너 None 이라면 동기식(블러킹)으로 처리
        """
        rqname = "상하한가요청"

        if self.is_simulation_mode:
            return Simulator.get_tr_message(rqname)["multi_data_list"]
            #**test
            # return []
        else:
            with self._lock_4_tr:
                # 시장구분 = 000:전체, 001:코스피, 101:코스닥
                KoaProxy.SetInputValue("시장구분", "000")
                # 상하한구분 = 1:상한, 2:상승, 3:보합, 4: 하한, 5:하락, 6:전일상한, 7:전일하한
                KoaProxy.SetInputValue("상하한구분", "6")
                # 정렬구분 = 1:종목코드순, 2:연속횟수순(상위100개), 3:등락률순
                KoaProxy.SetInputValue("정렬구분", "3")
                # 종목조건 = 0:전체조회,1:관리종목제외, 3:우선주제외, 4:우선주+관리종목제외, 5:증100제외, 6:증100만 보기,
                #   7:증40만 보기, 8:증30만 보기,  9:증20만 보기, 10:우선주+관리종목+환기종목제외
                KoaProxy.SetInputValue("종목조건", "1")
                # 거래량조건 = 00000:전체조회, 00010:만주이상, 00050:5만주이상, 00100:10만주이상, 00150:15만주이상, 00200:20만주이상, 00300:30만주이상, 00500:50만주이상, 01000:백만주이상
                KoaProxy.SetInputValue("거래량조건", "00000")
                # 신용조건 = 0:전체조회, 9:신용융자전체, 1:신용융자A군, 2:신용융자B군, 3:신용융자C군, 4:신용융자D군, 8:신용대주, 5:신용한도초과제외
                KoaProxy.SetInputValue("신용조건",  "0")
                # 매매금구분 = 0:전체조회, 1:1천원미만, 2:1천원~2천원, 3:2천원~3천원, 4:5천원~1만원, 5:1만원이상, 8:1천원이상
                KoaProxy.SetInputValue("매매금구분",  "8")

                KoaProxy.CommRqData(rqname, "opt10017", 0, ScreenNo.tr_data())

        if listener is None:
            _, multi_data_list = self._block_tr_until_event_fired(rqname)
            return multi_data_list
        else:
            self._emitter.once(rqname, listener)
            return None

    def get_stocks_of_after_hours_trade(self, listener=None):
        """
        :param listener: 데이터를 처리할 리스너 None 이라면 동기식(블러킹)으로 처리
        """
        rqname = "시간외단일가등락률순위"

        if self.is_simulation_mode:
            return Simulator.get_tr_message(rqname)["multi_data_list"]
            #**test
            # return []
        else:
            with self._lock_4_tr:
                # 시장구분 = 000:전체, 001:코스피, 101:코스닥
                KoaProxy.SetInputValue("시장구분", "000")
                # 정렬기준 = 1:상숭률, 2:상승폭, 3:하락률, 4:하락폭, 5:보합
                KoaProxy.SetInputValue("정렬기준", "1")
                # 종목조건 = 0:전체조회, 1:관리종목제외, 2:정리매매종목제외, 3:우선주제외, 4:관리종목우선주제외, 5:증100 제외,
                #   6:증100만 보기,13:증60만 보기, 12:증50만 보기, 7:증40만 보기, 8:증30만 보기, 9:증20만 보기,
                #   17:ETN제외, 14:ETF제외, 16:ETF+ETN 제외, 15:스팩제외
                KoaProxy.SetInputValue("종목조건", "16")
                # 거래량조건 = 0:전체조회, 10:1백주이상, 50:5백주이상, 100:1천주이상, 500:5천주이상, 1000:1만주이상, 5000:5만주이상, 10000:10만주이상
                KoaProxy.SetInputValue("거래량조건", "1000")
                # 신용조건 = 0:전체조회, 9:신용융자전체, 1:신용융자A군, 2:신용융자B군, 3:신용융자C군, 4:신용융자D군, 8:신용대주, 5:신용한도초과제외
                KoaProxy.SetInputValue("신용조건"	,  "0")
                # 거래대금 = 0:전체조회, 5:5백만원이상, 10:1천만원이상, 30:3천만원이상, 50:5천만원이상, 100:1억원이상, 300:3억원이상,
                #   500:5억원이상, 1000:10억원이상, 3000:30억원이상, 5000:50억원이상, 10000:100억원이상
                KoaProxy.SetInputValue("거래대금"	,  "100")

                KoaProxy.CommRqData(rqname, "opt10098", 0, ScreenNo.tr_data())

        if listener is None:
            _, multi_data_list = self._block_tr_until_event_fired(rqname)
            return multi_data_list
        else:
            self._emitter.once(rqname, listener)
            return None

    def get_stock_state(self, code):
        """
        종목의 종목상태를 반환

        :param code: 종목코드
        :return: ['증거금20%', '담보대출', '신용가능']
        """
        if self.is_simulation_mode:
            return ['증거금20%', '담보대출', '신용가능']
        else:
            with self._lock:
                data = KoaProxy.GetMasterStockState(code)
        return data.split("|")

    def get_theme_group_list(self, reverse=True):
        """
        테마 이름과 코드를 반환

        :param reverse: True 이면 {이름: 코드} False 이면 {코드: 이름}
        :return:
        """
        if self.is_simulation_mode:
            return []
        else:
            with self._lock:
                data = KoaProxy.GetThemeGroupList(type=reverse if 1 else 0)
        return data

    def get_stock_name(self, code) -> str:
        if self.is_simulation_mode:
            return "시뮬레이션"
        else:
            with self._lock:
                return KoaProxy.GetMasterCodeName(code)

    #
    # Real data
    #

    def start_real_data(self, code):
        if code in self._registered_realtime_data:
            self._registered_realtime_data[code] += 1
            return
        self._registered_realtime_data[code] = 1

        if self.is_simulation_mode:
            def handler(message_type, message):
                code_ = message["주식코드"]
                self._emitter.emit(
                    code_ + KoaPySimpleEvent.REAL_DATA, code_, message_type, message)
            Simulator.add_message_listener("주식체결", handler)
            Simulator.add_message_listener("주식호가잔량", handler)
            Simulator.add_message_listener("주식당일거래원", handler)
            Simulator.add_message_listener("종목프로그램매매", handler)
        else:
            with self._lock_4_real:
                combined_values = set(
                    FidList.get_all_values("주식체결") + FidList.get_all_values("주식호가잔량") +
                    FidList.get_all_values("주식당일거래원") + FidList.get_all_values("종목프로그램매매")
                )
                fid_list = ";".join(combined_values)
                QTimer.singleShot(0,
                                  lambda: KoaProxy.SetRealReg(ScreenNo.real_data(), code, fid_list, "1"))

        self._logger.debug(f"종목코드: {code} 실시간 데이터 시작")

    def stop_real_data(self, code):
        if code not in self._registered_realtime_data:
            return
        cnt = self._registered_realtime_data[code] - 1
        if cnt > 0:
            self._registered_realtime_data[code] = cnt
        else:
            with self._lock_4_real:
                del self._registered_realtime_data[code]
                QTimer.singleShot(0,
                                  lambda: KoaProxy.SetRealRemove("ALL", code))
            self._logger.debug(f"종목코드: {code} 실시간 데이터 취소")

    #
    # Condition search
    #

    def get_stocks_by_condition(self, condition_name, listener=None):
        if self.is_simulation_mode:
            data = Simulator.get_condition_message(condition_name)
            return data

        condition_list = self.get_condition_name_list()
        for index, name in condition_list:
            if name == condition_name:
                with self._lock_4_tr_condition:
                    KoaProxy.SendCondition(ScreenNo.condition(), condition_name, int(index), 1)  # 1: 실시간 조건검색
                    self._logger.debug(f"조건 '{condition_name}' 실시간 조회 시작.")
                    if listener:
                        self._emitter.once(KoaPySimpleEvent.CONDITION_LIST, listener)
                    else:
                        return self._block_tr_condition_until_event_fired()
                    break
        else:
            self._logger.debug(f"조건 '{condition_name}'을 찾을 수 없습니다.")

    def get_condition_name_list(self):
        """
        :return: [('000', 'me-전고점 근접'), ('017', 'me-VI따-상따')
        """
        if self.is_simulation_mode:
            return []
        else:
            with self._lock:
                KoaProxy.GetConditionLoad()
            self._condition_event_loop.exec_()

        conditions = KoaProxy.GetConditionNameList()
        return conditions

    #
    # Order
    #

    def send_order(self, request_name: str,
                   account_no: str, code: str, quantity: int, price: int,
                   order_type: OrderType,
                   orderbook_type: OrderBookType,
                   original_order_no: str = ""):
        """
        증권사에 주문을 전송한다.

        [개요]
        OpenAPI를 이용하면 국내주식과 코스피200 지수선물/옵션을 거래할 수 있습니다.
        상품별로 전용 주문함수가 있으며 SendOrderCredit()함수를 이용해서 대주를 제외한 신용주문을 지원합니다.
        ※ 정정주문은 원주문에 대한 가격정정만 가능하며 거래구분을 변경하는 정정주문은 지원하지 않습니다.

        [계좌비밀번호 설정]
        OpenAPI는 로그인 후 한번 계좌비밀번호를 입력/등록 해야 합니다.
        계좌비밀번호 설정은 계좌비밀번호 입력창에서만 가능합니다.
        이 입력창을 출력하는 방법은 2가지로 제공됩니다.
        1. 메뉴이용 - 로그인후 윈도우의 작업표시줄상에 깜박이는 트레이아이콘의
          마우스우측 메뉴(모니터 오른쪽 하단)에서 "계좌비밀번호 저장" 선택
        2. 함수이용 - 로그인후 OpenAPI.KOA_Functions(_T("ShowAccountWindow"), _T("")) 호출
        ※ 계좌비밀번호 입력시 주의할 점 :
        OpenAPI는 계좌비밀번호를 검증하지 않고 입력된 값을 그대로 암호화하여 서버로 전송합니다.
        계좌비밀번호 오류가 일어나지 않도록 오타 등 입력에 주의하시기 바랍니다.


        [주문처리단계]
        주문 처리 순서
        SendOrder(주문발생) -> OnReceiveTRData(주문응답) -> OnReceiveMsg(주문메세지수신) -> OnReceiveChejan(주문접수/체결)
        ※ 주의(역전현상) : 주문건수가 폭증하는 경우 OnReceiveChejan 이벤트가 OnReceiveTRData 이벤트보다 앞서 수신될 수 있습니다.

        각 단계 설명
        SendOrder - 사용자가 호출. 리턴값 0인 경우 함수호출 정상 (주문성공이 아님)
        OnReceiveTRData - 주문발생시 첫번째 서버응답. 주문번호 취득 (주문번호가 없다면 주문거부 등 비정상주문)
        OnReceiveMsg - 주문거부 사유를 포함한 서버메세지 수신
        OnReceiveChejan - 주문 상태에따른 실시간수신 (주문접수, 주문체결, 잔고변경 각 단계별로 수신됨)

        주문성공 여부 판단
        OnReceiveTRData()이벤트는 주로 조회요청후 데이터수신 이벤트지만 주문시에도 발생됩니다.
        주문정상인 경우 이 이벤트내부에서 주문번호를 얻을 수 있습니다.
        비정상주문(주문실패)인 경우 주문번호는 공백("")으로 전달됩니다.
        각 주문함수의 리턴값이 0(성공)이여도 장 개시전 주문, 시장가 주문가격입력, 호가범위를 벗어난 주문가격 입력등
        주문은 다양한 원인으로 실패할수 있습니다.

        주문성공여부 판단 예시
        OnReceiveTRData(sScreenNo, sRqName, sTrCode, ....) // 이벤트 처리부분
        {
            sData = OpenAPI.GetCommData(sTrCode, sRqName, 0, "주문번호");// sData에 주문번호가 있으면 주문성공, 공백이면 주문실패
        }

        [주문체결, 잔고 수신]
        OnReceiveChejan()이벤트는 주문접수, 체결, 잔고변경시 발생됩니다.
        이 이벤트를 통해 대부분의 주문관련 정보를 얻을 수 있습니다.
        주문요청에 대한 응답은 주문접수, 주문체결, 잔고수신 순서로 진행됩니다.
        (정정/취소 주문의 경우 주문접수 후에 주문확인 신호가 한번 더 수신됩니다.)
        하나의 주문에대해 부분체결되는 경우 아래와 같은 순서로 OnReceiveChejan 이벤트가 발생됩니다.

        주문 ---> 접수 ---> 체결1 ---> 잔고1  ---> 체결2  ---> 잔고2... ---> 체결n  ---> 잔고n

        주문 응답의 구분은 OnReceiveChejanData()이벤트가 발생될때 전달되는 sGubun값을 이용합니다.
        sGubun값은 접수와 체결시 '0'값, 잔고변경은 '1'값을 가지게 됩니다.
        이값에 따라 실시간타입 "주문체결" 또는 "잔고" 타입이 사용됩니다.

        Args:
            request_name (str): 구분을 위한 임의의 요청명
            account_no (str): 계좌번호 10자리, 여기서는 계좌번호 목록에서 첫번째로 발견한 계좌번호로 매수처리
            code (str): 종목코드
            quantity (int): 수량
            price (int): 주문가격, 시장가 매수는 가격 설정 의미 없으므로 기본값 0 으로 설정
            order_type (OrderType): 주문 유형
            orderbook_type (OrderBookType): 호가 유형
            original_order_no (str): 원주문번호, 주문 정정/취소 등에서 사용
        """

        self._emitter.emit(KoaPySimpleEvent.ORDER_ORDERING, code, order_type, orderbook_type)
        if self.is_simulation_mode:
            def handler(message_type, message):
                code_ = message["종목코드,업종코드"][1:]
                if message_type == "주문접수":
                    self._emitter.emit(
                        code_ + KoaPySimpleEvent.ORDER_RECEIPTED, message)
                else:
                    self._emitter.emit(
                        code_ + KoaPySimpleEvent.ORDER_CONTRACTED, message)

            Simulator.add_message_listener("주문접수", handler)
            Simulator.add_message_listener("주문체결", handler)
        else:
            with self._lock_4_order:
                KoaProxy.SendOrder(request_name, ScreenNo.order(), account_no,
                                   order_type.value, code, quantity, price,
                                   orderbook_type.value, original_order_no)

    #
    # Simulation
    #
    @property
    def is_simulation_mode(self):
        return self._simulation
