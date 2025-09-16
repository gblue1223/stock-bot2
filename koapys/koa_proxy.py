"""
refer: https://github.com/sharebook-kr/pykiwoom
"""
import time
from datetime import datetime
from PyQt5.QAxContainer import *


class KoaProxy:
    ocx = None

    _delay = 0.2
    _last_tr_call_time = 0.0
    _last_order_call_time = 0.0

    @staticmethod
    def _check_tr_delay():
        current_time = time.time()
        if current_time - KoaProxy._last_tr_call_time < KoaProxy._delay:
            time.sleep(KoaProxy._delay - (current_time - KoaProxy._last_tr_call_time))
        KoaProxy._last_tr_call_time = time.time()

    @staticmethod
    def _check_order_delay():
        current_time = time.time()
        if current_time - KoaProxy._last_order_call_time < KoaProxy._delay:
            time.sleep(KoaProxy._delay - (current_time - KoaProxy._last_order_call_time))
        KoaProxy._last_order_call_time = time.time()

    @staticmethod
    def Init():
        """
        로그인 윈도우를 실행합니다.
        :return: None
        """
        if KoaProxy.ocx is None:
            KoaProxy.ocx = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")

    @staticmethod
    def CommConnect():
        """
        로그인 윈도우를 실행합니다.
        :return: 0 이면 성공
        """
        return KoaProxy.ocx.dynamicCall("CommConnect()")

    @staticmethod
    def CommRqData(rqname, trcode, next, screen):
        """
        TR을 서버로 송신합니다.
        :param rqname: 사용자가 임의로 지정할 수 있는 요청 이름
        :param trcode: 요청하는 TR의 코드
        :param next: 0: 처음 조회, 2: 연속 조회
        :param screen: 화면번호 ('0000' 또는 '0' 제외한 숫자값으로 200개로 한정된 값
        :return: 0 이면 성공
        """
        KoaProxy._check_tr_delay()
        ret = KoaProxy.ocx.dynamicCall("CommRqData(QString, QString, int, QString)", rqname, trcode, next, screen)
        return ret

    @staticmethod
    def GetLoginInfo(tag: str):
        """
        로그인한 사용자 정보를 반환하는 메서드
        :param tag: ("ACCOUNT_CNT, "ACCNO", "USER_ID", "USER_NAME", "KEY_BSECGB", "FIREW_SECGB")
        :return: tag에 대한 데이터 값
        """
        data = KoaProxy.ocx.dynamicCall("GetLoginInfo(QString)", tag)

        if tag == "ACCNO":
            return data.split(';')[:-1]
        else:
            return data

    @staticmethod
    def SendOrder(rqname, screen, accno, order_type, code, quantity, price, hoga, order_no):
        """
        주식 주문을 서버로 전송하는 메서드
        시장가 주문시 주문단가는 0으로 입력해야 함 (가격을 입력하지 않음을 의미)
        :param rqname: 사용자가 임의로 지정할 수 있는 요청 이름
        :param screen: 화면번호 ('0000' 또는 '0' 제외한 숫자값으로 200개로 한정된 값
        :param accno: 계좌번호 10자리
        :param order_type: 1: 신규매수, 2: 신규매도, 3: 매수취소, 4: 매도취소, 5: 매수정정, 6: 매도정정
        :param code: 종목코드
        :param quantity: 주문수량
        :param price: 주문단가
        :param hoga: 00: 지정가, 03: 시장가,
                     05: 조건부지정가, 06: 최유리지정가, 07: 최우선지정가,
                     10: 지정가IOC, 13: 시장가IOC, 16: 최유리IOC,
                     20: 지정가FOK, 23: 시장가FOK, 26: 최유리FOK,
                     61: 장전시간외종가, 62: 시간외단일가, 81: 장후시간외종가
        :param order_no: 원주문번호로 신규 주문시 공백, 정정이나 취소 주문시에는 원주문번호를 입력
        :return:
        """
        KoaProxy._check_order_delay()
        ret = KoaProxy.ocx.dynamicCall("SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
                                       [rqname, screen, accno, order_type, code, quantity, price, hoga, order_no])
        return ret

    @staticmethod
    def SendOrderCredit(rqname, screen, accno, order_type, code, quantity, price, hoga, credit_type, loan_date, order_no):
        """
        주식 주문(신용)을 서버로 전송하는 메서드
        시장가 주문시 주문단가는 0으로 입력해야 함 (가격을 입력하지 않음을 의미)
        :param rqname: 사용자가 임의로 지정할 수 있는 요청 이름
        :param screen: 화면번호 ('0000' 또는 '0' 제외한 숫자값으로 200개로 한정된 값
        :param accno: 계좌번호 10자리
        :param order_type: 1: 신규매수, 2: 신규매도, 3: 매수취소, 4: 매도취소, 5: 매수정정, 6: 매도정정
        :param code: 종목코드
        :param quantity: 주문수량
        :param price: 주문단가
        :param hoga: 00: 지정가, 03: 시장가,
                     05: 조건부지정가, 06: 최유리지정가, 07: 최우선지정가,
                     10: 지정가IOC, 13: 시장가IOC, 16: 최유리IOC,
                     20: 지정가FOK, 23: 시장가FOK, 26: 최유리FOK,
                     61: 장전시간외종가, 62: 시간외단일가, 81: 장후시간외종가
        :param credit_type: 03: 신용매수 - 자기융자
                            33: 신용매도 - 자기융자
                            99: 신용매도 - 자기융자 합
        :param loan_date: 대출일
                          YYYYMMDD형식 날짜를 입력합니다. (ex 대출일이 2023년 1월 1일이면 "20230101"입력)
                          신용매도 - 자기융자 일때는 종목별 대출일을 입력하고 신용매도 - 융자합이면 "99991231"을 입력합니다.
        :param order_no: 원주문번호로 신규 주문시 공백, 정정이나 취소 주문시에는 원주문번호를 입력
        :return:
        """
        KoaProxy._check_order_delay()
        ret = KoaProxy.ocx.dynamicCall("SendOrderCredit(QString, QString, QString, int, QString, int, int, QString, QString, QString, QString)",
                                       [rqname, screen, accno, order_type, code, quantity, price, hoga, credit_type, loan_date, order_no])
        return ret

    @staticmethod
    def SetInputValue(id, value):
        """
        TR 입력값을 설정하는 메서드
        :param id: TR INPUT의 아이템명
        :param value: 입력 값
        :return: None
        """
        KoaProxy.ocx.dynamicCall("SetInputValue(QString, QString)", id, value)

    @staticmethod
    def DisconnectRealData(screen_no):
        """
        화면번호에 대한 리얼 데이터 요청을 해제하는 메서드
        :param screen_no: 화면번호
        :return: None
        """
        KoaProxy.ocx.dynamicCall("DisconnectRealData(QString)", screen_no)

    @staticmethod
    def GetRepeatCnt(trcode, rqname):
        """
        멀티데이터의 행(row)의 개수를 얻는 메서드
        :param trcode: TR코드
        :param rqname: 사용자가 설정한 요청이름
        :return: 멀티데이터의 행의 개수
        """
        count = KoaProxy.ocx.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        return count

    @staticmethod
    def CommKwRqData(arr_code, next, code_count, type, rqname, screen):
        """
        여러 종목 (한 번에 100종목)에 대한 TR을 서버로 송신하는 메서드
        :param arr_code: 여러 종목코드 예: '000020:000040'
        :param next: 0: 처음조회
        :param code_count: 종목코드의 개수
        :param type: 0: 주식종목 3: 선물종목
        :param rqname: 사용자가 설정하는 요청이름
        :param screen: 화면번호
        :return:
        """
        ret = KoaProxy.ocx.dynamicCall("CommKwRqData(QString, bool, int, int, QString, QString)", arr_code, next, code_count, type, rqname, screen);
        return ret

    @staticmethod
    def GetAPIModulePath():
        """
        OpenAPI 모듈의 경로를 반환하는 메서드
        :return: 모듈의 경로
        """
        ret = KoaProxy.ocx.dynamicCall("GetAPIModulePath()")
        return ret

    @staticmethod
    def GetCodeListByMarket(market):
        """
        시장별 상장된 종목코드를 반환하는 메서드
        :param market: 0: 코스피, 3: ELW, 4: 뮤추얼펀드 5: 신주인수권 6: 리츠
                       8: ETF, 9: 하이일드펀드, 10: 코스닥, 30: K-OTC, 50: 코넥스(KONEX)
        :return: 종목코드 리스트 예: ["000020", "000040", ...]
        """
        data = KoaProxy.ocx.dynamicCall("GetCodeListByMarket(QString)", market)
        tokens = data.split(';')[:-1]
        return tokens

    @staticmethod
    def GetConnectState():
        """
        현재접속 상태를 반환하는 메서드
        :return: 0:미연결, 1: 연결완료
        """
        ret = KoaProxy.ocx.dynamicCall("GetConnectState()")
        return ret

    @staticmethod
    def GetMasterCodeName(code):
        """
        종목코드에 대한 종목명을 얻는 메서드
        :param code: 종목코드
        :return: 종목명
        """
        data = KoaProxy.ocx.dynamicCall("GetMasterCodeName(QString)", code)
        return data

    @staticmethod
    def GetMasterListedStockCnt(code):
        """
        종목에 대한 상장주식수를 리턴하는 메서드
        :param code: 종목코드
        :return: 상장주식수
        """
        data = KoaProxy.ocx.dynamicCall("GetMasterListedStockCnt(QString)", code)
        return data

    @staticmethod
    def GetMasterConstruction(code):
        """
        종목코드에 대한 감리구분을 리턴
        :param code: 종목코드
        :return: 감리구분 (정상, 투자주의 투자경고, 투자위험, 투자주의환기종목)
        """
        data = KoaProxy.ocx.dynamicCall("GetMasterConstruction(QString)", code)
        return data

    @staticmethod
    def GetMasterListedStockDate(code):
        """
        종목코드에 대한 상장일을 반환
        :param code: 종목코드
        :return: 상장일 예: "20100504"
        """
        data = KoaProxy.ocx.dynamicCall("GetMasterListedStockDate(QString)", code)
        return datetime.datetime.strptime(data, "%Y%m%d")

    @staticmethod
    def GetMasterLastPrice(code):
        """
        종목코드의 전일가를 반환하는 메서드
        :param code: 종목코드
        :return: 전일가
        """
        data = KoaProxy.ocx.dynamicCall("GetMasterLastPrice(QString)", code)
        return int(data)

    @staticmethod
    def GetMasterStockState(code):
        """
        종목의 종목상태를 반환하는 메서드
        :param code: 종목코드
        :return: 종목상태
        """
        data = KoaProxy.ocx.dynamicCall("GetMasterStockState(QString)", code)
        return data.split("|")

    @staticmethod
    def GetDataCount(record):
        count = KoaProxy.ocx.dynamicCall("GetDataCount(QString)", record)
        return count

    @staticmethod
    def GetOutputValue(record, repeat_index, item_index):
        count = KoaProxy.ocx.dynamicCall("GetOutputValue(QString, int, int)", record, repeat_index, item_index)
        return count

    @staticmethod
    def GetCommData(trcode, rqname, index, item):
        """
        수순 데이터를 가져가는 메서드
        :param trcode: TR 코드
        :param rqname: 요청 이름
        :param index: 멀티데이터의 경우 row index
        :param item: 얻어오려는 항목 이름
        :return:
        """
        data = KoaProxy.ocx.dynamicCall("GetCommData(QString, QString, int, QString)", trcode, rqname, index, item)
        return data.strip()

    @staticmethod
    def GetCommRealData(code, fid):
        data = KoaProxy.ocx.dynamicCall("GetCommRealData(QString, int)", code, fid)
        return data

    @staticmethod
    def GetChejanData(fid):
        data = KoaProxy.ocx.dynamicCall("GetChejanData(int)", fid)
        return data

    @staticmethod
    def GetThemeGroupList(type=1):
        data = KoaProxy.ocx.dynamicCall("GetThemeGroupList(int)", type)
        tokens = data.split(';')
        if type == 0:
            grp = {x.split('|')[0]:x.split('|')[1] for x in tokens}
        else:
            grp = {x.split('|')[1]: x.split('|')[0] for x in tokens}
        return grp

    @staticmethod
    def GetThemeGroupCode(theme_code):
        data = KoaProxy.ocx.dynamicCall("GetThemeGroupCode(QString)", theme_code)
        data = data.split(';')
        return [x[1:] for x in data]

    @staticmethod
    def GetFutureList():
        data = KoaProxy.ocx.dynamicCall("GetFutureList()")
        return data

    @staticmethod
    def SetRealReg(screen, code_list, fid_list, opt_type):
        ret = KoaProxy.ocx.dynamicCall("SetRealReg(QString, QString, QString, QString)", screen, code_list, fid_list, opt_type)
        return ret

    @staticmethod
    def SetRealRemove(screen, del_code):
        ret = KoaProxy.ocx.dynamicCall("SetRealRemove(QString, QString)", screen, del_code)
        return ret

    @staticmethod
    def GetConditionLoad():
        KoaProxy.ocx.dynamicCall("GetConditionLoad()")

    @staticmethod
    def GetConditionNameList():
        """
        :return: [('000', 'perpbr'), ('001', 'macd'), ...]
        """
        data = KoaProxy.ocx.dynamicCall("GetConditionNameList()")
        conditions = data.split(";")[:-1]

        result = []
        for condition in conditions:
            cond_index, cond_name = condition.split('^')
            result.append((cond_index, cond_name))

        return result

    @staticmethod
    def SendCondition(screen, cond_name, cond_index, search):
        """
        조건검색 종목조회 TR을 송신

        :param screen (str): 화면번호
        :param cond_name (str): 조건명
        :param cond_index (int): 조건명 인덱스
        :param search (int): 0: 일반조회, 1: 실시간조회, 2: 연속조회

        :return None
        """
        KoaProxy._check_tr_delay()
        KoaProxy.ocx.dynamicCall("SendCondition(QString, QString, int, int)", screen, cond_name, cond_index, search)

    @staticmethod
    def SendConditionStop(screen, cond_name, index):
        KoaProxy.ocx.dynamicCall("SendConditionStop(QString, QString, int)", screen, cond_name, index)
