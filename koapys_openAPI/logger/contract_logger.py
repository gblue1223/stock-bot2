import os
import threading
import time
import pandas as pd

from datetime import datetime
from lib.io import ensure_directory_exists
from lib.pd import append as pd_append
from koapys.koa_py_simple import KoaPySimple

data_map = {}


class DataContainer:
    def __init__(self, code, name):
        now = datetime.now()
        self.code = code
        self.name = name
        self.filename = now.strftime(f"contracts_logs/{code}_{name}_%Y%m%d_%H%M%S.csv")
        self.last_save_time = time.time()  # 마지막 저장 시간 추가
        self.df_contract = None
        self.df_orderbook = None
        self.latest_row = None
        self.lock = threading.RLock()

    @property
    def empty(self):
        return (self.df is None) or self.df.empty

    def write_to_file(self):
        df = None
        with self.lock:
            df = self._merge_contract_and_orderbook_logs()
            if df is None:
                return

            ensure_directory_exists(self.filename)
            write_header = not os.path.exists(self.filename)

            df.to_csv(self.filename, encoding='utf-8-sig',
                      mode='a', index=False,
                      header=write_header)
            print(f'{self.filename} saved')

    def add_contract_data(self, columns, values, save_threshold=300, save_interval=60):
        with self.lock:
            if self.df_contract is None:
                self.df_contract = pd.DataFrame.from_dict(dict(zip(columns, values)), orient='index').T
            else:
                self.df_contract = self._add_data(self.df_contract, columns, values, save_threshold, save_interval)

    def add_orderbook_data(self, columns, values, save_threshold=300, save_interval=60):
        with self.lock:
            if self.df_orderbook is None:
                self.df_orderbook = pd.DataFrame.from_dict(dict(zip(columns, values)), orient='index').T
            else:
                self.df_orderbook = self._add_data(self.df_orderbook, columns, values, save_threshold, save_interval)

    def _add_data(self, df, columns, values, save_threshold=300, save_interval=60):
        if self.df_contract is None or self.df_orderbook is None:
            return

        if self.latest_row is None:
            merged_columns = list(self.df_contract.columns) + list(self.df_orderbook.columns)
            self.latest_row = pd.Series(index=merged_columns, dtype='object')

        # values 리스트에서 "--"를 "-"로 바꿈
        # values = ["-" if value == "--" else value for value in values]
        # values = ["+" if value == "+-" else value for value in values]
        #
        # 거래량 issue (https://bbn.kiwoom.com/m/bbs/VBbsBoardBWOAZDetailView)
        # [주식예상체결]
        #   첫번째 부호는 매수/매도 예상체결을 나타내며 반드시 붙어있고 두 번째 부호는 감소만(음수만)붙여줍니다.
        #   +4500  = (+)매수예상체결, (+)직전누적거래량 대비 4500 증가, 여기서 직전누적거래량 증가부호(+)는 생략됨.
        #   +-4500 = (+)매수예상체결, (-)직전누적거래량 대비 4500 감소
        #   -4500  = (-)매도예상체결, (+)직전누적거래량 대비 4500 증가, 여기서 직전누적거래량 증가부호(+)는 생략됨.
        #   --4500 = (-)매도예상체결, (-)직전누적거래량 대비 4500 감소
        values[6] = values[6].replace("+-", "+")
        values[6] = values[6].replace("--", "-")
        row = dict(zip(columns, values))
        self.latest_row.update(row)

        df = pd_append(df, pd.Series(row, dtype='object'))
        now = time.time()

        # 데이터 개수가 임계값을 넘거나 마지막 저장 후 save_interval 초가 지났을 때 저장
        if len(df) >= save_threshold or (now - self.last_save_time) >= save_interval:  # 60초 = 1분
            self.last_save_time = now
            thread = threading.Thread(target=self.write_to_file)
            thread.start()

        return df

    def _merge_contract_and_orderbook_logs(self):
        df1 = self.df_contract
        df2 = self.df_orderbook
        if df1 is None or df2 is None:
            return None

        df1 = df1.drop_duplicates()
        df2 = df2.drop_duplicates()

        merged_df = pd.DataFrame(columns=list(df1.columns) + list(df2.columns))
        new_row = self.latest_row

        i = n = 0
        df1_len = len(df1)
        df2_len = len(df2)
        max_len = max(df1_len, df2_len)
        row1 = row2 = time1 = time2 = None
        while i < max_len and n < max_len:
            if i < df1_len:
                row1 = df1.iloc[i]
                time1 = row1['체결시간']

            if n < df2_len:
                row2 = df2.iloc[n]
                time2 = row2['호가시간']

            if (row1 is not None) and (row2 is not None):
                if time1 == time2:
                    new_row.update(row1)
                    new_row.update(row2)
                    merged_df = pd_append(merged_df, new_row)
                    i += 1
                    n += 1
                else:
                    if int(time1) < int(time2):
                        new_row.update(row1)
                        merged_df = pd_append(merged_df, new_row)
                        i += 1
                    else:
                        new_row.update(row2)
                        merged_df = pd_append(merged_df, new_row)
                        n += 1
            else:
                if row1 is not None:
                    new_row.update(row1)
                    merged_df = pd_append(merged_df, new_row)
                    i += 1
                else:
                    new_row.update(row2)
                    merged_df = pd_append(merged_df, new_row)
                    n += 1

        # 데이터를 옮긴 후 초기화
        self.df_contract = pd.DataFrame(columns=self.df_contract.columns)
        self.df_orderbook = pd.DataFrame(columns=self.df_orderbook.columns)

        return merged_df.drop_duplicates()


def add_logs(koapys: KoaPySimple, code, columns, values, save_threshold=300):
    if code not in data_map:
        data_map[code] = DataContainer(
            code=code,
            name='test' if koapys is None else koapys.get_stock_name(code),
        )
    data_container = data_map[code]

    if "체결시간" in columns:
        data_container.add_contract_data(columns, values, save_threshold=save_threshold)
    elif "호가시간" in columns:
        data_container.add_orderbook_data(columns, values, save_threshold=save_threshold)
