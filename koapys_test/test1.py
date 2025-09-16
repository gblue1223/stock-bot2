import sys

from PyQt5.QtWidgets import *
from koapys.koa_py_simple import KoaPySimple
from lib.schedule import set_timeout


if __name__ == "__main__":
    app = QApplication(sys.argv)

    kiwoom = KoaPySimple()
    kiwoom.ensure_connected()
    # account_info = kiwoom.get_account_list()

    # print(kiwoom.get_condition_name_list())
    # print(kiwoom.get_theme_group_list())
    # kiwoom.get_stocks_by_condition("me-10%이상")

    set_timeout(lambda: {}, 2)
    code = "024740"
    kiwoom.get_stock_basic_info(code, lambda data: {
        print(111, data)
    })
    # delay(5, lambda: {})

    kiwoom.start_real_data(code)
    set_timeout(lambda: kiwoom.stop_real_data(code), 5)

    sys.exit(app.exec_())
