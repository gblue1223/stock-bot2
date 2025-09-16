import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel, QFileDialog, QTableWidget, QTableWidgetItem
)
from koapys.koa_py_simple import KoaPySimple
from koapys.logger.logger import logging, QTextEditHandler


class App(QWidget):
    def __init__(self):
        super().__init__()

        self._title = "KoaPySimple App"

        self._init_ui()
        self._init_logging()
        self._init_koapys()
        self._logger.debug("KoaPySimple App Ready")

    def _init_koapys(self):
        self.koapys = KoaPySimple(self._logger)
        self.koapys.ensure_connected()

    def _init_logging(self):
        # 로거 설정
        self._logger = logging.getLogger(self._title)
        self._logger.setLevel(logging.DEBUG)

        # QTextEdit 핸들러 추가
        text_edit_handler = QTextEditHandler(self.log_viewer)
        self._logger.addHandler(text_edit_handler)

    def _init_ui(self):
        self.setWindowTitle(self._title)
        self.setGeometry(300, 300, 400, 300)

        # Layout
        layout = QVBoxLayout()

        # Text area to show logs
        self.log_viewer = QTextEdit(self)
        self.log_viewer.setReadOnly(True)
        layout.addWidget(self.log_viewer)

        self.setLayout(layout)

    def set_log_level(self, log_level: str):
        """
        :param log_level:
            CRITICAL
            FATAL
            ERROR
            WARNING
            WARN
            INFO
            DEBUG
            NOTSET
        :return:
        """
        name_to_level = {
            'CRITICAL': logging.CRITICAL,
            'FATAL': logging.FATAL,
            'ERROR': logging.ERROR,
            'WARN': logging.WARNING,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
            'NOTSET': logging.NOTSET,
        }
        self._logger.setLevel(name_to_level[log_level])


if __name__ == '__main__':
    qapp = QApplication(sys.argv)

    app = App()
    app.show()
    # koapys_app.set_log_level("ERROR")

    code = "024740"
    app.koapys.get_stock_basic_info(code, lambda data: {
        print(data)
    })
    app.koapys.get_stocks_by_condition("me-25%이상")

    sys.exit(qapp.exec_())
