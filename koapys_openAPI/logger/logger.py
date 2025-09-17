import logging
from datetime import datetime
from PyQt5.QtCore import QtDebugMsg, QtInfoMsg, QtWarningMsg, QtCriticalMsg, QtFatalMsg, qInstallMessageHandler, QTimer

from lib.io import ensure_directory_exists

_format = '[%(asctime)s][%(levelname)s] %(message)s'


def custom_message_handler(msg_type, context, message):
    if msg_type == QtDebugMsg:
        msg_type_name = "DEBUG"
    elif msg_type == QtInfoMsg:
        msg_type_name = "INFO"
    elif msg_type == QtWarningMsg:
        msg_type_name = "WARNING"
    elif msg_type == QtCriticalMsg:
        msg_type_name = "CRITICAL"
    elif msg_type == QtFatalMsg:
        msg_type_name = "FATAL"
    else:
        msg_type_name = "UNKNOWN"

    # 원하는 포맷으로 메시지 출력
    koapysLogger.debug(f"[{msg_type_name}] {context.file}:{context.line} - {message}")


qInstallMessageHandler(custom_message_handler)


class QTextEditHandler(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.setFormatter(logging.Formatter(_format))

    def emit(self, record):
        def _emit():
            try:
                msg = self.format(record)
                self.text_edit.append(msg)
            except RuntimeError as e:
                koapysLogger.error(f"QTextEditHandler.Error {e}")
                pass
        QTimer.singleShot(0, _emit)


# 핸들러 설정
#
now = datetime.now()
filepath = now.strftime(f"logs/%Y%m%d_%H%M%S_koapys.log")
ensure_directory_exists(filepath)

fileHandler = logging.FileHandler(filepath, encoding='utf-8')
fileHandler.setFormatter(logging.Formatter(_format))
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(logging.Formatter(_format))

# 기본 로거 설정
#
logging.basicConfig(
    level=logging.DEBUG,  # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format=_format,  # 로그 출력 형식
    datefmt='%Y-%m-%d %H:%M:%S',  # 시간 형식
    handlers=[
        fileHandler,  # 로그를 파일에 기록
        streamHandler  # 콘솔에 출력
    ]
)

koapysLogger = logging.getLogger('KoaPySimple')
koapysLogger.setLevel(logging.DEBUG)

# 파일 전용 로거 설정
#
koapysFileOnlyLogger = logging.getLogger('KoaPySimple.FileOnly')
koapysFileOnlyLogger.setLevel(logging.DEBUG)
koapysFileOnlyLogger.propagate = False  # 상위 로거로 전파하지 않음
koapysFileOnlyLogger.addHandler(fileHandler)
