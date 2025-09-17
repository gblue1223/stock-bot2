import ast
import math
import re
import threading
import time
import pythoncom

from typing import Any, Callable
from datetime import datetime
from PyQt5.QtCore import QTimer

from koapys.logger.logger import koapysFileOnlyLogger, koapysLogger
from lib.event_emitter import EventEmitter

MessageListener = Callable[[str, Any], Any]


class Simulator:
    _log_file = None
    _log_file_lock = threading.RLock()

    _realtime_data_list = ["주식호가잔량", "주식체결", "주문접수", "주문체결", "잔고변경"]
    _event_emitter = EventEmitter()

    _timer = QTimer()
    _prev_message_timestamp = None

    _is_playing = False
    _playback_speed = 1.0  # Normal speed
    _current_position = 0  # Current position in the log file
    _line_offsets = []  # 라인별 파일 오프셋 저장용 리스트

    #
    # Tr, Real
    #

    @staticmethod
    def log_tr_data(message_type: str, message: str):
        Simulator._log_message(f"tr:{message_type}", message)

    @staticmethod
    def get_tr_message(message_type: str):
        return Simulator._get_message(f"tr:{message_type}")

    @staticmethod
    def log_real_data(message_type: str, message: str):
        Simulator._log_message(f"real:{message_type}", message)

    @staticmethod
    def get_real_message(message_type: str):
        return Simulator._get_message(f"real:{message_type}")

    #
    # Order
    #

    @staticmethod
    def log_order_data(condition_name: str, message: str):
        Simulator._log_message(f"order:{condition_name}", message)

    #
    # Condition
    #

    @staticmethod
    def log_condition_data(condition_name: str, message: str):
        Simulator._log_message(f"condition:{condition_name}", message)

    @staticmethod
    def log_condition_added_data(condition_name: str, message: str):
        Simulator._log_message(f"condition+:{condition_name}", message)

    @staticmethod
    def log_condition_removed_data(condition_name: str, message: str):
        Simulator._log_message(f"condition-:{condition_name}", message)

    @staticmethod
    def get_condition_message(condition_name: str):
        return Simulator._get_message(f"condition:{condition_name}")

    #
    # Public
    #

    @staticmethod
    def add_message_listener(message_type: str, listener: MessageListener):
        if message_type in Simulator._realtime_data_list:
            Simulator._event_emitter.on(message_type, listener)
        else:
            Simulator._event_emitter.once(message_type, listener)

    @staticmethod
    def is_playing():
        return Simulator._is_playing

    @staticmethod
    def play():
        """
        Start playing the log file.
        """
        Simulator._is_playing = True
        Simulator._timer.timeout.connect(Simulator._next_line)
        Simulator._timer.start(1)

    @staticmethod
    def stop():
        """
        Pause the playback.
        """
        Simulator.pause()
        Simulator._seek_to_position(0)

    @staticmethod
    def pause():
        """
        Pause the playback.
        """
        Simulator._is_playing = False
        Simulator._timer.stop()

    @staticmethod
    def forward():
        """
        Move forward in the log file.
        """
        Simulator._current_position += 1
        Simulator._next_line()

    @staticmethod
    def backward():
        """
        Move backward in the log file.
        """
        if Simulator._current_position > 0:
            Simulator._current_position -= 1
            Simulator._seek_to_position(Simulator._current_position)

    @staticmethod
    def set_playback_speed(speed: float):
        """
        Set the playback speed.
        """
        Simulator._playback_speed = speed

    @staticmethod
    def close():
        Simulator.stop()
        with Simulator._log_file_lock:
            Simulator._log_file.close()
            Simulator._log_file = None

    @staticmethod
    def open(log_file_path: str):
        """
        로그에서 필요한 데이터를 가져오고 이벤트를 발생시킨다.

        :param log_file_path: 파일 경로
        :return: None
        """
        assert Simulator._log_file is None
        with Simulator._log_file_lock:
            Simulator._log_file = open(log_file_path, 'r', encoding='utf-8')

        Simulator._build_line_offsets()  # 라인 오프셋 초기화
        Simulator._seek_to_position(0)

        with Simulator._log_file_lock:
            # 첫 라인 읽어 시간 초기화
            line = Simulator._log_file.readline()
            pattern = r'\[(.*?)\]\[.*?\] .+'
            match = re.search(pattern, line)
            if match:
                timestamp_str = match.group(1)
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                Simulator._prev_message_timestamp = timestamp

    #
    # Internal
    #

    @staticmethod
    def _log_message(message_type: str, message: str):
        koapysFileOnlyLogger.debug(f"<{message_type}> {message}")

    @staticmethod
    def _get_message(message_type: str):
        def listener(_, message):
            ref["message"] = message
        Simulator.add_message_listener(message_type, listener)

        until = 5 / Simulator._playback_speed
        ref = {
            "start_time": time.time(),
            "message": None
        }
        while not ref["message"]:
            if (time.time() - ref["start_time"]) >= until:
                raise TimeoutError(f"Simulator.get_message.timeout: {str(until)}sec, type={message_type}")
            try:
                #**test
                # Simulator._next_line()
                pythoncom.PumpWaitingMessages()
            except Exception as e:
                koapysLogger.error(f"Simulator.get_message.Error: {e}")

        if not ref["message"]:
            koapysLogger.error(f"Simulator.get_message.Error: No message found")
            raise InterruptedError()
        return ref["message"]

    @staticmethod
    def _build_line_offsets():
        """
        파일을 처음 열 때 라인별 파일 오프셋을 미리 계산하여 저장.
        """
        Simulator._line_offsets = []
        with Simulator._log_file_lock:
            Simulator._log_file.seek(0)
            offset = Simulator._log_file.tell()
            while True:
                line = Simulator._log_file.readline()
                if not line:
                    break
                Simulator._line_offsets.append(offset)
                offset = Simulator._log_file.tell()

    @staticmethod
    def _seek_to_position(position: int):
        """
        파일에서 특정 위치로 빠르게 이동할 수 있도록 라인 오프셋 사용.
        """
        with Simulator._log_file_lock:
            if not Simulator._line_offsets:
                Simulator._build_line_offsets()

            if position < len(Simulator._line_offsets):
                Simulator._log_file.seek(Simulator._line_offsets[position])
            else:
                koapysLogger.error(f"Position {position} out of bounds for the log file.")

    @staticmethod
    def _next_line():
        with Simulator._log_file_lock:
            line = Simulator._log_file.readline()
        if not line:
            return False
        print(line, end="")

        pattern = r'\[(.*?)\]\[.*?\] <(.+?)> ({.+})'
        match = re.search(pattern, line)
        if match:
            timestamp_str = match.group(1)
            message_type = match.group(2)
            message_str = match.group(3)

            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            latency = timestamp - Simulator._prev_message_timestamp
            latency = max(10, math.floor(latency.total_seconds() * 1000 / Simulator._playback_speed))

            Simulator._prev_message_timestamp = timestamp
            Simulator._timer.setInterval(latency)

            try:
                message = ast.literal_eval(message_str)
                Simulator._event_emitter.emit(message_type, message_type, message)
            except (ValueError, SyntaxError) as e:
                koapysLogger.error(f"Simulator._next_line.Error parsing line: {e}")

        return True
