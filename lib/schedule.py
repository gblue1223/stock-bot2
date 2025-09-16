import threading
from datetime import datetime, timedelta


def schedule_script(target_hour, target_minute, target_sec=0, listener=None):
    now = datetime.now()
    target_time = now.replace(
        hour=target_hour, minute=target_minute, second=target_sec, microsecond=0)

    delay = (target_time - now).total_seconds()
    if delay < 0:
        delay = 0

    timer_thread = threading.Timer(delay, listener)
    timer_thread.start()


def set_timeout(listener, seconds):
    timer_thread = threading.Timer(seconds, listener)
    timer_thread.start()


def set_interval(listener, interval):
    def wrapper():
        set_interval(listener, interval)  # 다시 실행
        listener()  # 주어진 함수 실행

    timer = threading.Timer(interval, wrapper)
    timer.start()
    return timer
