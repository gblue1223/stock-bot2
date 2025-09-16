import threading


class ThreadSafeDict:
    def __init__(self, initial_dict=None):
        if initial_dict is None:
            initial_dict = {}
        self._dict = initial_dict
        self._lock = threading.RLock()

    def get(self, key, default_value=None):
        with self._lock:
            return self._dict.get(key, default_value)

    def set(self, key, value):
        with self._lock:
            self._dict[key] = value

    def remove(self, key):
        with self._lock:
            return self._dict.pop(key, None)

    def keys(self):
        with self._lock:
            return self._dict.keys()

    def values(self):
        with self._lock:
            return self._dict.values()

    def items(self):
        with self._lock:
            return self._dict.items()

    # 항목 가져오기: self[key]
    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]  # key가 없으면 KeyError 발생

    # 항목 설정: self[key] = value
    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    # 항목 삭제: del self[key]
    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]  # key가 없으면 KeyError 발생

    # + 연산자 오버로딩 (병합)
    def __add__(self, other):
        if not isinstance(other, ThreadSafeDict):
            raise TypeError("Unsupported operand type(s) for +: 'ThreadSafeDict' and '{}'".format(type(other).__name__))

        new_dict = ThreadSafeDict()
        with self._lock:
            new_dict._dict.update(self._dict)  # 첫 번째 dict 내용 복사
        with other._lock:
            new_dict._dict.update(other._dict)  # 두 번째 dict 내용 복사
        return new_dict

    # in 연산자 오버로딩
    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    # == 연산자 오버로딩 (비교)
    def __eq__(self, other):
        if not isinstance(other, ThreadSafeDict):
            return False
        with self._lock, other._lock:
            return self._dict == other._dict

    # str() 출력 오버로딩
    def __str__(self):
        with self._lock:
            return str(self._dict)
