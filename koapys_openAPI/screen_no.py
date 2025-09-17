class _NumberRange:
    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end
        self._current = start

    def next(self) -> str:
        self._current += 1
        if self._current > self._end:
            self._current = self._start
        return f"{self._current:04d}"


class ScreenNo:
    # 범위 정의
    RANGES = {
        'tr_data': (1, 199),
        'real_data': (200, 499),
        'condition': (500, 599),
        'order': (600, 999)
    }

    # 각 범위에 대한 NumberRange 인스턴스 생성
    _number_ranges = {
        name: _NumberRange(start, end)
        for name, (start, end) in RANGES.items()
    }

    @classmethod
    def tr_data(cls) -> str:
        return cls._number_ranges['tr_data'].next()

    @classmethod
    def real_data(cls) -> str:
        return cls._number_ranges['real_data'].next()

    @classmethod
    def condition(cls) -> str:
        return cls._number_ranges['condition'].next()

    @classmethod
    def order(cls) -> str:
        return cls._number_ranges['order'].next()

    @staticmethod
    def to_string(screen_no: str):
        # 숫자 문자열을 정수로 변환
        try:
            num = int(screen_no)
            # 각 범위 체크
            for name, (start, end) in ScreenNo.RANGES.items():
                if start <= num <= end:
                    return name
            return "unknown"
        except ValueError:
            return screen_no if screen_no in ScreenNo.RANGES else "unknown"
