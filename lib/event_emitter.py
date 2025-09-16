from lib.thread_safe_dict import ThreadSafeDict


class EventEmitter:
    """
    # 사용 예시
    emitter = EventEmitter()

    def on_hello(name):
        print(f"Hello, {name}!")

    # 이벤트 리스너 등록
    emitter.on('greet', on_hello)

    # 이벤트 트리거
    emitter.emit('greet', 'Python')  # 출력: Hello, Python!

    # 이벤트 리스너 제거
    emitter.off('greet', on_hello)

    # 더 이상 리스너가 없으므로 아무것도 출력되지 않음
    emitter.emit('greet', 'Python')

    # 한번만 호출되는 이벤트 리스너 등록
    emitter.once('greet', on_hello)

    # 첫 번째 호출: 리스너가 호출됨
    emitter.emit('greet', 'Python')  # 출력: Hello, Python!

    # 두 번째 호출: 리스너가 제거되었으므로 아무것도 출력되지 않음
    emitter.emit('greet', 'Python')
    """
    def __init__(self):
        self._events = ThreadSafeDict()

    def on(self, event, listener):
        if event not in self._events:
            self._events[event] = []
        self._events[event].append(listener)

    def off(self, event, listener):
        if event in self._events:
            self._events[event].remove(listener)
            if not self._events[event]:
                del self._events[event]

    def emit(self, event, *args, **kwargs):
        if event in self._events:
            # Copy the listener list to avoid modification during iteration
            listeners = list(self._events[event])
            for listener in listeners:
                listener(*args, **kwargs)

    def once(self, event, listener):
        def wrapper(*args, **kwargs):
            # Call the original listener
            listener(*args, **kwargs)
            # Remove the listener after the first call
            self.off(event, wrapper)

        # Register the wrapper function instead of the original listener
        self.on(event, wrapper)

    def get_listeners(self, event):
        return self._events[event]
