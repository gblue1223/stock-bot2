from typing import Any, Tuple, Union, Callable

VoidListener = Callable[[], None]
IntListener = Callable[[int], Any]
StringListener = Callable[[str], Any]
DataListener = Callable[[dict], None]
