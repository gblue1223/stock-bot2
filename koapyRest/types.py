from enum import Enum

class OrderType(Enum):
    NONE = 0
    BUY = 1
    SELL = 2
    BUY_CANCEL = 3
    SELL_CANCEL = 4
    BUY_CORRECT = 5
    SELL_CORRECT = 6

class OrderBookType(Enum):
    LIMIT = "00"
    MARKET = "03"
    CONDITIONAL_LIMIT = "05"
    BEST_LIMIT = "06"
    PRIORITY_LIMIT = "07"
    LIMIT_IOC = "10"
    MARKET_IOC = "13"
    BEST_IOC = "16"
    LIMIT_FOK = "20"
    MARKET_FOK = "23"
    BEST_FOK = "26"
    PRE_MARKET_CLOSE = "61"
    AFTER_HOURS_SINGLE = "62"
    POST_MARKET_CLOSE = "81"
