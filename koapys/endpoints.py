# NOTE: Fill these with actual paths after parsing the PDF documentation.
# Keep names stable to minimize refactors across the codebase.

DEFAULT_BASE_URL = "https://api.kiwoom.com"  # Placeholder

# Auth / session
AUTH_TOKEN = "/auth/token"  # POST, returns access token
PING = "/ping"  # GET, healthcheck

# Accounts
ACCOUNTS_LIST = "/v1/accounts"  # GET
DEPOSIT_DETAIL = "/v1/accounts/{account_no}/deposit-detail"  # GET
EQUITY_BALANCE = "/v1/accounts/{account_no}/equity-balance"  # GET
EQUITY_BALANCE2 = "/v1/accounts/{account_no}/equity-balance2"  # GET (if exists)

# Market data
STOCK_BASIC_INFO = "/v1/stocks/{code}/basic-info"  # GET
CURRENT_PRICE = "/v1/stocks/{code}/price"  # GET (may be part of basic info)
UPPER_LIMITS = "/v1/stocks/upper-limits"  # GET with query params

# Orders
SEND_ORDER = "/v1/orders"  # POST
SEND_ORDER_MODIFY = "/v1/orders/{order_no}"  # PATCH
SEND_ORDER_CANCEL = "/v1/orders/{order_no}"  # DELETE
