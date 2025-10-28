# PyEventEmitter Migration Guide

## Overview

The WebSocket client implementation has been migrated from a callback-based pattern to use `pyee` (PyEventEmitter) from pip. This provides a more flexible and standardized event handling system.

## What Changed

### 1. WebSocketClient (websocket.py)

**Before:**
```python
client = WebSocketClient(access_token=token)

# Setting callbacks
client.on_connect = my_connect_handler
client.on_disconnect = my_disconnect_handler
client.on_login = my_login_handler
client.on_data = my_data_handler
client.on_error = my_error_handler
```

**After:**
```python
client = WebSocketClient(access_token=token)

# Registering event listeners
client.on('connect', my_connect_handler)
client.on('disconnect', my_disconnect_handler)
client.on('login', my_login_handler)
client.on('data', my_data_handler)
client.on('error', my_error_handler)
```

### 2. Event Names

The following events are available on `WebSocketClient`:

- **`connect`**: Fired when WebSocket connection is established
- **`disconnect`**: Fired when WebSocket connection is closed
- **`login`**: Fired when login is successful
- **`data`**: Fired when real-time data is received (passes `RealTimeData` object)
- **`error`**: Fired when an error occurs (passes `Exception` object)

### 3. Multiple Listeners

With PyEventEmitter, you can now register multiple listeners for the same event:

```python
client.on('data', handler1)
client.on('data', handler2)  # Both handlers will be called
```

### 4. One-Time Listeners

You can register listeners that only fire once:

```python
client.once('login', async_handler)  # Will only be called once
```

### 5. Removing Listeners

**Before:** Not possible (callbacks were overwritten)

**After:**
```python
# Remove a specific listener
client.remove_listener('data', my_data_handler)

# Or use the shorter alias
client.off('data', my_data_handler)

# Remove all listeners for an event
client.remove_all_listeners('data')
```

## Migration for Helper Classes

### ConditionSearchClient

**Before:**
```python
condition_client = ConditionSearchClient(websocket_client)
# Cleanup required manual callback restoration
condition_client.cleanup()
```

**After:**
```python
condition_client = ConditionSearchClient(websocket_client)
# Cleanup now properly removes event listeners
condition_client.cleanup()
```

The `ConditionSearchClient` now uses `client.on()` instead of replacing callbacks, allowing multiple helper classes to coexist without conflicts.

### RealtimeStockClient

Same pattern as `ConditionSearchClient`:

```python
stock_client = RealtimeStockClient(websocket_client)
# Multiple clients can listen to the same WebSocketClient
stock_client.cleanup()  # Properly removes listeners
```

## Benefits

1. **Multiple Listeners**: Register multiple handlers for the same event
2. **No Callback Conflicts**: Helper classes no longer overwrite each other's callbacks
3. **Standard API**: Uses the well-established PyEventEmitter pattern
4. **Better Cleanup**: Proper listener removal with `remove_listener()` and `remove_all_listeners()`
5. **Type Safety**: Event names are strings, reducing attribute typos

## Installation

The migration requires the `pyee` package:

```bash
pip install pyee>=11.0
```

This has been added to `requirements.txt`.

## Example: Multiple Handlers

```python
from kiwoom_rest_api.websocket import WebSocketClient
from kiwoom_rest_api.koreanstock.condition_search import ConditionSearchClient
from kiwoom_rest_api.koreanstock.realtime_stock import RealtimeStockClient

# Create WebSocket client
ws_client = WebSocketClient(access_token=token)

# Add custom data handler
async def my_custom_handler(realtime_data):
    print(f"Custom handler: {realtime_data.trnm}")

ws_client.on('data', my_custom_handler)

# Add condition search handler (won't conflict!)
condition_client = ConditionSearchClient(ws_client)
condition_client.on_condition_list = lambda conditions: print(f"Conditions: {len(conditions)}")

# Add stock realtime handler (also won't conflict!)
stock_client = RealtimeStockClient(ws_client)
stock_client.on_trade_data = lambda code, data: print(f"Trade: {code}")

# All handlers will receive events
await ws_client.start()
```

## Backward Compatibility Notes

⚠️ **Breaking Changes:**
- Direct attribute assignment (`client.on_data = handler`) will no longer work
- Must use `client.on('data', handler)` instead
- Helper class `cleanup()` methods now use `remove_listener()` instead of restoring callbacks

## Testing

After migration, test your code to ensure:
1. Event listeners are properly registered
2. Multiple listeners work as expected
3. Cleanup properly removes listeners
4. No callback conflicts between helper classes
