"""
Test script to verify koapys client migration to kiwoom_rest_api
"""
import logging
from koapys.client import KoapyRestSimple
from koapys.types import OrderType, OrderBookType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simulation_mode():
    """Test client in simulation mode"""
    logger.info("Testing simulation mode...")
    
    client = KoapyRestSimple(simulation=True)
    
    # Test connection
    client.ensure_connected()
    assert client.is_connected, "Should be connected in simulation mode"
    assert client.is_simulation_mode, "Should be in simulation mode"
    
    # Test account methods
    accounts = client.get_account_list()
    assert accounts == ["0000000000"], "Should return dummy account"
    
    deposit = client.get_deposit("0000000000")
    assert "예수금" in deposit, "Should have deposit info"
    
    balance = client.get_equity_balance("0000000000")
    assert "total" in balance and "stocks" in balance, "Should have balance structure"
    
    # Test market data methods
    stock_info = client.get_stock_basic_info("005930")
    assert "종목명" in stock_info, "Should have stock info"
    
    price = client.get_current_price("005930")
    assert price > 0, "Should return positive price"
    
    upper_limit = client.get_upper_limit_price("005930")
    assert upper_limit > 0, "Should return positive upper limit"
    
    # Test order methods
    order_result = client.send_order(
        rqname="test",
        account_no="0000000000",
        order_type=OrderType.BUY,
        code="005930",
        quantity=1,
        price=50000,
        hoga=OrderBookType.LIMIT
    )
    assert "order_no" in order_result, "Should have order number"
    
    modify_result = client.send_order_modify("SIM-ORDER-0001", price=51000)
    assert "order_no" in modify_result, "Should have order number"
    
    cancel_result = client.send_order_cancel("SIM-ORDER-0001")
    assert "order_no" in cancel_result, "Should have order number"
    
    client.close()
    logger.info("✓ Simulation mode tests passed")

def test_real_mode_structure():
    """Test client structure in real mode (without actual API calls)"""
    logger.info("Testing real mode structure...")
    
    # Should not fail during initialization even without credentials
    client = KoapyRestSimple(
        simulation=False
    )
    
    # Check that components are initialized
    assert client._token_manager is not None, "Token manager should be initialized"
    assert client._stock_info is not None, "Stock info should be initialized"
    assert client._account is not None, "Account should be initialized"
    assert client._order is not None, "Order should be initialized"
    
    client.close()
    logger.info("✓ Real mode structure tests passed")

if __name__ == "__main__":
    try:
        test_simulation_mode()
        test_real_mode_structure()
        logger.info("\n✅ All tests passed!")
    except Exception as e:
        logger.error(f"\n❌ Tests failed: {e}")
        raise
