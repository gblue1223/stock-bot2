"""
Real API connection test for koapys client
Requires valid KIWOOM_API_KEY and KIWOOM_API_SECRET environment variables
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from koapys.client import KoapyRestSimple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_connection():
    """Test basic connection to Kiwoom REST API"""
    logger.info("=" * 60)
    logger.info("Testing connection to Kiwoom REST API...")
    logger.info("=" * 60)
    
    # Check environment variables
    api_key = os.getenv("KIWOOM_API_KEY")
    api_secret = os.getenv("KIWOOM_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ KIWOOM_API_KEY or KIWOOM_API_SECRET not set!")
        logger.info("\nPlease set environment variables:")
        logger.info("  export KIWOOM_API_KEY='your_key'")
        logger.info("  export KIWOOM_API_SECRET='your_secret'")
        return False
    
    logger.info(f"✓ API Key: {api_key[:10]}...")
    logger.info(f"✓ API Secret: {api_secret[:10]}...")
    
    try:
        # Initialize client
        client = KoapyRestSimple(
            simulation=False
        )
        logger.info("✓ Client initialized")
        
        # Test connection
        client.ensure_connected()
        logger.info("✓ Connection successful!")
        logger.info(f"  - Connected: {client.is_connected}")
        logger.info(f"  - Simulation Mode: {client.is_simulation_mode}")
        
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_account_info():
    """Test account information retrieval"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing account information retrieval...")
    logger.info("=" * 60)
    
    api_key = os.getenv("KIWOOM_API_KEY")
    api_secret = os.getenv("KIWOOM_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ Environment variables not set!")
        return False
    
    try:
        client = KoapyRestSimple(
            simulation=False
        )
        client.ensure_connected()
        
        # Test 1: Get account list
        logger.info("\n[Test 1] Getting account list...")
        try:
            accounts = client.get_account_list()
            logger.info(f"✓ Accounts: {accounts}")
        except Exception as e:
            logger.warning(f"⚠ get_account_list failed (may not be implemented): {e}")
            accounts = []
        
        # Use first account or default
        account_no = accounts[0] if accounts else "DEFAULT"
        logger.info(f"\nUsing account: {account_no}")
        
        # Test 2: Get deposit information
        logger.info("\n[Test 2] Getting deposit information...")
        try:
            deposit = client.get_deposit(account_no)
            logger.info("✓ Deposit information retrieved:")
            for key, value in deposit.items():
                logger.info(f"  - {key}: {value}")
        except Exception as e:
            logger.error(f"❌ get_deposit failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Get equity balance
        logger.info("\n[Test 3] Getting equity balance...")
        try:
            balance = client.get_equity_balance(account_no)
            logger.info("✓ Equity balance retrieved:")
            
            if "total" in balance:
                logger.info("  Total:")
                for key, value in balance["total"].items():
                    logger.info(f"    - {key}: {value}")
            
            if "stocks" in balance:
                logger.info(f"  Stocks: {len(balance['stocks'])} items")
                for i, stock in enumerate(balance["stocks"][:3]):  # Show first 3
                    logger.info(f"    Stock {i+1}: {stock}")
        except Exception as e:
            logger.error(f"❌ get_equity_balance failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 4: Get equity balance2
        logger.info("\n[Test 4] Getting detailed equity balance...")
        try:
            balance2 = client.get_equity_balance2(account_no)
            logger.info("✓ Detailed equity balance retrieved:")
            if isinstance(balance2, dict):
                for key, value in list(balance2.items())[:5]:  # Show first 5 items
                    logger.info(f"  - {key}: {value}")
        except Exception as e:
            logger.error(f"❌ get_equity_balance2 failed: {e}")
            import traceback
            traceback.print_exc()
        
        client.close()
        logger.info("\n" + "=" * 60)
        logger.info("Account information tests completed!")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stock_info():
    """Test stock information retrieval"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing stock information retrieval...")
    logger.info("=" * 60)
    
    api_key = os.getenv("KIWOOM_API_KEY")
    api_secret = os.getenv("KIWOOM_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ Environment variables not set!")
        return False
    
    # Test with Samsung Electronics (005930)
    stock_code = "005930"
    logger.info(f"\nTesting with stock code: {stock_code} (삼성전자)")
    
    try:
        client = KoapyRestSimple(
            simulation=False
        )
        client.ensure_connected()
        
        # Test 1: Get stock basic info
        logger.info("\n[Test 1] Getting stock basic information...")
        try:
            stock_info = client.get_stock_basic_info(stock_code)
            logger.info("✓ Stock information retrieved:")
            for key, value in stock_info.items():
                logger.info(f"  - {key}: {value}")
        except Exception as e:
            logger.error(f"❌ get_stock_basic_info failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: Get current price
        logger.info("\n[Test 2] Getting current price...")
        try:
            price = client.get_current_price(stock_code)
            logger.info(f"✓ Current price: {price:,}원")
        except Exception as e:
            logger.error(f"❌ get_current_price failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Get upper limit price
        logger.info("\n[Test 3] Getting upper limit price...")
        try:
            upper_limit = client.get_upper_limit_price(stock_code)
            logger.info(f"✓ Upper limit price: {upper_limit:,}원")
        except Exception as e:
            logger.error(f"❌ get_upper_limit_price failed: {e}")
            import traceback
            traceback.print_exc()
        
        client.close()
        logger.info("\n" + "=" * 60)
        logger.info("Stock information tests completed!")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("\n" + "=" * 60)
    logger.info("Koapys Real API Connection Test Suite")
    logger.info("=" * 60)
    
    results = {}
    
    # Run tests
    results["connection"] = test_connection()
    results["account_info"] = test_account_info()
    results["stock_info"] = test_stock_info()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✅ All tests PASSED!")
    else:
        logger.info("❌ Some tests FAILED!")
    logger.info("=" * 60)
