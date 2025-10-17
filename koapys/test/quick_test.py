"""
Quick test script for Koapys client
Tests basic functionality with minimal output
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from koapys.client import KoapyRestSimple

def main():
    # Check environment variables
    api_key = os.getenv("KIWOOM_API_KEY")
    api_secret = os.getenv("KIWOOM_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ Error: KIWOOM_API_KEY and KIWOOM_API_SECRET must be set")
        print("\nSet environment variables:")
        print("  export KIWOOM_API_KEY='your_key'")
        print("  export KIWOOM_API_SECRET='your_secret'")
        sys.exit(1)
    
    print("Testing Koapys Client...")
    print("-" * 50)
    
    try:
        # Initialize client
        print("1. Initializing client...")
        client = KoapyRestSimple(
            simulation=False
        )
        print("   ✓ Client initialized")
        
        # Test connection
        print("2. Testing connection...")
        client.ensure_connected()
        print(f"   ✓ Connected: {client.is_connected}")
        
        # Test stock info (Samsung Electronics)
        print("3. Testing stock info (005930 - 삼성전자)...")
        stock_info = client.get_stock_basic_info("005930")
        print(f"   ✓ Stock info retrieved")
        
        if stock_info:
            # Try to get stock name and price
            name = stock_info.get("종목명") or stock_info.get("stk_nm", "N/A")
            price = stock_info.get("현재가") or stock_info.get("cur_prc", "N/A")
            print(f"   - Name: {name}")
            print(f"   - Price: {price}")
        
        # Clean up
        client.close()
        
        print("-" * 50)
        print("✅ Quick test completed successfully!")
        
    except Exception as e:
        print("-" * 50)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
