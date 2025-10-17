#!/usr/bin/env python3
"""
거래 모니터링 스크립트

실시간 거래 상태 및 성과 모니터링

사용법:
    python scripts/real/monitor_trading.py --log logs/live_trading.log
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TradingMonitor:
    """거래 모니터링"""
    
    def __init__(self, log_path: str):
        """
        Args:
            log_path: 로그 파일 경로
        """
        self.log_path = log_path
        self.last_position = 0
        
        # 통계
        self.stats = {
            'total_predictions': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'buy_executed': 0,
            'sell_executed': 0,
            'auto_exits': 0,
            'stop_losses': 0,
            'take_profits': 0,
            'errors': 0
        }
    
    def parse_log_line(self, line: str):
        """로그 라인 파싱"""
        try:
            # 타임스탬프 추출
            if ' - ' not in line:
                return
            
            parts = line.split(' - ')
            if len(parts) < 4:
                return
            
            timestamp = parts[0]
            level = parts[2]
            message = parts[3].strip()
            
            # 통계 업데이트
            if 'BUY' in message and '🔵' in message:
                self.stats['buy_executed'] += 1
                print(f"[{timestamp}] {message}")
            
            elif 'SELL' in message and '🔴' in message:
                self.stats['sell_executed'] += 1
                print(f"[{timestamp}] {message}")
            
            elif 'Auto exit' in message:
                self.stats['auto_exits'] += 1
                if 'Stop loss' in message:
                    self.stats['stop_losses'] += 1
                elif 'Take profit' in message:
                    self.stats['take_profits'] += 1
            
            elif level == 'ERROR':
                self.stats['errors'] += 1
                print(f"[{timestamp}] ❌ {message}")
            
            elif '📊 Trading Statistics' in message:
                # 통계 섹션 시작
                print(f"\n{message}")
        
        except Exception as e:
            pass
    
    def tail_log(self):
        """로그 파일 실시간 모니터링"""
        print("=" * 80)
        print("📊 Trading Monitor")
        print("=" * 80)
        print(f"Monitoring: {self.log_path}")
        print("Press Ctrl+C to stop")
        print("=" * 80)
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                # 파일 끝으로 이동
                f.seek(0, 2)
                self.last_position = f.tell()
                
                while True:
                    # 현재 위치
                    current_position = f.tell()
                    
                    # 새 라인 읽기
                    line = f.readline()
                    
                    if line:
                        self.parse_log_line(line)
                    else:
                        # 파일 끝에 도달
                        time.sleep(0.5)
                        
                        # 파일이 truncate되었는지 확인
                        f.seek(0, 2)
                        if f.tell() < current_position:
                            f.seek(0)
                    
                    # 통계 출력 (10초마다)
                    if int(time.time()) % 10 == 0:
                        self.print_stats()
        
        except KeyboardInterrupt:
            print("\n⚠️ Monitoring stopped")
        except FileNotFoundError:
            print(f"❌ Log file not found: {self.log_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def print_stats(self):
        """통계 출력"""
        print("\n" + "=" * 80)
        print(f"📈 Monitor Stats (as of {datetime.now().strftime('%H:%M:%S')})")
        print("=" * 80)
        print(f"Buy Executed: {self.stats['buy_executed']}")
        print(f"Sell Executed: {self.stats['sell_executed']}")
        print(f"Auto Exits: {self.stats['auto_exits']}")
        print(f"  - Stop Losses: {self.stats['stop_losses']}")
        print(f"  - Take Profits: {self.stats['take_profits']}")
        print(f"Errors: {self.stats['errors']}")
        print("=" * 80)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Trading Monitor")
    parser.add_argument(
        '--log',
        type=str,
        default='logs/live_trading.log',
        help='Log file path to monitor'
    )
    
    args = parser.parse_args()
    
    # 모니터 실행
    monitor = TradingMonitor(args.log)
    monitor.tail_log()


if __name__ == '__main__':
    main()
