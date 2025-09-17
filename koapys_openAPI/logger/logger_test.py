from koapys.logger.logger import parse_log_messages


# 사용 예시:
def simulate_messages(messages: list):
   for message in messages:
       if message['type'] == '주식체결':
           # 주식체결 데이터 처리
           data = message['data']
           print(f"체결: {data['주식코드']} - {data['현재가']}")
       elif message['type'] == '주식호가잔량':
           # 주식호가잔량 데이터 처리
           data = message['data']
           print(f"호가: {data['주식코드']} - {data['매도호가1']}")

# 테스트
log_path = './koapys.txt'
messages = parse_log_messages(log_path)
simulate_messages(messages)
