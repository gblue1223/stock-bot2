from datetime import datetime  # 현재 시간을 얻기 위한 라이브러리 추가
import numpy as np
import matplotlib.pyplot as plt  # 그래프 생성을 위한 라이브러리 추가


def plot_and_save(prices, balances, quantities, actions):
    # 현재 시간을 기반으로 파일명 생성
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"train_logs/lr-{now}.png"

    plt.figure(figsize=(14, 8))

    # 매도, 매수, 수익률 그래프 그리기
    plt.subplot(3, 1, 1)
    plt.plot(prices, label="Price")
    buy_signals = [i for i in range(len(actions)) if actions[i] == 1]
    sell_signals = [i for i in range(len(actions)) if actions[i] == 2]
    plt.scatter(buy_signals, np.array(prices)[buy_signals], marker='.', color='red', label="Buy", s=100)
    plt.scatter(sell_signals, np.array(prices)[sell_signals], marker='.', color='blue', label="Sell", s=100)
    plt.title("Stock Price and Actions")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(balances, label="Balance")
    plt.title("Account Balance")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))  # 정수 값으로 표시하고 천 단위 구분자 추가

    plt.subplot(3, 1, 3)
    plt.plot(quantities, label="Quantity")
    plt.title("Stock Quantity")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)  # 그래프를 이미지 파일로 저장
    plt.show()
