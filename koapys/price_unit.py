def _get_tick_unit(price: int):
    if price < 2000:
        return 1
    if price < 5000:
        return 5
    if price < 20000:
        return 10
    if price < 50000:
        return 50
    if price < 200000:
        return 100
    if price < 500000:
        return 500
    return 1000


def get_price_by_tick(price: int, tick_step: int):
    """
    price 를 기준으로 tick_step 만큼 위 아래의 가격을 계산한다.
    ex) price 가 1000 원이고 tick_step 이 2 이면 1002 를 리턴
        price 가 1000 원이고 tick_step 이 -2 이면 998 를 리턴

    :param price: 기준 가격
    :param tick_step: 기준 가격으로 부터 이동할 틱 개수. 양수면 위로, 음수면 아래로
    :return: 새로운 가격
    """
    order_price = price
    for _ in range(abs(tick_step)):
        unit = _get_tick_unit(price)
        if tick_step > 0:
            order_price += unit
        else:
            order_price -= unit
    return order_price


def get_price_by_percentage(price: int, percentage: float):
    """
    price 를 기준으로 percentage 만큼 위 아래의 가격을 계산한다.
    ex) price 가 1000 원이고 percentage 가 1% 이면 1010 원, -1% 이면 990 원을 리턴

    :param price: 기준 가격
    :param percentage: 기준 가격으로부터 이동할 퍼센트. 양수면 위로, 음수면 아래로
    :return: 새로운 가격
    """
    # 퍼센트를 기준으로 목표 가격 계산
    target_price = price * (1 + percentage / 100)

    # 목표 가격과 현재 가격의 차이를 이용해 필요한 틱 수 계산
    tick_step = round((target_price - price) / _get_tick_unit(price))

    # get_price_by_tick 을 사용해 최종 가격 계산
    return get_price_by_tick(price, tick_step)
