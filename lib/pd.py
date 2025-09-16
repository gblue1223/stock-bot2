import pandas as pd


def append(df, row):
    row = row.to_frame().T.reset_index(drop=True)
    return pd.concat([df, row], ignore_index=True)  # Series를 DataFrame으로 변환 후 추가
