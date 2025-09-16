
def is_empty_str(string):
    return not string or string.strip() == ""


def drop_keys(src_data: dict, exclude_keys: list) -> dict:
    data = src_data.copy()
    for key in exclude_keys:
        if key in data:
            del data[key]
    return data
