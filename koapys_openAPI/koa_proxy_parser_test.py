from koapys.koa_proxy_parser import read_enc, parse_dat


if __name__ == "__main__":
    import pprint

    #lines = read_enc("opt10001")
    #data = parse_dat("opt10001", lines)
    #pprint.pprint(data)

    # lines = read_enc("opt10098")
    # data = parse_dat("opt10098", lines)
    # pprint.pprint(data)

    lines = read_enc("opw00001")
    data = parse_dat("opw00001", lines)
    pprint.pprint(data)
