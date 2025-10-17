# deploy

## prodution
```
    pip install kiwoom-rest-api
    uv add kiwoom-rest-api
```

### deploy to pypi [site](https://pypi.org/project/kiwoom-rest-api/)
```
    poetry shell
    poetry publish --build
```


## test
```
    pip install -i https://test.pypi.org/simple/ kiwoom-rest-api
```
```
    uv add -i https://test.pypi.org/simple/ kiwoom-rest-api
```

### deploy to test-pypi [site](https://test.pypi.org/project/kiwoom-rest-api/)
```
    poetry shell
    poetry publish --build -r test-pypi
```


## CLI Usage

```bash
    # Linux/macOS/Windows(git bash)
    export KIWOOM_API_KEY="YOUR_ACTUAL_API_KEY"
    export KIWOOM_API_SECRET="YOUR_ACTUAL_API_SECRET"

    # Windows (CMD)
    set KIWOOM_API_KEY="YOUR_ACTUAL_API_KEY"
    set KIWOOM_API_SECRET="YOUR_ACTUAL_API_SECRET"

    # Windows (PowerShell)
    $env:KIWOOM_API_KEY="YOUR_ACTUAL_API_KEY"
    $env:KIWOOM_API_SECRET="YOUR_ACTUAL_API_SECRET"
```

```bash
    # 가상 환경 활성화 (필요시)
    poetry shell

    # 도움말 보기
    python src/kiwoom_rest_api/cli/main.py --help
    python src/kiwoom_rest_api/cli/main.py ka10001 --help

    # ka10001 명령어 실행 (환경 변수 사용 시)
    python src/kiwoom_rest_api/cli/main.py ka10001 005930 # 삼성전자 예시

    # ka10001 명령어 실행 (옵션 사용 시)
    python src/kiwoom_rest_api/cli/main.py --api-key "YOUR_KEY" --api-secret "YOUR_SECRET" ka10001 005930

    # 다른 base URL 사용 시
    python src/kiwoom_rest_api/cli/main.py --base-url "https://mockapi.kiwoom.com" ka10001 005930
```



# docs
[pypi-keys]()
