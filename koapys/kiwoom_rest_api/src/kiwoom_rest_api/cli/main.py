import os
import typer
import json
from rich import print as rprint # print_json 대신 사용할 수 있음
from rich.pretty import pprint # 객체 예쁘게 출력

# 필요한 클래스 임포트
from kiwoom_rest_api.koreanstock.stockinfo import StockInfo
from kiwoom_rest_api.auth.token import TokenManager
from kiwoom_rest_api.core.base import APIError

# Typer 앱 인스턴스 생성
# no_args_is_help=True: 인자 없이 실행 시 도움말 표시
app = typer.Typer(no_args_is_help=True, add_completion=False)

# --- 앱의 기본 동작 (선택 사항, 도움말 개선 등) ---
@app.callback()
def main_callback(ctx: typer.Context):
    """
    키움증권 Open API CLI 도구
    """
    # 서브커맨드가 없으면 도움말 표시 (Typer가 기본 처리)
    # 여기에 앱 전역 설정을 추가할 수도 있음
    pass

# --- ka10001 서브커맨드 정의 ---
@app.command()
def ka10001(
    stock_code: str = typer.Argument(..., help="조회할 주식 종목 코드 (예: 005930)"),
    api_key: str = typer.Option(
        None, "--api-key", "-k",
        help="키움증권 API Key (환경 변수 KIWOOM_API_KEY)",
        envvar="KIWOOM_API_KEY",
        show_envvar=True,
    ),
    api_secret: str = typer.Option(
        None, "--api-secret", "-s",
        help="키움증권 API Secret (환경 변수 KIWOOM_API_SECRET)",
        envvar="KIWOOM_API_SECRET",
        show_envvar=True,
    ),
    base_url: str = typer.Option(
        "https://api.kiwoom.com",
        "--base-url", "-u",
        help="API 기본 URL"
    ),
):
    """
    주식 기본 정보 요청 (KA10001) API를 호출합니다.
    """
    if not api_key:
        typer.secho("오류: API Key가 제공되지 않았습니다.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if not api_secret:
        typer.secho("오류: API Secret이 제공되지 않았습니다.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"종목 코드 {stock_code} 요청 시작 (URL: {base_url})")

    try:
        token_manager = TokenManager()
        stock_info = StockInfo(base_url=base_url, token_manager=token_manager, use_async=False)
        result = stock_info.basic_stock_information_request_ka10001(stock_code)

        typer.echo("\n--- API 응답 ---")
        # rich의 pprint 사용 (print_json 대신)
        pprint(result, expand_all=True)
        typer.echo("----------------")

    except APIError as e:
        typer.secho(f"\nAPI 오류 (HTTP {e.status_code}): {e.message}", fg=typer.colors.RED, err=True)
        if e.error_data:
            typer.echo("오류 데이터:", err=True)
            pprint(e.error_data, expand_all=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"\n예상치 못한 오류: {type(e).__name__}", fg=typer.colors.RED, err=True)
        typer.secho(f"메시지: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

# --- 다른 서브커맨드 추가 가능 ---
# @app.command()
# def another_command(...):
#     ...

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    app()
