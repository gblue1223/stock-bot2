import win32process
import win32con
import ctypes
from ctypes import windll


def get_process_handle_with_permissions(pid, desired_access=win32con.PROCESS_ALL_ACCESS):
    """
    특정 프로세스에 대한 핸들을 권한과 함께 얻습니다.

    :param pid: 프로세스 ID
    :param desired_access: 요청할 접근 권한 (기본값: 모든 권한)
    :return: 프로세스 핸들
    """
    try:
        # OpenProcess API 직접 호출
        handle = windll.kernel32.OpenProcess(
            desired_access,  # 원하는 접근 권한
            False,  # 자식 프로세스가 핸들을 상속할지 여부
            pid     # 대상 프로세스 ID
        )

        if handle == 0:
            error_code = ctypes.get_last_error()
            print(f"OpenProcess 실패. 에러 코드: {error_code}")
            return None

        return handle

    except Exception as e:
        print(f"프로세스 핸들 얻기 오류: {e}")
        return None


def close_process_handle(handle):
    """
    프로세스 핸들을 안전하게 닫습니다.

    :param handle: 닫을 프로세스 핸들
    """
    if handle:
        windll.kernel32.CloseHandle(handle)


def get_process_handle_by_hwnd(hwnd):
    """
    :param hwnd: 대상 윈도우 핸들
    """
    try:
        # 윈도우 핸들로부터 프로세스 ID 얻기
        _, pid = win32process.GetWindowThreadProcessId(hwnd)

        # 프로세스 핸들 얻기 (다양한 권한 조합 가능)
        return get_process_handle_with_permissions(
            pid,
            win32con.PROCESS_VM_OPERATION |
            win32con.PROCESS_VM_READ |
            win32con.PROCESS_VM_WRITE
        )

    except Exception as e:
        print(f"프로세스 핸들 얻기 실패: {e}")


# 주요 접근 권한 플래그 (일부)
ACCESS_FLAGS = {
    'PROCESS_TERMINATE': 0x0001,           # 프로세스 종료 권한
    'PROCESS_CREATE_THREAD': 0x0002,       # 스레드 생성 권한
    'PROCESS_VM_OPERATION': 0x0008,        # 가상 메모리 연산 권한
    'PROCESS_VM_READ': 0x0010,             # 메모리 읽기 권한
    'PROCESS_VM_WRITE': 0x0020,            # 메모리 쓰기 권한
    'PROCESS_DUP_HANDLE': 0x0040,          # 핸들 복제 권한
    'PROCESS_CREATE_PROCESS': 0x0080,      # 새 프로세스 생성 권한
    'PROCESS_SET_QUOTA': 0x0100,           # 프로세스 할당량 설정 권한
    'PROCESS_SET_INFORMATION': 0x0200,     # 프로세스 정보 설정 권한
    'PROCESS_QUERY_INFORMATION': 0x0400,   # 프로세스 정보 조회 권한
    'PROCESS_SUSPEND_RESUME': 0x0800,      # 프로세스 일시 정지/재개 권한
    'PROCESS_ALL_ACCESS': 0x1F0FFF         # 모든 가능한 권한
}
