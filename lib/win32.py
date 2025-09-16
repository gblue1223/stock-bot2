from typing import Tuple, Union, List, Any

import win32api
import win32con
import win32gui


def bring_child_window_to_top(child_hwnd):
    try:
        # 자식 창을 최상위로 이동
        win32gui.SetWindowPos(
            child_hwnd,
            win32con.HWND_TOP,
            0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
        )
        print(f"창({child_hwnd})을 최상위로 이동했습니다.")
    except Exception as e:
        print(f"bring_child_window_to_top.error: {e}")


def send_key(hwnd, vk_code):
    win32gui.SendMessage(hwnd, win32con.WM_KEYDOWN, vk_code, 0)
    win32gui.SendMessage(hwnd, win32con.WM_KEYUP, vk_code, 0)


def send_click(hwnd, x, y):
    lparam = win32api.MAKELONG(x, y)
    win32api.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
    win32api.SendMessage(hwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, lparam)


def inject_text(hwnd, text):
    pass


def find_child_window(
        parent_hwnd: int,
        partial_child_window_title: str,
        find_all: bool = False) -> Union[Union[list[Any], tuple[int, str]], Any]:
    empty = -1, ""
    windows = []
    try:
        def callback(hwnd, windows_):
            child_text = win32gui.GetWindowText(hwnd)
            if partial_child_window_title in child_text:
                windows_.append((hwnd, child_text))
                return find_all
            return True

        win32gui.EnumChildWindows(parent_hwnd, callback, windows)
    except Exception as e:
        if e.strerror != 'No error message is available':
            print(f"find_child_window.error: {e}")

    if find_all:
        return windows
    return windows[0] if windows else empty


def find_child_window_by_class(
        parent_hwnd: int,
        child_window_class: str,
        find_all: bool = False) -> Union[Union[list[Any], tuple[int, str]], Any]:
    empty = -1, ""
    windows = []
    try:
        def callback(hwnd, windows_):
            class_name = win32gui.GetClassName(hwnd)
            if class_name == child_window_class:
                text = win32gui.GetWindowText(hwnd)
                windows_.append((hwnd, text))
                return find_all
            return True

        win32gui.EnumChildWindows(parent_hwnd, callback, windows)
    except Exception as e:
        if e.strerror != 'No error message is available':
            print(f"find_child_windows_by_class.error: {e}")

    if find_all:
        return windows
    return windows[0] if windows else empty


def find_window(partial_title: str, find_all: bool = False) -> Union[Union[list[Any], tuple[int, str]], Any]:
    empty = -1, ""
    windows = []
    try:
        def callback(hwnd, windows_):
            window_text = win32gui.GetWindowText(hwnd)
            if partial_title in window_text:
                windows_.append((hwnd, window_text))
                return find_all
            return True

        win32gui.EnumWindows(callback, windows)
    except Exception as e:
        if e.strerror != 'No error message is available':
            print(f"find_window.error: {e}")

    if find_all:
        return windows
    return windows[0] if windows else empty


def find_mdi_child_window(
        parent_title: str,
        mdi_class_name: str,
        child_title_part: str,
        find_all: bool = False) -> Union[Union[list[Any], tuple[int, str]], Any]:
    def find_parent_window_(parent_title_):
        # 부모 윈도우 찾기
        hwnd = win32gui.FindWindow(None, parent_title_)
        return hwnd

    def find_mdi_client_(parent_hwnd_):
        # MDI Client 윈도우 찾기
        hwnd = win32gui.FindWindowEx(parent_hwnd_, None, mdi_class_name, None)
        return hwnd

    empty = -1, ""
    try:
        # 부모 윈도우 찾기
        parent_hwnd = find_parent_window_(parent_title)
        if parent_hwnd == 0:
            print(f"'{parent_title}' 윈도우를 찾을 수 없습니다.")
            return empty

        # MDI 클라이언트 윈도우 찾기
        mdi_hwnd = find_mdi_client_(parent_hwnd)
        if mdi_hwnd == 0:
            print(f"'{mdi_class_name}' 윈도우를 찾을 수 없습니다.")
            return empty

        # 자식 윈도우 찾기
        child_hwnd, child_text = find_child_window(mdi_hwnd, child_title_part, find_all)
        if child_hwnd == 0:
            print(f"'{child_title_part}' 창을 찾을 수 없습니다.")
            return empty

        return child_hwnd, child_text

    except Exception as e:
        print(f"find_mdi_child_window.error: {e}")
        return empty


def activate_mdi_child(child_hwnd):
    try:
        # 부모 MDI 윈도우 찾기
        mdi_hwnd = win32gui.GetParent(child_hwnd)
        parent_hwnd = win32gui.GetParent(mdi_hwnd)
        print(win32gui.GetWindowText(parent_hwnd))
        try:
            win32gui.SetForegroundWindow(parent_hwnd)
        except Exception as e:
            print(f"SetForegroundWindow {e}")

        #**보안상 접근이 안 되는것 같다.
        # # MDI 자식 창 활성화
        # import time
        # time.sleep(0.1)
        # # win32gui.SendMessage(parent_hwnd, win32con.WM_MDIACTIVATE, child_hwnd, 0)
        #
        # bring_child_window_to_top(child_hwnd)
        # print(win32gui.GetWindowText(child_hwnd))
        #
        # # 포커스 설정
        # # win32gui.SetFocus(child_hwnd)
        #
        # # 클라이언트 영역 좌표로 변환
        # screen_rect = win32gui.GetWindowRect(child_hwnd)
        # client_rect = win32gui.GetClientRect(child_hwnd)
        # print(screen_rect, client_rect)
        #
        # # 클라이언트 영역 내의 상대 좌표 사용
        # x = screen_rect[0] + 1
        # y = screen_rect[1] + 1
        # lparam = win32api.MAKELONG(x, y)
        #
        # # win32gui.SendMessage(child_hwnd, win32con.WM_SETFOCUS, 1, 0)
        # # win32gui.SendMessage(child_hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
        # # 마우스 메시지 전송
        # # win32gui.SendMessage(child_hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
        # # win32gui.SendMessage(child_hwnd, win32con.WM_LBUTTONUP, 0, lparam)
        # # win32gui.SendMessage(child_hwnd, win32con.WM_NCHITTEST, 0, lparam)
        # # 결과가 HTCLIENT인 경우 클라이언트 영역을 클릭한 것으로 처리
        # # win32gui.SendMessage(child_hwnd, win32con.WM_ACTIVATEAPP, 1, 0)
        # # win32gui.SendMessage(child_hwnd, win32con.WM_SETFOCUS, 1, 0)
        # # win32gui.SendMessage(child_hwnd, win32con.WM_NCACTIVATE, 1, 0)
        # # win32gui.SendMessage(child_hwnd, win32con.WM_SETFOCUS, 1, 0)
        #
        # # 부모에게 자식 창 클릭 알림
        # send_click(parent_hwnd, 469, 20)
        # time.sleep(0.1)
        # send_click(mdi_hwnd, 469, 20)
        # time.sleep(0.1)
        # send_click(child_hwnd, 469, 20)
        # # win32gui.SendMessage(child_hwnd, win32con.WM_PARENTNOTIFY,
        # #                      win32con.WM_LBUTTONDOWN, lparam)
        # win32gui.SendMessage(parent_hwnd, win32con.WM_PARENTNOTIFY,
        #                      win32con.WM_LBUTTONDOWN, lparam)
        #
        # # # 자식 창에 클릭 이벤트 전송
        # # win32gui.SendMessage(child_hwnd, win32con.WM_LBUTTONDOWN,
        # #                    win32con.MK_LBUTTON, lparam)
        # # win32gui.SendMessage(child_hwnd, win32con.WM_LBUTTONUP,
        # #                    0, lparam)
        #
        # #
        # # # 마우스 활성화 메시지 전송
        # # win32gui.SendMessage(child_hwnd, win32con.WM_MOUSEACTIVATE,
        # #                    parent_hwnd,
        # #                    win32api.MAKELONG(win32con.HTCAPTION, win32con.WM_LBUTTONDOWN))
        # #
        # # # 타이틀바 클릭 시뮬레이션
        # # win32gui.SendMessage(child_hwnd, win32con.WM_LBUTTONDOWN,
        # #                    win32con.MK_LBUTTON, lparam)
        # # win32gui.SendMessage(child_hwnd, win32con.WM_LBUTTONUP,
        # #                    0, lparam)
        # # win32gui.SendMessage(child_hwnd, win32con.WM_SETFOCUS, 1, 0)
        # #
        # # # MDI 자식 창 활성화
        # # win32gui.SendMessage(parent_hwnd, win32con.WM_MDIACTIVATE, child_hwnd, 0)

        print(f"activate_mdi_child.ok: {child_hwnd}")
    except Exception as e:
        print(f"activate_mdi_child.error: {e}")
