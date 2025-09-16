import win32con
import win32gui
import win32ui
import mss
from typing import Optional
from PIL import Image


def capture_window_with_gdi(hwnd) -> Optional['Image']:
    try:
        # 윈도우 크기 가져오기
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        # 윈도우 DC 가져오기
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        mem_dc = mfc_dc.CreateCompatibleDC()

        # 비트맵 생성
        screenshot = win32ui.CreateBitmap()
        screenshot.CreateCompatibleBitmap(mfc_dc, width, height)
        mem_dc.SelectObject(screenshot)

        # 윈도우 내용을 메모리 DC로 복사
        mem_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

        # WM_PRINT 메시지를 사용하여 창의 내용을 복사
        # print_flags = win32con.PRF_CLIENT | win32con.PRF_NONCLIENT | win32con.PRF_CHILDREN
        # win32gui.SendMessage(hwnd, win32con.WM_PRINT, mem_dc.GetSafeHdc(), print_flags)

        # 비트맵을 PIL 이미지로 변환
        bmpinfo = screenshot.GetInfo()
        bmpstr = screenshot.GetBitmapBits(True)
        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr,
            'raw',
            'BGRX',
            0,
            1
        )

        # 메모리 정리
        mem_dc.DeleteDC()
        win32gui.DeleteObject(screenshot.GetHandle())
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        # 이미지 파일 저장
        filename = f"{win32gui.GetWindowText(hwnd)}_screenshot.png"
        img.save(filename)
        print(f"스크린샷 저장: {filename}")

        return img

    except Exception as e:
        print(f"GDI 캡처 오류 발생: {e}")
        return None


def capture_window_with_mss(hwnd):
    try:
        # 창의 좌표 가져오기
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)

        # mss로 화면 캡처
        with mss.mss() as sct:
            monitor = {"top": top, "left": left, "width": right - left, "height": bottom - top}
            screenshot = sct.grab(monitor)

        # PIL 이미지를 생성
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        # 이미지 저장
        filename = f"[{win32gui.GetWindowText(hwnd)}]_screenshot.png"
        img.save(filename)
        print(f"스크린샷 저장: {filename}")

        return img
    except Exception as e:
        print(f"MSS 캡처 오류 발생: {e}")
        return None
