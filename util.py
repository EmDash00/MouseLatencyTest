import ctypes
import platform
import time

import numpy as np

match platform.system():
    case "Linux" | "Darwin":
        libc = ctypes.CDLL(
            "libc.so.6" if platform.system() == "Linux" else "libc.dylib"
        )

        class timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

        libc.nanosleep.argtypes = [ctypes.POINTER(timespec), ctypes.POINTER(timespec)]
        libc.nanosleep.restype = ctypes.c_int

        def nanosleep(duration_sec):
            """High-precision sleep using nanosleep."""
            req = timespec(
                int(duration_sec), int((duration_sec - int(duration_sec)) * 1e9)
            )
            libc.nanosleep(req, None)

    case "Windows":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # Windows API types
        c_longlong_ptr = ctypes.POINTER(ctypes.c_longlong)
        DWORD = ctypes.c_uint32

        # Timer functions
        CreateWaitableTimerW = kernel32.CreateWaitableTimerW
        CreateWaitableTimerW.argtypes = (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_wchar_p,
        )
        CreateWaitableTimerW.restype = ctypes.c_void_p

        SetWaitableTimer = kernel32.SetWaitableTimer
        SetWaitableTimer.argtypes = (
            ctypes.c_void_p,
            c_longlong_ptr,
            DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        SetWaitableTimer.restype = ctypes.c_int

        WaitForSingleObject = kernel32.WaitForSingleObject
        WaitForSingleObject.argtypes = (ctypes.c_void_p, DWORD)
        WaitForSingleObject.restype = DWORD

        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = (ctypes.c_void_p,)
        CloseHandle.restype = ctypes.c_int

        def nanosleep(duration_sec):
            """High-precision sleep using Windows WaitableTimer."""
            timer = CreateWaitableTimerW(None, True, None)
            if not timer:
                raise ctypes.WinError(ctypes.get_last_error())

            # Convert seconds to 100-nanosecond intervals (Windows time format)
            interval = ctypes.c_longlong(int(-duration_sec * 10_000_000))
            success = SetWaitableTimer(
                timer, ctypes.byref(interval), 0, None, None, True
            )
            if not success:
                CloseHandle(timer)
                raise ctypes.WinError(ctypes.get_last_error())

            WaitForSingleObject(timer, 0xFFFFFFFF)  # INFINITE wait
            CloseHandle(timer)


def precision_sleep(t: float, check_time: float = 0.001, check_interval: float = 0):
    """
    Waits for `t` seconds. Sleeps until `t - check_time`.
    After the sleep period, checks to see if period `t` has elaspsed every
    `check_interval` seconds.


    Parameters
    ----------
    t : float
        The number of seconds to wait.

    check_time : float, default = 0.001
        The amount of time to actively check if period `t` has elapsed.

    check_interval : float, default=0
        The number of seconds to sleep between checks to see if t seconds
        have passed. If check interval is negative, busy waits.


    Notes
    -----
    `time.sleep(0)` forces an immediate context switch. Context will switch
    back as soon as possible.
    """

    t0 = time.perf_counter()

    sleep_time = max(t - check_time, 0)
    if sleep_time > 0:
        time.sleep(sleep_time)

    if check_interval > 0:
        while time.perf_counter() - t0 < t:
            time.sleep(check_interval)

        return

    while time.perf_counter() - t0 < t:
        pass


def spin(t: float):
    """
    Busy waits for t seconds.


    Parameters
    ----------
    t : float
        The number of seconds to busy wait.
    """

    t0 = time.perf_counter()

    while time.perf_counter() - t0 < t:
        pass


def find_closest(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size == 0:
        return np.full(y.shape, np.nan)

    sorted_x = np.sort(x)

    indices = np.searchsorted(sorted_x, y)

    left_idx = np.clip(indices - 1, 0, len(sorted_x) - 1)
    right_idx = np.clip(indices, 0, len(sorted_x) - 1)

    left_vals = sorted_x[left_idx]
    right_vals = sorted_x[right_idx]

    left_diff = np.abs(left_vals - y)
    right_diff = np.abs(right_vals - y)

    closest_indices_in_sorted = np.where(left_diff < right_diff, left_idx, right_idx)
    original_indices = np.argsort(x)

    closest_indices = original_indices[closest_indices_in_sorted]
    return closest_indices
