from win32.win32gui import *

def GetConsoleWindow() -> int: ...

# Actually returns a list of int|tuple, but lists don't support positional types
def GetWindowRgnBox(hWnd: int, /) -> tuple[int, tuple[int, int, int, int]]: ...
