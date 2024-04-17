from enum import Enum



def from_hex(value):
    hex = tuple(int(value.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    return "rgb({},{},{})".format(hex[0], hex[1], hex[2])

class Color(Enum):
    transparent = "rgba(0,0,0,0)"

    primary = from_hex("#0d6efd")
    secondary = from_hex("#6c757d")
    success = from_hex("#198754")
    info = from_hex("#0dcaf0")
    warning = from_hex("#ffc107")
    danger = from_hex("#dc3545")
    gray = from_hex("#6c757d")
    black = from_hex("#000000")

    primarySubtle = from_hex("#cfe2ff")
    secondarySubtle = from_hex("#e2e3e5")
    successSubtle = from_hex("#d1e7dd")
    infoSubtle = from_hex("#cff4fc")
    warningSubtle = from_hex("#fff3cd")
    dangerSubtle = from_hex("#f8d7da")
    lightSubtle = from_hex("#fcfcfd")
    darkSubtle = from_hex("#ced4da")

    primaryBorderSubtle = from_hex("#9ec5fe")
    secondaryBorderSubtle = from_hex("#c4c8cb")
    successBorderSubtle = from_hex("#a3cfbb")
    infoBorderSubtle = from_hex("#9eeaf9")
    warningBorderSubtle = from_hex("#ffe69c")
    dangerBorderSubtle = from_hex("#f1aeb5")
    lightBorderSubtle = from_hex("#e9ecef")
    darkBorderSubtle = from_hex("#adb5bd")

    gray100 = from_hex("#f8f9fa")
    gray200 = from_hex("#e9ecef")
    gray300 = from_hex("#dee2e6")
    gray400 = from_hex("#ced4da")
    gray500 = from_hex("#adb5bd")
    gray600 = from_hex("#6c757d")
    gray700 = from_hex("#495057")
    gray800 = from_hex("#343a40")
    gray900 = from_hex("#212529")
