"""
Utilties to format strings
"""
from typing import Sequence


def _clip_str(s: str, nlength=20) -> str:
    """
    if a string is too long, clip it to a fixed length and add ...
    :param s: string
    :param nlength: length to clip
    :return: clipped string
    """
    if len(s) <= nlength:
        return s
    else:
        return f"{s[:nlength]}..."


def _clip_list(s: Sequence, nlength=20) -> str:
    """
    if a list is too long as a string, clip it to a fixed length
    :param s: string
    :param nlength: length to clip
    :return: clipped string
    """
    return f"[{', '.join([str(si) for si in s])}]"