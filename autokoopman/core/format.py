"""
Utilties to format strings
"""
from typing import Sequence, Optional


def _clip_str(s: Optional[str], nlength=20) -> str:
    """
    if a string is too long, clip it to a fixed length and add ...
    :param s: string
    :param nlength: length to clip
    :return: clipped string
    """
    if s is None:
        return "None"
    if len(s) <= nlength:
        return s
    else:
        return f"{s[:nlength]}..."


def _clip_list(s: Optional[Sequence], nlength=20) -> str:
    """
    if a list is too long as a string, clip it to a fixed length
    :param s: string
    :param nlength: length to clip
    :return: clipped string
    """
    if s is None:
        return "None"
    return f"[{', '.join([str(si) for si in s])}]"
