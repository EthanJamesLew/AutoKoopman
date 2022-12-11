"""
Utilties to format strings
"""
import sys
import os
import warnings

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


class hide_prints:
    """context manager to suppress python print output

    this was created to hide "ugly stuff" 
    (e.g., GPyOpt prints)
    """

    def __enter__(self, hide_warnings = True):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        self.hw = hide_warnings

        # TODO: it might make more sense to capture this
        # as a log
        if self.hw:
            self.cw = warnings.catch_warnings()
            self.cw.__enter__()
            warnings.filterwarnings('ignore')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        if self.hw:
            self.cw.__exit__()

