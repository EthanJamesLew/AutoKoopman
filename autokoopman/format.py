def _clip_str(s, nlength=20):
    if len(s) <= nlength:
        return s
    else:
        return f"{s[:nlength]}..."


def _clip_list(s, nlength=20):
    return f"[{', '.join([str(si) for si in s])}]"