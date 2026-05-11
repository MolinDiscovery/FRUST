from contextlib import contextmanager

darkmode: bool = False


def set_theme(dark: bool = True) -> None:
    """Set module-wide visualization theme.

    Parameters
    ----------
    dark
        If ``True``, helpers that support a dark background render with dark
        mode defaults.
    """
    global darkmode
    darkmode = dark


@contextmanager
def use_darkmode(on: bool = True):
    """Temporarily enable dark mode within a context manager.

    Parameters
    ----------
    on
        Whether dark mode should be active inside the context.

    Yields
    ------
    None
        Control returns to the caller with the previous setting restored when
        the context exits.
    """
    global darkmode
    prev = darkmode
    darkmode = on
    try:
        yield
    finally:
        darkmode = prev
