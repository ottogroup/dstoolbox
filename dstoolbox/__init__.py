"""Set __version__ attribute."""

import pkg_resources


try:
    __version__ = pkg_resources.get_distribution("dstoolbox").version
except:  # pylint: disable=bare-except
    __version__ = 'n/a'
