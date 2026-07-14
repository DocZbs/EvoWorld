from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("evoworld")
except PackageNotFoundError:
    # Process bridges can execute directly from a source checkout inside an
    # upstream environment without overlaying EvoWorld's dependency set.
    __version__ = "0+source"
