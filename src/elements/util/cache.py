import functools

# Memoizes a function on its arguments, keeping only the latest few results: cached operators
# can be large dense matrices, and continuous parameters such as slider values rarely repeat.
cached = functools.lru_cache(maxsize=8)
