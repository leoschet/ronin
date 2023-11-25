import asyncio
from functools import wraps


def coroutine(coro):
    @wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper
