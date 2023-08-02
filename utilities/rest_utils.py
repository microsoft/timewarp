import asyncio
import os
from functools import partial
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union, no_type_check

import aiohttp
import requests
from aiohttp.client_exceptions import ServerDisconnectedError
from tqdm import tqdm

from utilities.logging import get_logger

logger = get_logger(__name__)


def download(src: str, dest: Union[os.PathLike, str]) -> None:
    """Simple request.get call with a progress bar
    Args:
        src: URL to be retrieved
        dest: Local destination path
    """
    r = requests.get(src, stream=True)
    tsize = int(r.headers.get("content-length", 0))
    progress = tqdm(total=tsize, unit="iB", unit_scale=True, position=0, leave=False)

    with open(dest, "wb") as handle:
        progress.set_description(os.path.basename(dest))
        for chunk in r.iter_content(chunk_size=1024):
            handle.write(chunk)
            progress.update(len(chunk))


def get_async_request(
    async_func: Coroutine, headers: Optional[Dict[str, str]] = None, **kwargs
) -> Dict[Any, Any]:
    """Simple async `requests.get()`. Mainly used for debugging.

    Args:
        async_func: The async coroutine to call
        headers: Optional headers to pass.
        **kwargs: Keyword arguments to pass to `async_func`
    """

    @no_type_check
    async def _async_wrapper(fun: Coroutine, headers: Optional[Dict[str, str]] = headers):
        session = get_aiohttp_session(limit_connections=1, headers=headers)

        async with session:
            task = asyncio.create_task(fun(session=session))
            res = [await t for t in asyncio.as_completed([task])]
        return res

    fun = partial(async_func, **kwargs)  # type: ignore

    res = asyncio.run(_async_wrapper(fun))
    return res[0]


def get_aiohttp_session(
    headers: Optional[Dict[str, str]] = None,
    limit_connections: int = 3,
    max_timeout_minutes: float = 60.0,
) -> aiohttp.ClientSession:
    """Returns an `aiohttp.ClientSession` instance with additional optional parameters

    Args:
        headers: Headers to use in the GET request for queries.
        limit_connections: Maximum number of concurrent connections.
        max_timeout_minutes: Maximum overall time before sessions shows a `TimeoutError` exception.
    """
    connector = aiohttp.TCPConnector(limit=limit_connections, limit_per_host=limit_connections)
    timeout = aiohttp.ClientTimeout(total=max_timeout_minutes * 60)
    session = aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout)
    return session


async def get_with_retry(
    url: str, session: aiohttp.ClientSession, wait_time: int = 5
) -> Optional[str]:
    """async GET request with retrial after sleep if unsuccessful

    Args:
        url: URL to send the GET request
        session: aiohttp session
        wait_time: Waiting time (in seconds) if request unsuccessful.
    """
    request_ok = False
    while not request_ok:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    info = await resp.text()
                    request_ok = True
                elif resp.status == 202:
                    logger.warning("Obtained processing status code 202. Waiting and retrying")
                    await asyncio.sleep(wait_time)
                elif resp.status == 429:  # too many requests
                    logger.warning("Obtained too-many-requests status code. Waiting and retrying")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Obtained non-200 status code: {resp.status}. Returning `None` for url: {url}"
                    )
                    return None
        except ServerDisconnectedError:
            logger.warning("Server disconnected. Waiting and retrying...")
            await asyncio.sleep(wait_time)
    return info


async def post_with_retry(
    url: str,
    session: aiohttp.ClientSession,
    wait_time: int = 5,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[dict] = None,
    data: Optional[Any] = None,
) -> Optional[str]:
    """async POST request with retrial after sleep if unsuccessful

    Args:
        url: URL to send the POST request
        session: aiohttp session
        wait_time: Waiting time (in seconds) if request unsuccessful.
    """
    request_ok = False
    while not request_ok:
        try:
            async with session.post(url, headers=headers, json=json, data=data) as resp:
                if resp.status == 200 or resp.status == 202:
                    info = await resp.text()
                    request_ok = True
                elif resp.status == 429:  # too many requests
                    logger.warning("Obtained too-many-requests status code. Waiting and retrying")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Obtained non-200 or non-202 status code: {resp.status}: {resp.reason}. Returning `None` for url: {url}"
                    )
                    return None
        except ServerDisconnectedError:
            logger.warning("Server disconnected. Waiting and retrying...")
            await asyncio.sleep(wait_time)
    return info


async def aiohttp_worker(
    queue: asyncio.Queue,
    func: Callable,
    session: aiohttp.ClientSession,
    callback: Optional[Callable] = None,
    **kwargs,
):
    """Dummy worker function that gets an item from the queue and runs a function on it
    Args:
        queue: An async queue
        func: The function to call
        session: The current aiohttp session
        kwargs: keyword arguments to pass to `func`
    """
    while True:
        kwargs = await queue.get()
        await func(session=session, **kwargs)

        queue.task_done()
        if callback is not None:
            callback()


async def map_aiohttp_func(
    func: Callable,
    kwarg_list: List[Dict[str, Any]],
    n_concurrent: int = 3,
    limit_connections: int = 3,
    progress: bool = True,
):
    """Apply an async function to a list, using the same
    aiohttp session.
    Args:
        func: function to be called
        kwarg_list: list of keyword arguments dicts, one per call
        n_concurrent: number of concurrent threads to go over the list
        limit_connections: Number of concurrent connections for the aiohttp session
        progress: Whether to report remaining items in the queue
    """
    session = get_aiohttp_session(limit_connections=limit_connections)
    queue: asyncio.Queue = asyncio.Queue()

    for kwargs in kwarg_list:
        queue.put_nowait(kwargs)

    tasks: List[asyncio.Task] = []

    progress_bar = tqdm(total=len(kwarg_list), disable=not progress)

    with progress_bar as pbar:
        async with session:
            for _ in range(n_concurrent):
                task = asyncio.create_task(
                    aiohttp_worker(queue, func, session, callback=lambda: pbar.update(1))
                )
                tasks.append(task)

            await queue.join()

            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
