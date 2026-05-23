import multiprocessing as mp
import os
import queue
import traceback
from typing import Any, Callable, Iterable, List, Optional, Sequence

from tqdm import tqdm


def default_process_start_method() -> str:
    methods = mp.get_all_start_methods()
    if os.name != "nt" and "fork" in methods:
        return "fork"
    if "spawn" in methods:
        return "spawn"
    return methods[0]


def _worker_loop(
    task_queue,
    result_queue,
    process_fn: Callable[[Any], Any],
    init_worker: Optional[Callable[..., None]],
    initargs: Sequence[Any],
) -> None:
    if init_worker is not None:
        init_worker(*initargs)

    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            result = process_fn(task)
        except Exception as exc:  # pragma: no cover - defensive fallback
            result = {
                "ok": False,
                "idx": -1,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        result_queue.put(result)


def run_tasks_with_true_processes(
    tasks: Iterable[Any],
    num_workers: int,
    process_fn: Callable[[Any], Any],
    init_worker: Optional[Callable[..., None]] = None,
    initargs: Sequence[Any] = (),
    desc: str = "Processing",
    unit: str = "task",
    start_method: Optional[str] = None,
) -> List[Any]:
    task_list = list(tasks)
    if not task_list:
        return []

    if int(num_workers) <= 1:
        if init_worker is not None:
            init_worker(*initargs)
        results = []
        for task in tqdm(task_list, total=len(task_list), desc=desc, unit=unit):
            try:
                results.append(process_fn(task))
            except Exception as exc:  # pragma: no cover - defensive fallback
                results.append(
                    {
                        "ok": False,
                        "idx": -1,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        return results

    method = start_method or default_process_start_method()
    ctx = mp.get_context(method)
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []
    try:
        for _ in range(int(num_workers)):
            proc = ctx.Process(
                target=_worker_loop,
                args=(task_queue, result_queue, process_fn, init_worker, tuple(initargs)),
            )
            proc.daemon = False
            proc.start()
            workers.append(proc)

        for task in task_list:
            task_queue.put(task)
        for _ in workers:
            task_queue.put(None)

        results = []
        completed = 0
        with tqdm(total=len(task_list), desc=desc, unit=unit) as pbar:
            while completed < len(task_list):
                try:
                    item = result_queue.get(timeout=1.0)
                except queue.Empty:
                    crashed = [proc for proc in workers if proc.exitcode not in (None, 0)]
                    if crashed:
                        raise RuntimeError(
                            "worker exited unexpectedly: "
                            + ", ".join(f"pid={proc.pid}, exitcode={proc.exitcode}" for proc in crashed)
                        )
                    if not any(proc.is_alive() for proc in workers):
                        raise RuntimeError(
                            f"all workers exited before all tasks completed ({completed}/{len(task_list)})"
                        )
                    continue
                results.append(item)
                completed += 1
                pbar.update(1)

        for proc in workers:
            proc.join()
            if proc.exitcode not in (0, None):
                raise RuntimeError(f"worker pid={proc.pid} exited with code {proc.exitcode}")
        return results
    except Exception:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join(timeout=5.0)
        raise
    finally:
        task_queue.close()
        result_queue.close()
