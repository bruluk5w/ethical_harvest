import atexit
import concurrent.futures
import multiprocessing
from multiprocessing import parent_process


_PROCESS_POOL = None  # type: concurrent.futures.ProcessPoolExecutor


def _process_pool_init():
    global _PROCESS_POOL
    if _PROCESS_POOL is None and parent_process() is None:
        _PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())


def get_process_pool():
    _process_pool_init()
    return _PROCESS_POOL


@atexit.register
def _cleanup():
    print("Cleanup")
    if _PROCESS_POOL is not None:
        _PROCESS_POOL.shutdown(wait=False)


def init_tensorflow():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
