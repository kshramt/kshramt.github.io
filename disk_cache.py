#!/usr/bin/python

def disk_cache(
        cache_dir=None,
        open_data=open,
        suffix=".pickle",
):
    """
    You can use `@disc_cache(open_data=gzip.open, suffix=".picklez")` to compress caches.
    """
    import os

    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "file_cache.py.d",
        )

    def deco(f):
        import functools
        import inspect
        import hashlib
        import pickle
        import re
        import sys

        cwd = os.path.abspath(os.getcwd())
        path_to_main = os.path.abspath(sys.argv[0])
        path_to_call_site = os.path.abspath(inspect.stack()[1].filename)
        path_to_source = os.path.abspath(inspect.getsourcefile(f))
        code = inspect.getsource(f)
        signature = "\n".join((cwd, path_to_main, path_to_call_site, path_to_source, code))
        f_dir = os.path.join(cache_dir, hashlib.sha256(bytes(signature, "utf-8")).hexdigest())
        os.makedirs(f_dir, exist_ok=True)
        with open(os.path.join(f_dir, "signature.txt"), "w") as fp:
            fp.write(signature)

        @functools.wraps(f)
        def _f(*args, **kwargs):
            path = os.path.join(
                f_dir,
                hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest() + suffix,
            )
            try:
                with open_data(path, "rb") as fp:
                    val = pickle.load(fp)
            except:
                val = f(*args, **kwargs)
                with open_data(path, "wb") as fp:
                    pickle.dump(val, fp)
            return val
        return _f
    return deco
