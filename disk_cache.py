#!/usr/bin/python

import sys


class C:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return k(self._y)

    def z(self):
        return self.x() + self.y()


def k(x):
    return x + 1


def disk_cache(f):
    import functools
    import inspect
    import hashlib
    import os
    import pickle
    import re

    code = inspect.getsource(f)
    f_dir = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "file_cache.py.d",
        re.sub(r"^" + os.path.sep + r"+", "", os.path.abspath(__file__)),
        re.sub(r"^" + os.path.sep + r"+", "", os.path.abspath(os.getcwd())),
        hashlib.sha256(bytes(code, "utf-8")).hexdigest(),
    )
    os.makedirs(f_dir, exist_ok=True)
    with open(os.path.join(f_dir, "code.py"), "w") as fp:
        fp.write(code)

    @functools.wraps(f)
    def _f(*args, **kwargs):
        path = os.path.join(
            f_dir,
            hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest() + ".pickle",
        )
        try:
            with open(path, "rb") as fp:
                val = pickle.load(fp)
        except:
            val = f(*args, *kwargs)
            with open(path, "wb") as fp:
                pickle.dump(val, fp)
        return val
    return _f


@disk_cache
def f(x, y):
    print(x, y)
    return [x, y, "ok?"]


def main(argv):
    print(f([1, 2, 3], C([1, 2], {4: [1, 2, 3]})))
    print(f([1, 2, 3], C([1, 2], {4: [1, 2, 3, 4]})))
    print(f([1, 2, 3], C([1, 2], {4: [1, 2, 3, 4, 5]})))
    print(f([1, 2, 3], C([1, 2], {4: [1, 2, 3, 4, 5, 6]})))


def _usage_and_exit(s=1):
    if s == 0:
        fp = sys.stdout
    else:
        fp = sys.stderr
    print('{}'.format(__file__), file=fp)
    exit(s)


if __name__ == '__main__':
    main(sys.argv)
