#!/usr/bin/python

import gzip
import sys

import disk_cache


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


@disk_cache.disk_cache(cache_dir="__cache__", open_data=gzip.open, suffix=".picklez")
def f(x, y):
    print(x, y)
    return [x, y, "ok?"]


def main(argv):
    print(f([1, 2, 3], C([1, 2], {4: [1, 2, 3]})))
    print(f([1, 2, 3], C([1, 2], {4: [1, 2, 3, 4]})))
    print(f([1, 2, 3], C([1, 2], {4: [1, 2, 3, 4, 5]})))
    print(f([1, 2, 3], C([1, 2], {4: [1, 2, 3, 4, 5]})))
    print(f(list(range(100)), list(range(200))))


def _usage_and_exit(s=1):
    if s == 0:
        fp = sys.stdout
    else:
        fp = sys.stderr
    print('{}'.format(__file__), file=fp)
    exit(s)


if __name__ == '__main__':
    main(sys.argv)
