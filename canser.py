"""
Supported data types:

- Integer (64 bits)
- Float (64 bits)
- String (UTF-8)
- List
- Dictionary

Little endianness is assumed.
"""

import struct


def save(x, fp):
    if isinstance(x, float):
        fp.write(b"f")
        fp.write(struct.pack("<d", x))
    elif isinstance(x, int):
        fp.write(b"i")
        _save_int(x, fp)
    elif isinstance(x, str):
        b = x.encode("utf-8")
        fp.write(b"s")
        _save_int(len(b), fp)
        fp.write(b)
    elif isinstance(x, list):
        fp.write(b"l")
        _save_int(len(x), fp)
        for v in x:
            save(v, fp)
    elif isinstance(x, dict):
        fp.write(b"d")
        _save_int(len(x), fp)
        for k in sorted(x.keys()):
            save(k, fp)
            save(x[k], fp)
    else:
        raise ValueError(f"Unsupported argument {x} of type {type(x)} for `save`")


def load(fp):
    tag = fp.read(1)
    if tag == b"f":
        return struct.unpack("<d", fp.read(8))[0]
    elif tag == b"i":
        return _load_int(fp)
    elif tag == b"s":
        return fp.read(_load_int(fp)).decode("utf-8")
    elif tag == b"l":
        return [load(fp) for _ in range(_load_int(fp))]
    elif tag == b"d":
        d = dict()
        for _ in range(_load_int(fp)):
            k = load(fp)
            v = load(fp)
            d[k] = v
        return d
    else:
        raise ValueError(f"Invalid tag {tag} for {fp}")


def _save_int(x, fp):
    return fp.write(struct.pack("<q", x))


def _load_int(fp):
    return struct.unpack("<q", fp.read(8))[0]
