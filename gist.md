```python
import asyncio
import collections
from typing import Any, Generic, TypeVar

_T1 = TypeVar("_T1")


class LazyTask(Generic[_T1]):
    def __init__(self, coro: collections.abc.Coroutine[Any, Any, _T1]) -> None:
        self._coro = coro
        self._task: None | asyncio.Task[_T1] = None

    def __await__(self) -> collections.abc.Generator[Any, None, _T1]:
        if self._task is None:
            self._task = asyncio.create_task(self._coro)
        return self._task.__await__()


async def f() -> int:
    return 8


t = LazyTask(f())


async def g() -> None:
    print(await t)
    print(await t)


asyncio.run(g())
```


```typescript
class LruCache<K, V> {
  capacity: number;
  buf: Map<K, V>;
  constructor(capacity: number) {
    if (capacity < 1) {
      throw Error("capacity < 1");
    }
    this.capacity = capacity;
    this.buf = new Map<K, V>();
  }
  has = (k: K) => this.buf.has(k);
  get = (k: K) => {
    if (this.has(k)) {
      const v = this.buf.get(k) as V;
      this.buf.delete(k);
      this.buf.set(k, v);
      return v;
    }
  };
  set = (k: K, v: V) => {
    if (this.buf.has(k)) {
      this.buf.delete(k);
    } else if (this.buf.size === this.capacity) {
      this.buf.delete(this.oldest());
    }
    this.buf.set(k, v);
    return this;
  };
  delete = (k: K) => this.buf.delete(k);
  clear = () => this.buf.clear();
  oldest = () => this.buf.keys().next().value;
}

const amemo1 = <K, V>(f: (k: K) => Promise<V>, capacity: number) => {
  const cache = new LruCache<K, Promise<V>>(capacity);
  const res = async (k: K) => {
    if (cache.has(k)) {
      return cache.get(k) as Promise<V>;
    }
    const v = (async () => {
      try {
        // await is intentional.
        return await f(k);
      } catch (e) {
        cache.delete(k);
        throw e;
      }
    })();
    cache.set(k, v);
    return v;
  };
  res.clear = cache.clear;
  return res;
};

let G = 0;

let slow = async (s) => {
  console.log("slow", s);
  await new Promise((resolve) => setTimeout(resolve, 1000));
  G += 1;
  console.log("settimeout", G);
  return `${s}/${G}`;
};

const cslow = amemo1(slow);

cslow("p").then((v) => console.log("p1", v, cache));
cslow("p").then((v) => console.log("p2", v, cache));
cslow("q").then((v) => console.log("q1", v, cache));
cslow("p").then((v) => {
  console.log("p3", v, cache);
  cslow("p").then((v) => console.log("p4", v, cache));
});
```

```python3
import google.cloud.storage


def open2_of(gs=None):
    def open_of(uri):
        if gs and uri.startswith("gs://"):
            return gs
        else:
            return open

    def open2(uri: str, mode="r"):
        return open_of(uri)(uri, mode=mode)

    return open2


def gs_open_of(client: google.cloud.storage.Client):
    def gs_open(uri: str, mode: str = "r"):
        blob = google.cloud.storage.Blob.from_string(uri, client)
        return blob.open(mode=mode)

    return gs_open


open2 = open2_of(gs=gs_open_of(client=google.cloud.storage.Client()))
```

```python3
import numpy as np
import torch

a = np.frombuffer(b"abc", np.uint8)
t = torch.as_tensor(a)
```

```python3
import tensorflow as tf
import numpy as np


ntrain = 100
xtrain = np.random.randn(ntrain, 2)
ytrain = np.random.randn(ntrain, 1)

nval = 50
xval = np.random.randn(nval, 2)
yval = np.random.randn(nval, 1)

dtrain = tf.data.Dataset.from_tensor_slices((xtrain, ytrain)).batch(10).repeat()
dval = tf.data.Dataset.from_tensor_slices((xval, yval)).batch(10).repeat()


class LWLD(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        lr_init: float,
        lr_high: float,
        lr_last: float,
        n_warmup: int,
        n_decay: int,
    ):
        assert 0 < n_warmup < n_decay, (n_warmup, n_decay)
        assert 0 < lr_init, lr_init
        assert 0 < lr_high, lr_high
        assert 0 < lr_last, lr_last
        super().__init__()
        self.lr_init = lr_init
        self.lr_high = lr_high
        self.lr_last = lr_last
        self.n_warmup = n_warmup
        self.n_decay = n_decay

    @tf.function  # Suppress "Tensorflow error: Using a `tf.Tensor` as a Python `bool` is not allowed" error.
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        if step <= self.n_warmup:
            ret = self.lr_init + (self.lr_high - self.lr_init) * step / self.n_warmup
        elif step <= self.n_decay:
            ret = self.lr_high + (self.lr_last - self.lr_high) * (
                step - self.n_warmup
            ) / (self.n_decay - self.n_warmup)
        else:
            ret = self.lr_last
        tf.print("step", step, ret)
        return ret


model = tf.keras.Sequential([tf.keras.layers.Dense(10), tf.keras.layers.Dense(1)])
model.layers[0].trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LWLD(1e-5, 1e-4, 1e-5, 3, 20)),
    loss=tf.keras.losses.MeanSquaredError(),
)
model.fit(dtrain, epochs=2, steps_per_epoch=3)
model.layers[0].trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LWLD(1e-5, 1e-4, 1e-5, 3, 20)),
    loss=tf.keras.losses.MeanSquaredError(),
)
model.fit(
    dtrain,
    epochs=13,
    steps_per_epoch=3,
    initial_epoch=10,
)
```

```bash
# Dockerfile
# chmod a+rwx /home

docker run \
--init \
--security-opt seccomp=unconfined \
--rm \
-p 8888:8888 \
--mount type=bind,source=/etc/passwd,target=/etc/passwd,readonly \
--mount type=bind,source=/etc/group,target=/etc/group,readonly \
--mount type=bind,source=/data,target=/data \
-w"$(pwd)" \
-u"$(id -u):$(id -g)" \
IMAGE:"$(date +'%y%m%d')" \
nice -n19 jupyter lab --LabApp.token='' --no-browser --port=8888 --ip='*'

```

```
run chmod a+rwx /home
```

```bash
pstree -al
```

```bash
docker build --force-rm -t xx/yy:1 .
```

```bash
docker run \
--init \
--security-opt seccomp=unconfined \
--rm \
-it \
-p 8888:8888 \
--mount type=bind,source="$(pwd)",target=/work/h \
IMAGE:"$(date +'%y%m%d')" \
nice -n19 jupyter lab \
--LabApp.token='' \
--no-browser \
--port=8888 \
--allow-root \
--ip='*'
```

```
import ipdb


def main():
    a=1
    try:
        b = 8
        # raise
        print(b)
    except:
        ipdb.set_trace()


if __name__ == "__main__":
    main()
```

```
import time
import random
import multiprocessing
import numpy as np


def seed(q, delta):
    seed = q.get()
    q.put(seed + delta)
    random.seed(seed)
    np.random.seed(seed)
    print(multiprocessing.current_process(), seed, random.random(), np.random.random())


def f(x):
    time.sleep(1)
    print(multiprocessing.current_process(), x, random.random(), np.random.random())


if __name__ == "__main__":
    jobs = 4
    q = multiprocessing.Queue(1)
    q.put(1)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(jobs, seed, (q, 1)) as pool:
        pool.map(f, range(10))
```

```
import gzip
import json
import os
import functools
import pickle


def dump_pickle(path, x, open=open):
    mkdir(dirname(path))
    with open(path, "wb") as fp:
        return pickle.dump(x, fp)


def load_pickle(path, open=open):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def dump_json(path, x, open=open, **kwargs):
    mkdir(dirname(path))
    with open(path, "wt") as fp:
        return json.dump(x, fp, **kwargs)


def load_json(path, open=open):
    with open(path, "rt") as fp:
        return json.load(fp)

dump_pickle_gz = functools.partial(dump_pickle, open=gzip.open)
load_pickle_gz = functools.partial(load_pickle, open=gzip.open)
dump_json_gz = functools.partial(dump_json, open=gzip.open)
load_json_gz = functools.partial(load_json, open=gzip.open)


def dirname(path):
    return os.path.dirname(path) or os.path.curdir

def mkdir(path):
    return os.makedirs(path, exist_ok=True)
```

```
import contextlib
import  logging
import time

logger = logging.getLogger()

@contextlib.contextmanager
def timing(msg, fn=None):
    if fn is None:
        fn = logger.info
    t1 = time.monotonic()
    yield
    t2 = time.monotonic()
    fn(msg, t2 - t1)
```

```
import re


def trie_pat_of_v2(xss, esc=re.escape):
    assert xss, xss
    xss = sorted(xss, reverse=True)
    ns = [len(xs) for xs in xss]

    def update_branches(branches, x_prev, j, n, ii1, ii2):
        if x_prev is not None:
            branch = esc(x_prev)
            if j < n:
                branch += impl(ii1, ii2, j + 1)
            branches.append(branch)

    def impl(i1, i2, j):
        assert 0 < i2 - i1
        branches = []
        ii1 = i1
        x_prev = None
        n_prev = None
        for i in range(i1, i2):
            xs = xss[i]
            n = ns[i]
            x = xs[j] if j < n else ""
            if x != x_prev:
                ii2 = i
                update_branches(branches, x_prev, j, n_prev, ii1, ii2)
                ii1 = ii2
                x_prev = x
            n_prev = n
        ii2 = i2
        update_branches(branches, x_prev, j, n_prev, ii1, ii2)
        assert branches, branches
        return branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"

    return impl(0, len(xss), 0)


def trie_pat_of_v1(xss):
    xss = sorted(xss, reverse=True)
    root = dict()
    for xs in xss:
        node = root
        for x in xs:
            if x not in node:
                node[x] = dict()
            node = node[x]
        node[""] = dict()

    def impl(node):
        if len(node) <= 0:
            return ""
        elif len(node) == 1:
            k = next(iter(node))
            return re.escape(k) + impl(node[k])
        else:
            return "(?:" + "|".join((re.escape(k) + impl(node[k])) for k in node) + ")"

    return impl(root)


if __name__ == "__main__":
    import functools
    import random
    import string

    def gen_str(rng: random.Random, n1, n2):
        return "".join(rng.choice(string.printable) for _ in range(rng.randint(n1, n2)))
    def gen_list(rng, n1, n2, gen_element):
        return [gen_element() for _ in range(rng.randint(n1, n2))]
    rng = random.Random(42)
    gen_element = functools.partial(gen_str, rng, 0, 20)
    for _ in range(1000):
        xss = gen_list(rng, 1, 20, gen_element)
        pat1 = trie_pat_of_v1(xss)
        pat2 = trie_pat_of_v2(xss)
        assert pat1 == pat2, pat1 + "\n" + pat2
```

```
gs -dPrinted=false -sDEVICE=pdfwrite -o cropped.pdf -c '[/CropBox [0 0 621 792] /PAGES pdfmark' -f input.pdf
```

```
# pip install mecab-python3
import dataclasses
import typing

import MeCab


class MecabV1:
    def __init__(self, args="-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"):
        self.mecab = MeCab.Tagger(args)

    def __call__(self, t: str):
        return list(self._call(t))

    def tokenize(self, t: str):
        node = self.mecab.parseToNode(t)
        while node:
            yield node.surface, node.feature.split(",")
            node = node.next

    def _call(self, t: str):
        n = len(t)
        i1 = 0
        for surface, features in self.tokenize(t):
            if features[0] == "BOS/EOS":
                continue
            l = len(surface)
            while True:
                if i1 >= n:
                    raise ValueError(f"Failed to align {surface} for {t}")
                if surface == t[i1 : i1 + l]:
                    break
                i1 += 1
            i2 = i1 + l
            yield MecabTokenV1(surface, features, (i1, i2))
            i1 = i2


@dataclasses.dataclass
class MecabTokenV1:
    surface: str
    features: typing.List[str]
    span: typing.Tuple[int, int]

mecab = MecabV1("")
for x in mecab("この先生きのこるにはどうすれば良いのか？"):
    print(x)
```

```
import os


class AppendDbV1:
    def __init__(self, path):
        mkdir(path)
        self.path = path
        self.fp_index = open(jp(path, "index.i64"), "a+b")
        self.fp_data = open(jp(path, "data.txt"), "a+b")

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.path)})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fp_index.close()
        self.fp_data.close()

    def __len__(self):
        s = os.fstat(self.fp_index.fileno()).st_size
        assert s % 64 == 0, (self.path, s)
        return s // 64

    def __getitem__(self, i: int):
        i = range(len(self))[i]
        ib1, ib2 = self._ib1_of(i), self._ib2_of(i)
        self.fp_data.seek(ib1)
        return self.fp_data.read(ib2 - ib1).decode("utf-8")

    def append(self, t: str):
        b = t.encode("utf-8")
        l = len(self)
        ib1 = 0 if l == 0 else self._ib_of(l - 1)
        self.fp_data.seek(ib1)
        dib = self.fp_data.write(b)
        try:
            self.fp_index.write(self._bytes(ib1 + dib))
        except OSError:
            self.fp_index.truncate(l * 64)
            raise
        self.flush()

    def flush(self):
        self.fp_index.flush()
        self.fp_data.flush()

    def _ib1_of(self, i):
        return 0 if i == 0 else self._ib_of(i - 1)

    def _ib2_of(self, i):
        return self._ib_of(i)

    def _ib_of(self, i):
        self.fp_index.seek(i * 64)
        return self._int(self.fp_index.read(64))

    @staticmethod
    def _bytes(i: int):
        return i.to_bytes(64, "little", signed=False)

    @staticmethod
    def _int(b: bytes):
        return int.from_bytes(b, "little", signed=False)


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def jp(path, *more):
    return os.path.normpath(os.path.sep.join((path, os.path.sep.join(more))))
```

```
import argparse


def namespacify(x):
    if isinstance(x, dict):
        return argparse.Namespace(**{k, namespacify(v) for k, v in x.items()})
    elif isinstance(x, list):
        return [namespacify(v)) for v in x]
    elif isinstance(x, tuple):
        return tuple(namespacify(v) for v in x)
    else:
        return x


def dictify(x):
    if isinstance(x, argparse.Namespace):
        return dictify(vars(x))
    elif isinstance(x, dict):
        return {k: dictify(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [dictify(v)) for v in x]
    elif isinstance(x, tuple):
        return tuple(dictify(v) for v in x)
    else:
        return x
```


# `~/.gitconfig`

```
[alias]
	co = checkout
	ci = commit
	st = status
	br = branch
	nffm = merge --no-ff
	wdiff = diff --word-diff -w
[core]
	quotepath = false
```

```
python3 -m venv venv
venv/bin/pip install jupyterlab scikit-learn lightgbm torch matplotlib black pip install mypy-ls pyls_isort python-lsp-server pandas tqdm jupyterlab_code_formatter ipywidgets tensorflow
venv/bin/jupyter nbextension enable --py widgetsnbextension
venv/bin/jupyter labextension install @jupyterlab/toc
venv/bin/jupyter labextension install @ryantam626/jupyterlab_code_formatter
venv/bin/jupyter serverextension enable --py jupyterlab_code_formatter
```

```
cat /path | sed -e 's/ー *。/。/g' -e 's/ヽ//g' -e 's/ー *！//g' -e 's/ー *？/？/g' | say -r290 -o path.m4a
```

```
round(
    datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ")
    .replace(tzinfo=datetime.timezone.utc)
    .timestamp()
    * 1000
)
```

```js
const updateIm = (o, path, e) => {
  if (path.length) {
    const k = path[0];
    const t = typeof k;
    path = path.slice(1, path.length);
    if (t === "number") {
      o = o.slice();
    } else if (t === "string") {
      o = {
        ...o,
      };
    } else {
      throw "Unsupported type " + t;
    }
    o[k] = updateIm(o[k], path, e);
    return o;
  } else {
    return e;
  }
};
```

```
ssh -i ~/.ssh/google_compute_engine -N -L localhost:8888:localhost:8888 me@"$(gcloud compute instances list --filter name=instance-1 --format="value(networkInterfaces[].accessConfigs[0].natIP)")"

gcloud compute --project "gcp-project" ssh --zone "us-central1-a" "instance-1" -- -N -L localhost:8888:localhost:8888
```

```
#!/usr/bin/python3

import contextlib
import copy
import itertools
import logging
import os
import queue
import sys


__version__ = "0.1.0"
logger = logging.getLogger(__name__)



class Resource:
    """
    with resources.get() as data:
        work(data)
    """

    def __init__(self, data, queue):
        self.data = data
        self.queue = queue

    def __enter__(self):
        return self.data

    def __exit__(self, exc_type, exc_value, traceback):
        self.queue.put(self.data)


def make_resource_pool(resource_of, n_resources, n_per_resource=1):
    assert n_resources >= 0, n_resources
    assert n_per_resource >= 0, n_per_resource
    data_list = [resource_of(i, n_resources) for i in range(n_resources)]
    data_list = itertools.chain(*(copy.deepcopy(data_list) for _ in range(n_per_resource)))
    queue = queue.Queue(maxsize=len(data_list))
    for data in data_list:
        queue.put(Resource(data, queue))
    return queue
```

```
conda install -c defaults -c conda-forge <pkg>
```

```
julia -e 'println(join(map(_->rand("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"), 1:20), ''))'
```

```
sshfs -onoapplexattr,noappledouble,auto_cache,reconnect me@there:/ ~/disk/there
```

```
%load_ext autoreload
%autoreload 1
%aimport lib
```

# task.py

```python3
import queue
import threading


class TaskContext:

    def __init__(self, q=queue.Queue):
        self.stop = False

        self.queue = q()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def task(self, f):
        return Task(self, f)

    def _loop(self):
        while not self.stop:
            task = self.queue.get(block=True)
            try:
                if not next(task):
                    task.put()

            except StopIteration:
                pass


class Task:

    def __init__(self, ctx, f):
        self._ctx = ctx
        # `f` should behave as if a generator.
        self._g = iter(f(self))

        self.value = None
        self.error = None
        self._waited = queue.Queue()
        self._done = threading.Event()

        self.put()

    def __repr__(self):
        return f"{self.__class__.__name__} {self._g}"

    def __next__(self):
        if self._done.is_set():
            raise StopIteration
        try:
            return next(self._g)
        except StopIteration as e:
            self._done.set()
            self.value = e.value
            self._put_waited()
            raise
        except Exception as e:
            self._done.set()
            self.error = e
            raise

    def __iter__(self):
        return self

    def wait(self, child=None):
        """
        def f(self):
            child = ...
            yield self.wait(child)

        f.wait()
        """
        if child is None:
            self._done.wait()
            return self
        else:
            # I do not need lock here since the `yield self.wait(child)` pattern does not occur in another thread.
            if child._done.is_set():
                return None  # switch = bool(None)
            child._waited.put(self)
            return self  # switch = bool(self)

    def put(self):
        self._ctx.queue.put(self)

    def _put_waited(self):
        assert self._done.is_set(), self
        while True:
            try:
                self._waited.get(block=False).put()
            except queue.Empty:
                break


def _test():
    ctx = TaskContext()

    outs = []

    @ctx.task
    def f(self):
        outs.append(1)
        yield
        outs.append(3)
        yield
        outs.append(4)

    @ctx.task
    def g(self):
        outs.append(2)
        yield self.wait(f)
        outs.append(5)

    @ctx.task
    def h(self):
        yield self.wait(g)

    h.wait()
    oo = outs.copy()
    assert oo == [1, 2, 3, 4, 5], oo
```

```
# https://stackoverflow.com/a/5506483
git stash push <path>
```

```bash
python3 -m doctest -v lib.py
```

```bash
sleep 3600 && osascript -e 'tell application "Finder" to sleep'
```

```bash
ssh -o ProxyCommand='ssh -W %h:%p who@gate.com' -p 22 me@target.com
```

# task.py

```python3
import queue
import threading


work_queue = queue.Queue()


class Task(object):

    def __init__(self, gen):
        super().__init__()
        self.state = "runnable"
        self.gen = gen

    def __repr__(self):
        return f"{self.__class__} gen={self.gen} state={self.state}"

    def schedule(self):
        if self.state in ("runnable", "running"):
            work_queue.put(self)
        return self

    def execute(self):
        assert self.state != "done", self
        self.state = "running"
        try:
            next(self.gen)
        except StopIteration:
            self.state = "done"
            return
        self.schedule()


def schedule(fn):
    return Task(fn()).schedule()


def wait():
    while True:
        try:
            t = work_queue.get(block=False)
        except queue.Empty:
            return
        t.execute()


def _consume():
    while True:
        work_queue.get(block=True).execute()


work_queue_thread = threading.Thread(target=_consume, daemon=True).start()
```

```python3
import time

import task


@task.schedule
def f():
    print(1)
    time.sleep(2)
    yield
    print(2)


@task.schedule
def g():
    print(3)
    time.sleep(2)
    yield
    print(4)


task.wait()
```

# transducers.hs

```haskell
filter' f rf acc x = if f x then rf acc x else acc
map' f rf acc x = rf acc (f x)
transduce tf rf = foldl (tf rf) -- Data.List.foldl
-- Prelude> transduce (map' (\x-> x * x) . filter' (\x-> (rem x 2 == 1))) (+) 0 [1, 2, 3]
-- 10
```

```
cat --show-nonprinting
```

# Word diff

```bash
git diff --no-index --color-words draft.tex draft_v180402074422.tex
```

# `hex_of_deps`

```py
def hex_of_deps(deps):
    def h(s):
        import hashlib
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h("".join(map(h, sorted(set(deps)))))
```

`name/params/deps/multi.dat`

# `random_access_line.py`

```py
def getline(fp, heads, i):
    fp.seek(heads[i])
    return fp.readline()


def heads_of(fp):
    """
    == Inputs
    fp:: Should be opened with a binary mode.

    == Returns

    * Offsets in byte.
    """
    assert isinstance(fp.read(0), bytes), fp
    i_prev = 0
    fp.seek(i_prev)
    for l in fp:
        i_now = i_prev + len(l)
        yield i_prev
        i_prev = i_now


def _test():
    import functools
    import gzip
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # with the EOF \n
        file = os.path.join(td, "s.txt")
        with open(file, "w") as fp:
            fp.write("あい\n")
            fp.write("abc\n")
            fp.write("かきく\n")
        with open(file, "rb") as fp:
            heads = list(heads_of(fp))
        with open(file, "r") as fp:
            gl = functools.partial(getline, fp, heads)
            assert gl(2) == "かきく\n"
            assert gl(0) == "あい\n"
            assert gl(1) == "abc\n"
            assert gl(2) == "かきく\n"
            assert gl(1) == "abc\n"
            assert gl(0) == "あい\n"
            assert gl(1) == "abc\n"
            assert gl(1) == "abc\n"
            assert gl(-1) == "かきく\n"
        # without the EOF \n
        file = os.path.join(td, "s.txt")
        with open(file, "w") as fp:
            fp.write("あい\n")
            fp.write("abc\n")
            fp.write("かきく")
        with open(file, "rb") as fp:
            heads = list(heads_of(fp))
        with open(file, "r") as fp:
            assert getline(fp, heads, 2) == "かきく"
            assert getline(fp, heads, 0) == "あい\n"
            assert getline(fp, heads, 1) == "abc\n"
            assert getline(fp, heads, 2) == "かきく"
            assert getline(fp, heads, 1) == "abc\n"
            assert getline(fp, heads, 0) == "あい\n"
            assert getline(fp, heads, 1) == "abc\n"
            assert getline(fp, heads, 1) == "abc\n"
            assert getline(fp, heads, -1) == "かきく"
        # GZip with the EOF \n
        file = os.path.join(td, "s.txt.gz")
        with gzip.open(file, "wt") as fp:
            fp.write("あい\n")
            fp.write("abc\n")
            fp.write("かきく\n")
        with gzip.open(file, "rb") as fp:
            heads = list(heads_of(fp))
        with gzip.open(file, "rt") as fp:
            gl = functools.partial(getline, fp, heads)
            assert gl(2) == "かきく\n"
            assert gl(0) == "あい\n"
            assert gl(1) == "abc\n"
            assert gl(2) == "かきく\n"
            assert gl(1) == "abc\n"
            assert gl(0) == "あい\n"
            assert gl(1) == "abc\n"
            assert gl(1) == "abc\n"
            assert gl(-1) == "かきく\n"
        # GZip with the EOF \n
        file = os.path.join(td, "s.txt.gz")
        with gzip.open(file, "wt") as fp:
            fp.write("あい\n")
            fp.write("abc\n")
            fp.write("かきく")
        with gzip.open(file, "rb") as fp:
            heads = list(heads_of(fp))
        with gzip.open(file, "rt") as fp:
            assert getline(fp, heads, 2) == "かきく"
            assert getline(fp, heads, 0) == "あい\n"
            assert getline(fp, heads, 1) == "abc\n"
            assert getline(fp, heads, 2) == "かきく"
            assert getline(fp, heads, 1) == "abc\n"
            assert getline(fp, heads, 0) == "あい\n"
            assert getline(fp, heads, 1) == "abc\n"
            assert getline(fp, heads, 1) == "abc\n"
            assert getline(fp, heads, -1) == "かきく"


if __name__ == "__main__":
    _test()
```

# `prep.py`

```py
#!/usr/bin/python

import collections


def stats_of(df, col_fns):
    """
    == Examples

    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(a=[1, 2, 9], b=[-1, 2, 8]))
    >>> stats_of(df, [("a", ["median"]), ("b", ["mean", "median"])])
    OrderedDict([('a', {'median': 2.0}), ('b', {'mean': 3.0, 'median': 2.0})])
    """
    ret = collections.OrderedDict()
    for col, fns in col_fns:
        ret[col] = {fn: getattr(df[col], fn)() for fn in fns}
    return ret


def flag_nan(df, cols):
    """
    == Examples

    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(a=[1.0, None, 9.0], b=[9, 8, 7]))
    >>> flag_nan(df, ["a"])
         a  b  a_nan
    0  1.0  9      0
    1  NaN  8      1
    2  9.0  7      0
    """
    ret = df.copy(deep=False)
    seen = set(ret.columns)
    for col in cols:
        set_uniquely(ret, f"{col}_nan", df[col].isna().astype(int), seen)
    return ret


def impute(df, col_val):
    """
    Replace NaN's in df[col<N>] with val<N>.

    == Examples

    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(a=[1.0, None, 9.0], b=[9, 8, 7]))
    >>> impute(df, [("a", 0)])
         a  b
    0  1.0  9
    1  0.0  8
    2  9.0  7
    """
    ret = df.copy(deep=False)
    for col, val in col_val:
        vs = ret[col].values.copy()
        vs[ret[col].isna().values] = val
        ret[col] = vs
    return ret


def levels_of(df, cols):
    """
    == Examples

    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(a=["a", None, "a", "a", "b", "b"], b=[1, 2, 2, 3, 3, 3]))
    >>> levels_of(df, ["a", "b"])
    OrderedDict([('a', OrderedDict([('a', 3), ('b', 2)])), ('b', OrderedDict([(3, 3), (2, 2), (1, 1)]))])
    """
    ret = collections.OrderedDict()
    for col in cols:
        ret[col] = collections.OrderedDict(sorted(collections.Counter(df[col].dropna().values).items(), key=lambda x: x[1], reverse=True))
    return ret


def one_hot(df, col_levels):
    """
    == Examples

    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(a=["a", None, "a", "a", "b", "b"], b=[1, 2, 2, 3, 3, 3]))
    >>> one_hot(df, [("a", ["a", "b"]), ("b", [2, 1])])
       a_a  a_b  b_2  b_1
    0    1    0    0    1
    1    0    0    1    0
    2    1    0    1    0
    3    1    0    0    0
    4    0    1    0    0
    5    0    1    0    0
    """
    ret = df.copy(deep=False)
    seen = set(ret.columns)
    for col, levels in col_levels:
        ser = ret[col]
        ret = ret.drop(col, axis="columns")
        for level in levels:
            set_uniquely(ret, f"{col}_{level}", (ser == level).values.astype(int), seen)
    return ret


def set_uniquely(df, col, vals, seen):
    assert col not in seen, (col, seen)
    seen.add(col)
    df[col] = vals
    return df
```

# ja

```
javascript:var%20t=((window.getSelection&&window.getSelection())||(document.getSelection&&document.getSelection())||(document.selection&&document.selection.createRange&&document.selection.createRange().text));var%20e=(document.charset||document.characterSet);if(t!=''){location.href='http://translate.google.com/?text='+t+'&hl=ja&langpair=auto|ja&tbb=1&ie='+e;}else{location.href='http://translate.google.com/translate?u='+encodeURIComponent(location.href)+'&hl=ja&langpair=auto|ja&tbb=1&ie='+e;};
```

# plt.rcParms

```py
import matplotlib.pyplot as plt
plt.rcParams["figure.titlesize"] = "xx-large"
plt.rcParams["axes.titlesize"] = "x-large"
plt.rcParams["axes.labelsize"] = "x-large"
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans", "Bitstream Vera Sans"]
plt.rcParams["font.serif"] = ["Noto Serif CJK JP", "DejaVu Serif", "Bitstream Vera Serif"]
```

# dot

```bash
dot -Tpdf -Grankdir=LR -Nshape=plaintext -Ecolor='#00000088'
```

# Cython

```
cat <<EOF >| cy.py
import sys


class C(object):

    def __init__(self, x):
        self.x = x


def main(argv):
    print(argv)
    c = C(3)
    print(c)
    print(c.x)


if __name__ == "__main__":
    main(sys.argv)
EOF

cython --embed cy.py
gcc-mp-7 -O3 -I /opt/local/Library/Frameworks/Python.framework/Versions/3.6/include/python3.6m -L/opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib cy.c
strip -R .note -R .comment a.out

./a.out
 ```

# `ddict`

```py
class ddict(object):
    """
    >>> conf = ddict()
    >>> conf.z = 99
    >>> conf
    ddict(z=99)
    >>> conf = ddict(a=1, b=ddict(c=2, d=ddict(e=3)))
    >>> conf
    ddict(a=1, b=ddict(c=2, d=ddict(e=3)))
    >>> conf.a
    1
    >>> conf.b.c
    2
    >>> conf.a = 99
    >>> conf.b.c = 88
    >>> conf
    ddict(a=99, b=ddict(c=88, d=ddict(e=3)))
    >>> conf.a = 1
    >>> conf.b.c = 2
    >>> conf._update(dict(p=9, r=10))
    ddict(a=1, b=ddict(c=2, d=ddict(e=3)), p=9, r=10)
    >>> conf._to_dict_rec()
    {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}, 'p': 9, 'r': 10}
    >>> conf._of_dict_rec({'a': 1, 'b': {'c': 2, 'd': {'e': 3}}})
    ddict(a=1, b=ddict(c=2, d=ddict(e=3)))
    >>> conf._to_dict()
    {'a': 1, 'b': ddict(c=2, d=ddict(e=3))}
    >>> conf._of_dict({'a': 1, 'b': {'c': 2, 'd': {'e': 3}}, 'p': 9, 'r': 10})
    ddict(a=1, b={'c': 2, 'd': {'e': 3}}, p=9, r=10)
    """

    def __init__(self, **kwargs):
        super().__setattr__("_data", dict())
        self._update(kwargs)

    def __setattr__(self, k, v):
        if k in self.__dict__:
            raise ValueError(f"Tried to overwrite {k} of {self} by {v}")
        self._data[k] = v

    def __getattr__(self, k):
        return self._data[k]

    def __repr__(self):
        args = ", ".join(f"{k}={repr(v)}" for k, v in self._data.items())
        return f"{self.__class__.__name__}({args})"

    def _update(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        return self

    def _to_dict(self):
        return self._data.copy()

    def _to_dict_rec(self):
        return {k: v._to_dict_rec() if isinstance(v, self.__class__) else v for k, v in self._data.items()}

    def _of_dict(self, d):
        self._data.clear()
        return self._update(d)

    def _of_dict_rec(self, d):
        self._data.clear()
        for k, v in d.items():
            setattr(self, k, self.__class__()._of_dict_rec(v) if isinstance(v, dict) else v)
        return self
```

# `normalizer_of`

```py
import numpy as np


def normalizer_of(X, lower, upper, threshold):
    assert 0 <= lower <= upper <= 100
    assert 0 < threshold <= 100
    mean, std = trim_mean_std_of(X, lower, upper)
    Z = X[L2_mask_of((X - mean)/std, threshold)]
    return dict(mean=np.mean(Z, axis=0), std=np.std(Z, axis=0))


def trim_mean_std_of(X, lower, upper):
    assert 0 <= lower <= upper <= 100
    mean = []
    std = []
    for j in range(X.shape[1]):
        Xj = X[:, j]
        Xj_masked = X[(np.percentile(Xj, lower) <= Xj) & (Xj <= np.percentile(Xj, upper)), j]
        mean.append(np.mean(Xj_masked))
        std.append(np.std(Xj_masked))
    return mean, std


def L2_mask_of(X, upper):
    assert 0 <= upper <= 100
    distances = np.sum(X**2, axis=1)
    return distances <= np.percentile(distances, upper)


if __name__ == '__main__':
    a = np.random.randn(900, 5)
    b = np.random.randn(100, 5) + 10
    c = np.vstack([a, b])
    print(normalizer_of(c, 5, 95, 90))
```

# `conf_of`

```
import collections

def _conf_of(**kwargs):
    return collections.namedtuple("_Conf", kwargs.keys())(**kwargs)
```

```
sshfs -o auto_cache -o reconnect
```

```py
with open("lib.py") as fp:
    exec(fp.read())
```

# `build.py`

```py
import collections
import datetime
import json
import os
import pprint
from os.path import basename
import shutil
import subprocess
import sys
import tempfile

import buildpy.v2

os.environ["SHELL"] = "/bin/bash"
os.environ["SHELLOPTS"] = "pipefail:errexit:nounset:noclobber"
os.environ.setdefault("PYTHON", sys.executable)


def info(*xs):
    print("INFO\t" + datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + "\t" + "\t".join(map(str, xs)), file=sys.stderr)


current_time = datetime.datetime.now()
current_hash = subprocess.run(["git", "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
info("CURRENT_TIME:", current_time)
info("CURRENT_HASH:", current_hash)


all_files = set(subprocess.run(["git", "ls-files", "-z"], check=True, universal_newlines=True, stdout=subprocess.PIPE).stdout.split("\0"))

dsl = buildpy.v2.DSL(use_hash=True)
let = dsl.let
loop = dsl.loop
file = dsl.file
phony = dsl.phony
sh = dsl.sh
jp = dsl.jp

dirname = dsl.dirname
mkdir = dsl.mkdir
mv = dsl.mv


julia = os.environ.setdefault("JULIA", "julia")
python = os.environ.setdefault("PYTHON", sys.executable)


vx_b = "bin"
vx_d = "data"
vxw = jp(os.environ.get("WORK_DIR", ".."), "work")
vxt = jp(os.environ.get("TMP_DIR", ".."), "tmp")
daily = jp("..", "daily", current_time.strftime("%y%m%d"))
mkdir(vxt)
mkdir(daily)


def set_daily(path, ekp, daily, env, job="daily"):
    name = kof(path, ekp)
    target = jp(daily, name) + os.path.splitext(path)[1]
    @file([target], [env[name]])
    def _(j):
        assert len(j.ts) == len(j.ds)
        for t, d in zip(j.ts, j.ds):
            mkdir(dirname(t))
            shutil.copy(d, t)
    phony(job, [target])


def filing(
    targets_root_dir, # targets_root_dir/SH/A1_hash/target1
    targets, # [(base_name, prop_dict), ...]
    deps, # {name: path, ...} to ease access inside scripts
    params, # {k: v, ...} specific to this job
    env, # key -> path # human friendly alias
    desc=None,
    use_hash=None,
):
    assert all(isinstance(prop_dict, dict) for _, prop_dict in targets), targets
    assert isinstance(deps, dict), deps
    import functools
    # `deps` should contain all necessary information except for job specific parameters
    arg = _json_of(dict(params=params, deps=deps))
    targets_root_dir = jp(targets_root_dir, _dir_of_str(arg))
    targets = [(jp(targets_root_dir, base_name), kof(base_name, prop_dict))
               for base_name, prop_dict in targets]
    for path, env_key in targets:
        set_uniq(env, env_key, path)
    targets = [path for path, _ in targets]

    # this is the actual decorator
    def deco(f):
        @functools.wraps(f)
        @file(
            targets,
            list(flatten_deps(deps)),
            desc=desc,
            use_hash=use_hash,
        )
        def _f(j):
            arg_path = jp(targets_root_dir, "__arg__.json")
            # make arg.json if necessary
            exist = True
            prev = None
            try:
                with open(arg_path) as fp:
                    prev = fp.read()
            except:
                exist = False
            if (not exist) or (prev != arg):
                mkdir(dirname(arg_path))
                with open(arg_path, "w") as fp:
                    fp.write(arg)
            return f(j, arg_path)
        return _f
    return deco


def make_run_with_temp(exe, tmp, env):
    def run_with_temp(j, arg):
        with tempfile.TemporaryDirectory(dir=tmp) as td:
            temps = [jp(td, basename(t)) for t in j.ts]
            sh(f"{exe} {j.ds[0]} {arg}", input="\0".join(temps), env={**os.environ, **env})
            assert len(temps) == len(j.ts), (temps, j.ts)
            for temp, target in zip(temps, j.ts):
                mkdir(dirname(target))
                mv(temp, target)
    return run_with_temp


# key_of
def kof(name, params):
    return _json_of([name, params])


def flatten_deps(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from flatten_deps(v)
    elif isinstance(x, list):
        for v in x:
            yield from flatten_deps(v)
    else:
        yield x


def _dir_of_str(s):
    import hashlib
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return jp(h[:2], h[2:])


def _json_of(x):
    import json
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def copy_dict(d, ks):
    if isinstance(d, dict):
        return {k: d[k] for k in ks}
    else:
        return {k: getattr(d, k) for k in ks}


def set_uniq(d, k, v):
    if k in d:
        raise ValueError(f"{k} in \n{pprint.pformat(d)}")
    d[k] = v
    return d


def vcat(*ls):
    import itertools
    return list(itertools.chain(*ls))


def write_list(path, l):
    mkdir(dirname(path))
    with open(path, "w") as fp:
        for x in l:
            print(x, file=fp)


def error(msg):
    raise Exception(msg)


def _walk_leaves(d, tform, path):
    for k, v in d.items():
        assert not k.startswith("data"), d
        assert not k.endswith("data"), d
        assert "__" not in k, d
        # todo: this O(len(path)**2)
        p = path + [str(k)]
        if isinstance(v, dict):
            yield from _walk_leaves(v, tform, path=p)
        else:
            yield "__".join(p), tform(v)


def _tuplize(x):
    if isinstance(x, list):
        return tuple(map(_tuplize, x))
    else:
        return x


def get_conf():
    dsl = buildpy.v2.DSL()
    file = dsl.file
    phony = dsl.phony
    sh = dsl.sh

    # conf_yaml = "conf.yaml"
    # conf_json = "conf.json"
    conf_yaml = "conf_small.yaml"
    conf_json = "conf_small.json"

    phony("all", [conf_json])

    @file(conf_json, [jp(vx_b, "make_conf.jl"), conf_yaml])
    def _j(j):
        sh(f"{julia} {j.ds[0]} --out={j.ts[0]} --conf={j.ds[1]}")

    dsl.main([sys.argv[0]])

    # check if `conf.json` have been updated
    assert os.path.getmtime(conf_json) >= os.path.getmtime(conf_yaml)
    with open(conf_json) as fp:
        return dict(_walk_leaves(json.load(fp), _tuplize, []))


conf_dict = get_conf()
Conf = collections.namedtuple("Conf", list(conf_dict.keys()))
conf = Conf(**conf_dict)


# I like side effects
all_figs = []
# for safety and convenience
env = dict()


@let
def _():
    @phony(f"check", [])
    def _(j):
        assert dict(_walk_leaves(dict(a=1, b=dict(c=2, d=dict(e=3))), lambda x: x, [])) == {"a": 1, "b__c": 2, "b__d__e": 3}
        assert list(flatten_deps(dict(a=[1, 2], b=3, d=dict(e=[4, 5])))) == [1, 2, 3, 4, 5]


@phony(f"touch", [], desc=f"Touch all version-controlled files to avoid unwanted rebuilds")
def _(j):
    for path in all_files:
        # mtime = 0 seems to have some special meanings for some build systems
        sh(f"touch --no-create -m --date=1970-01-02 00:00:00 {path}")


# choose shallow layer data
set_uniq(env, "shallow_layers_json", jp(vx_d, conf.cal__shallow_model + "_layers.json"))


@filing(
    vxw,
    [("p_pp.json", dict())],
    dict(exe=jp(vx_b, "get_P_PP.py")),
    dict(
        source_depth_in_km=0,
        distance_in_degree_1=25,
        distance_in_degree_2=95,
        distance_n=71,
    ),
    env,
)
def _(j, arg):
    sh(f"{python} {j.ds[0]} {arg}", input="\0".join(j.ts))


@filing(
    vxw,
    [("recs.jld2", dict())],
    dict(exe=jp(vx_b, "make_recs.jl"), data=env[kof("p_pp.json", dict())]),
    copy_dict(
        conf,
        [
            "rec__dt",
            "rec__n",
            "rec__nt",
            "src__latitude_hypo",
            "src__longitude_hypo",
            "rec__seed",
        ],
    ),
    env,
)
def _(j, arg):
    sh(f"{julia} {j.ds[0]} {arg}", input="\0".join(j.ts))


@filing(
    vxw,
    [("recs.pdf", dict())],
    dict(exe=jp(vx_b, "plot_recs.jl"), data=env[kof("recs.jld2", dict())]),
    copy_dict(
        conf,
        [
            "src__latitude_hypo",
            "src__longitude_hypo",
        ],
    ),
    env,
)
def _(j, arg):
    sh(f"{julia} {j.ds[0]} {arg}", input="\0".join(j.ts))
all_figs.append(env[kof("recs.pdf", dict())])
set_daily("recs.pdf", dict(), daily, env)


@loop(conf.cal__layers_seed_list)
def _(layers_src_seed):
    params = copy_dict(
        conf,
        [
            "cal__sigma_density",
            "cal__sigma_density_max",
            "cal__sigma_vp",
            "cal__sigma_vp_max",
            "cal__sigma_vs",
            "cal__sigma_vs_max",
            "cal__sigma_depth",
            "cal__sigma_depth_max",
        ],
    )
    set_uniq(params, "layers_src_seed", layers_src_seed)
    filing(
        vxw,
        [("layers_src.jld2", dict(layers_src_seed=layers_src_seed))],
        dict(exe=jp(vx_b, "make_layers_src.jl"), data=env["shallow_layers_json"]),
        params,
        env,
    )(make_run_with_temp(julia, vxt, {"OPENBLAS_NUM_THREADS": "1"}))


filing(
    vxw,
    [("layers_src.jld2", dict(layers_src_seed="cal"))],
    dict(
        exe=jp(vx_b, "make_cal_layers_src.jl"),
        layers_src_list=[env[kof("layers_src.jld2", dict(layers_src_seed=layers_src_seed))]
                         for layers_src_seed in conf.cal__layers_seed_list],
    ),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("layers_src.jld2", dict(layers_src_seed="ref"))],
    dict(exe=jp(vx_b, "make_ref_layers_src.jl"), data=env["shallow_layers_json"]),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("layers_src.jld2", dict(layers_src_seed="true"))],
    dict(exe=jp(vx_b, "make_true_layers_src.jl"), data=env["shallow_layers_json"]),
    copy_dict(
        conf,
        [
            "cal__true_layers_seed",
            "cal__sigma_density",
            "cal__sigma_density_max",
            "cal__sigma_vp",
            "cal__sigma_vp_max",
            "cal__sigma_vs",
            "cal__sigma_vs_max",
            "cal__sigma_depth",
            "cal__sigma_depth_max",
            "syn__true_ratio",
        ],
    ),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("layers_src_list.pdf", dict())],
    dict(
        exe=jp(vx_b, "plot_layers_src_list.jl"),
        true_layers_src=env[kof("layers_src.jld2", dict(layers_src_seed="true"))],
        cal_layers_src=env[kof("layers_src.jld2", dict(layers_src_seed="cal"))],
        layers_src_list=[env[kof("layers_src.jld2", dict(layers_src_seed=layers_src_seed))]
                         for layers_src_seed in conf.cal__layers_seed_list],
    ),
    copy_dict(
        conf,
        [
            "cal__base_model",
        ],
    ),
    env,
)(make_run_with_temp(julia, vxt, dict()))
all_figs.append(env[kof("layers_src_list.pdf", dict())])
set_daily("layers_src_list.pdf", dict(), daily, env)


filing(
    vxw,
    [("src_params.jld2", dict())],
    dict(exe=jp(vx_b, "make_src_params.jl")),
    copy_dict(
        conf,
        [
            "src__longitude_hypo",
            "src__latitude_hypo",
            "src__depth",
            "src__strike",
            "src__dip",
            "src__rake",
            "src__vr",
            "src__dt",
            "src__nt_local",
            "src__nt_total",
            "src__nt_sub",
            "src__dx",
            "src__ix_hypo",
            "src__nx",
            "src__nx_sub",
            "src__dy",
            "src__iy_hypo",
            "src__ny",
            "src__ny_sub",
            "src__is_free_surface",
        ],
    ),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("true_m.jld2", dict())],
    dict(exe=jp(vx_b, "make_true_m.jl"), data=env[kof("src_params.jld2", dict())]),
    copy_dict(
        conf,
        [
            "syn__amp_list",
            "syn__it_local_list",
            "syn__ix_list",
            "syn__iy_list",
            "syn__sigma_t_list",
            "syn__sigma_x_list",
            "syn__sigma_y_list",
        ],
    ),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("true_m.pdf", dict())],
    dict(
        exe=jp(vx_b, "plot_m.jl"),
        src_params=env[kof("src_params.jld2", dict())],
        m=env[kof("true_m.jld2", dict())],
    ),
    copy_dict(
        conf,
        [
            "src__nt_total",
            "src__dt",
            "src__ix_hypo",
            "src__iy_hypo",
            "src__nx",
            "src__ny",
        ],
    ),
    env,
)(make_run_with_temp(julia, vxt, dict()))
all_figs.append(env[kof("true_m.pdf", dict())])
set_daily("true_m.pdf", dict(), daily, env)


@loop(vcat(["true", "ref"], conf.cal__layers_seed_list))
def _(layers_src_seed):
    filing(
        vxw,
        [("propagator_params.jld2", dict(layers_src_seed=layers_src_seed))],
        dict(
            exe=jp(vx_b, "make_propagator_params.jl"),
            recs=env[kof("recs.jld2", dict())],
            src_params=env[kof("src_params.jld2", dict())],
            layers_src=env[kof("layers_src.jld2", dict(layers_src_seed=layers_src_seed))],
        ),
        copy_dict(
            conf,
            [
                "cal__nt_green",
                "cal__dt_green",
                "cal__base_model",
                "src__longitude_hypo",
                "src__latitude_hypo",
                "src__depth",
            ],
        ),
        env,
    )(make_run_with_temp(julia, vxt, {"OPENBLAS_NUM_THREADS": "1"}))


filing(
    vxw,
    [("true_d.jld2", dict())],
    dict(
        exe=jp(vx_b, "make_true_d.jl"),
        propagator_params=env[kof("propagator_params.jld2", dict(layers_src_seed="true"))],
        true_m=env[kof("true_m.jld2", dict())],
    ),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("data.jld2", dict())],
    dict(
        exe=jp(vx_b, "make_data.jl"),
        recs=env[kof("recs.jld2", dict())],
        true_d=env[kof("true_d.jld2", dict())],
    ),
    copy_dict(
        conf,
        [
            "syn__obs_noise_seed",
            "syn__sigma_bg_ratio",
        ],
    ),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("Cb.jld2", dict())],
    dict(
        exe=jp(vx_b, "make_Cb.jl"),
        data=env[kof("data.jld2", dict())],
    ),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("Ks.jld2", dict())],
    dict(exe=jp(vx_b, "make_Ks.jl"), data=env[kof("src_params.jld2", dict())]),
    copy_dict(
        conf,
        [
            "analysis__search_space_time",
            "src__is_free_surface",
            "src__ny",
            "src__vr",
        ],
    ),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("d.jld2", dict())],
    dict(exe=jp(vx_b, "make_d.jl"), data=env[kof("data.jld2", dict())]),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("d.pdf", dict())],
    dict(exe=jp(vx_b, "plot_d.jl"), data=env[kof("d.jld2", dict())]),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))
all_figs.append(env[kof("d.pdf", dict())])
set_daily("d.pdf", dict(), daily, env)


@loop(vcat(["true", "ref"], conf.cal__layers_seed_list))
def _(layers_src_seed):
    filing(
        vxw,
        [("inv.jld2", dict(layers_src_seed=layers_src_seed))],
        dict(
            exe=jp(vx_b, "make_inv.jl"),
            d=env[kof("d.jld2", dict())],
            Ks=env[kof("Ks.jld2", dict())],
            Cb=env[kof("Cb.jld2", dict())],
            propagator_params=env[kof("propagator_params.jld2", dict(layers_src_seed=layers_src_seed))],
        ),
        dict(),
        env,
    )(make_run_with_temp(julia, vxt, {"OPENBLAS_NUM_THREADS": "1"}))

    filing(
        vxw,
        [("inv.json", dict(layers_src_seed=layers_src_seed))],
        dict(
            exe=jp(vx_b, "summarize_inv.jl"),
            data=env[kof("inv.jld2", dict(layers_src_seed=layers_src_seed))],
        ),
        dict(),
        env,
    )(make_run_with_temp(julia, vxt, {"OPENBLAS_NUM_THREADS": "1"}))


@loop(
    (
        ("plot_weights.py", "weights.pdf"),
        ("plot_hypers.py", "hypers.pdf"),
        ("plot_lml_logdethess.py", "lml_logdethess.pdf"),
        ("plot_heights.py", "lml_heights.pdf"),
    ),
    tform=lambda x: x,
)
def _(exe_name, target):
    filing(
        vxw,
        [(target, dict())],
        dict(
            exe=jp(vx_b, exe_name),
            data_list=[env[kof("inv.json", dict(layers_src_seed=layers_src_seed))] for layers_src_seed in conf.cal__layers_seed_list],
        ),
        dict(),
        env,
    )(make_run_with_temp(python, vxt, dict()))
    all_figs.append(env[kof(target, dict())])
    set_daily(target, dict(), daily, env)


filing(
    vxw,
    [("marginal_likelihood.jld2", dict())],
    dict(
        exe=jp(vx_b, "make_marginal_likelihood.jl"),
        d=env[kof("d.jld2", dict())],
        Ks=env[kof("Ks.jld2", dict())],
        Cb=env[kof("Cb.jld2", dict())],
        propagator_params=env[kof("propagator_params.jld2", dict(layers_src_seed="ref"))],
        inv=env[kof("inv.json", dict(layers_src_seed="ref"))],
    ),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("marginal_likelihood.pdf", dict())],
    dict(
        exe=jp(vx_b, "plot_marginal_likelihood.jl"),
        data=env[kof("marginal_likelihood.jld2", dict())],
    ),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))
all_figs.append(env[kof("marginal_likelihood.pdf", dict())])
set_daily("marginal_likelihood.pdf", dict(), daily, env)


filing(
    vxw,
    [("posterior_mean_cov.jld2", dict())],
    dict(
        exe=jp(vx_b, "make_posterior_mean_cov.jl"),
        inv_json_list=[env[kof("inv.json", dict(layers_src_seed=layers_src_seed))] for layers_src_seed in conf.cal__layers_seed_list],
        inv_jld2_list=[env[kof("inv.jld2", dict(layers_src_seed=layers_src_seed))] for layers_src_seed in conf.cal__layers_seed_list],
    ),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("posterior_mean.jld2", dict())],
    dict(
        exe=jp(vx_b, "make_posterior_mean.jl"),
        data=env[kof("posterior_mean_cov.jld2", dict())],
    ),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))


filing(
    vxw,
    [("m_list.pdf", dict())],
    dict(
        exe=jp(vx_b, "plot_m_list.jl"),
        src_params=env[kof("src_params.jld2", dict())],
        inv_json_list=[env[kof("inv.json", dict(layers_src_seed=layers_src_seed))] for layers_src_seed in conf.cal__layers_seed_list],
    ),
    copy_dict(
        conf,
        [
            "src__nt_total",
            "src__dt",
            "src__ix_hypo",
            "src__iy_hypo",
            "src__nx",
            "src__ny",
        ],
    ),
    env,
)(make_run_with_temp(julia, vxt, dict()))
all_figs.append(env[kof("m_list.pdf", dict())])
set_daily("m_list.pdf", dict(), daily, env)


filing(
    vxw,
    [("mean_of_posterior_mean.jld2", dict())],
    dict(
        exe=jp(vx_b, "make_mean_of_posterior_mean.jl"),
        inv_json_list=[env[kof("inv.json", dict(layers_src_seed=layers_src_seed))] for layers_src_seed in conf.cal__layers_seed_list],
    ),
    dict(),
    env,
)(make_run_with_temp(julia, vxt, dict()))


@loop(("true", "ref"))
def _(layers_src_seed):
    filing(
        vxw,
        [("posterior_mean.jld2", dict(layers_src_seed=layers_src_seed))],
        dict(
            exe=jp(vx_b, "extract_m_hat.jl"),
            data=env[kof("inv.json", dict(layers_src_seed=layers_src_seed))],
        ),
        dict(),
        env,
    )(make_run_with_temp(julia, vxt, dict()))


@loop(
    (
        (
            [("posterior_mean.pdf", dict())],
            dict(
                exe=jp(vx_b, "plot_m.jl"),
                src_params=env[kof("src_params.jld2", dict())],
                m=env[kof("posterior_mean.jld2", dict())],
            ),
            [
                "src__nt_total",
                "src__dt",
                "src__ix_hypo",
                "src__iy_hypo",
                "src__nx",
                "src__ny",
            ],
            (lambda: all_figs.append(env[kof("posterior_mean.pdf", dict())]),
             lambda: set_daily("posterior_mean.pdf", dict(), daily, env)),
        ),
        (
            [("compare_mean_of_mean_and_true_m.pdf", dict())],
            dict(
                exe=jp(vx_b, "plot_compare_mean_of_mean_m_and_true_m.jl"),
                src_params=env[kof("src_params.jld2", dict())],
                mean_of_posterior_mean=env[kof("mean_of_posterior_mean.jld2", dict())],
                true_m=env[kof("true_m.jld2", dict())],
            ),
            [
                "src__nt_total",
                "src__dt",
                "src__ix_hypo",
                "src__iy_hypo",
                "src__nx",
                "src__ny",
            ],
            (lambda: all_figs.append(env[kof("compare_mean_of_mean_and_true_m.pdf", dict())]),
             lambda: set_daily("compare_mean_of_mean_and_true_m.pdf", dict(), daily, env)),
        ),
    ),
    tform=lambda x: x,
)
def _(targets, deps, ks, posts):
    filing(
        vxw,
        targets,
        deps,
        copy_dict(
            conf,
            ks,
        ),
        env,
    )(make_run_with_temp(julia, vxt, dict()))
    for post in posts:
        post()


@loop(("true", "ref"))
def _(layers_src_seed):
    filing(
        vxw,
        [(f"compare_{layers_src_seed}_posterior_mean_and_true_m.pdf", dict())],
        dict(
            exe=jp(vx_b, "plot_compare_x_m_and_true_m.jl"),
            src_params=env[kof("src_params.jld2", dict())],
            posterior_mean=env[kof("posterior_mean.jld2", dict(layers_src_seed=layers_src_seed))],
            true_m=env[kof("true_m.jld2", dict())],
            inv=env[kof("inv.jld2", dict(layers_src_seed=layers_src_seed))],
        ),
        copy_dict(
            conf,
            [
                "src__nt_total",
                "src__dt",
                "src__ix_hypo",
                "src__iy_hypo",
                "src__nx",
                "src__ny",
            ],
        ),
        env,
    )(make_run_with_temp(julia, vxt, dict()))
    all_figs.append(env[kof(f"compare_{layers_src_seed}_posterior_mean_and_true_m.pdf", dict())])
    set_daily(f"compare_{layers_src_seed}_posterior_mean_and_true_m.pdf", dict(), daily, env)


filing(
    vxw,
    [("compare_posterior_mean_and_true_m.pdf", dict())],
    dict(
        exe=jp(vx_b, "plot_compare_m_and_true_m.jl"),
        src_params=env[kof("src_params.jld2", dict())],
        posterior_mean=env[kof("posterior_mean.jld2", dict())],
        true_m=env[kof("true_m.jld2", dict())],
        posterior_mean_cov=env[kof("posterior_mean_cov.jld2", dict())],
    ),
    copy_dict(
        conf,
        [
            "src__nt_total",
            "src__dt",
            "src__ix_hypo",
            "src__iy_hypo",
            "src__nx",
            "src__ny",
        ],
    ),
    env,
)(make_run_with_temp(julia, vxt, dict()))
all_figs.append(env[kof("compare_posterior_mean_and_true_m.pdf", dict())])
set_daily("compare_posterior_mean_and_true_m.pdf", dict(), daily, env)


@loop(
    (
        ("compare_posterior_mean_and_true_m_total.pdf", "plot_compare_m_and_true_m_total.jl"),
        ("compare_posterior_mean_and_true_m_rate.pdf", "plot_compare_m_and_true_m_rate.jl"),
        ("potency_cov.pdf", "plot_potency_cov.jl"),
    ),
    tform=lambda x: x,
)
def _(name, exe):
    filing(
        vxw,
        [(name, dict())],
        dict(
            exe=jp(vx_b, exe),
            src_params=env[kof("src_params.jld2", dict())],
            posterior_mean=env[kof("posterior_mean.jld2", dict())],
            true_m=env[kof("true_m.jld2", dict())],
            posterior_mean_cov=env[kof("posterior_mean_cov.jld2", dict())],
        ),
        copy_dict(
            conf,
            [
                "src__nt_total",
                "src__dt",
                "src__ix_hypo",
                "src__iy_hypo",
                "src__nx",
                "src__ny",
            ],
        ),
        env,
    )(make_run_with_temp(julia, vxt, dict()))
    all_figs.append(env[kof(name, dict())])
    set_daily(name, dict(), daily, env)


@filing(
    vxw,
    [
        ("all_figs.pdf", dict()),
        ("all_figs.txt", dict()),
    ],
    dict(data_list=all_figs),
    dict(),
    env,
)
def _(j, arg):
    for t in j.ts:
        mkdir(dirname(t))
    ds = sorted(j.ds, key=lambda path: tuple(reversed(os.path.split(path))))
    sh(f"pdfunite {' '.join(ds)} {j.ts[0]}")
    write_list(j.ts[1], ds)

    trash_dir = jp(os.path.expanduser("~"), "d", "trash", "_ssi", current_time.strftime("%Y-%m-%dT%H:%M:%S"))
    mkdir(trash_dir)
    for t in j.ts:
        sh(f"cp -f {t} {trash_dir}")
phony("all", ["daily", env[kof("all_figs.pdf", dict())], env[kof("all_figs.txt", dict())]])


if __name__ == "__main__":
    dsl.main(sys.argv)

# AsciiDoctor

```
:stem: latexmath

\(\alpha^{\beta_{\gamma}}\)
```

# `logging`

```py
def setup_logger():
    import logging
    import sys

    logger = logging.getLogger(__name__)
    hdl = logging.StreamHandler(sys.stderr)
    hdl.setFormatter(logging.Formatter("%(levelname)s\t%(asctime)s\t%(filename)s\t%(funcName)s\t%(lineno)d\t%(message)s"))
    logger.addHandler(hdl)
    logger.setLevel(logging.DEBUG)
    return logger


logger = setup_logger()
```

# `plain.jl`

```jl
#!/usr/bin/env julia


doc"""
@l[/path/to/link]
@i[/path/to/image]
"""
function main(args)
    write(STDOUT, "<html><pre>")
    is_at = false
    at_flag = '\0'
    is_link = false
    is_image = false
    path_buf = Vector{Char}()
    for line in eachline(STDIN, chomp=false)
        for c in line
            if is_link
                if c == ']'
                    is_link = false
                    write(STDOUT, "'>")
                    for x in path_buf
                        write(STDOUT, x)
                    end
                    write(STDOUT, "</a>")
                    empty!(path_buf)
                else
                    write(STDOUT, c)
                    push!(path_buf, c)
                end
            elseif is_image
                if c == ']'
                    is_image = false
                    write(STDOUT, "' />")
                else
                    write(STDOUT, c)
                end
            elseif at_flag == 'l'
                at_flag = '\0'
                if c == '['
                    is_link = true
                    write(STDOUT, "<a href='")
                else
                    write(STDOUT, '@')
                    write(STDOUT, 'l')
                    write(STDOUT, c)
                end
            elseif at_flag == 'i'
                at_flag = '\0'
                if c == '['
                    is_image = true
                    write(STDOUT, "<img width=50% max-width=70ex max-height=70ex src='")
                else
                    write(STDOUT, '@')
                    write(STDOUT, 'i')
                    write(STDOUT, c)
                end
            elseif is_at
                is_at = false
                if c == 'l' || c == 'i'
                    at_flag = c
                else
                    write(STDOUT, '@')
                    write(STDOUT, c)
                end
            elseif c == '&'
                write(STDOUT, "&amp")
            elseif c == '<'
                write(STDOUT, "&lt")
            elseif c == '>'
                write(STDOUT, "&gt")
            elseif c == '"'
                write(STDOUT, "&quot")
            elseif c == '\''
                write(STDOUT, "&#39")
            elseif c == '\t'
                write(STDOUT, "&#9")
            elseif c == '@'
                is_at = true
            else
                write(STDOUT, c)
            end
        end
    end
    write(STDOUT, "</pre></html>")
end


function _usage_and_exit(s=1)
    io = s == 0 ? STDOUT : STDERR
    println(io, "$PROGRAM_FILE")
    exit(s)
end


if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
```

# `rasterized=True`

```py
#!/usr/bin/python

import sys

import scipy as sp
import matplotlib.pyplot as plt


def f(x, y):
    r = x**2 + y**2
    return sp.sin(r)/r


def main(argv):
    fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=600)
    xs = sp.linspace(-5, 5, 70)
    ys = sp.linspace(-5, 5, 100)
    X, Y = sp.meshgrid(xs, ys)
    Z = f(X, Y)
    handle = ax1.pcolormesh(X, Y, Z, cmap="viridis", rasterized=True)
    handle = fig.colorbar(handle, aspect=30)
    handle.set_label("Goodness")
    ax1.set_xlabel(r"$X$")
    ax1.set_ylabel(r"$Y$")

    ax2.scatter(sp.randn(50), sp.randn(50), rasterized=True)
    ax2.set_xlabel(r"$X$")
    ax2.set_ylabel(r"$Y$")

    fig.savefig(sys.stdout.buffer, format="pdf", transparent=True)


if __name__ == '__main__':
    main(sys.argv)
```

# `tlmgr` packages

```bash
tlmgr list --only-installed | cut -f2 -d\  | sed -e 's/:$//g' | grep -v x86_64
```

```
accfonts
adobemapping
ae
afm2pl
aichej
ajl
algorithm2e
amscls
amsfonts
amsmath
amsrefs
aobs-tikz
apacite
apalike2
archaeologie
arphic
arphic-ttf
avantgar
babel
babel-english
babel-japanese
babelbib
baekmuk
beamer
beamertheme-metropolis
beebe
besjournals
bestpapers
bibarts
biber
bibexport
bibhtml
biblatex
biblatex-abnt
biblatex-anonymous
biblatex-apa
biblatex-archaeology
biblatex-arthistory-bonn
biblatex-bookinarticle
biblatex-bookinother
biblatex-bwl
biblatex-caspervector
biblatex-chem
biblatex-chicago
biblatex-claves
biblatex-dw
biblatex-enc
biblatex-fiwi
biblatex-gb7714-2015
biblatex-gost
biblatex-historian
biblatex-ieee
biblatex-ijsra
biblatex-iso690
biblatex-juradiss
biblatex-lni
biblatex-luh-ipw
biblatex-manuscripts-philology
biblatex-mla
biblatex-morenames
biblatex-multiple-dm
biblatex-musuos
biblatex-nature
biblatex-nejm
biblatex-nottsclassic
biblatex-opcit-booktitle
biblatex-oxref
biblatex-philosophy
biblatex-phys
biblatex-publist
biblatex-realauthor
biblatex-sbl
biblatex-science
biblatex-shortfields
biblatex-source-division
biblatex-subseries
biblatex-swiss-legal
biblatex-trad
biblatex-true-citepages-omit
biblist
bibtex
bibtexperllibs
bibtopic
bibtopicprefix
bibunits
biolett-bst
bookdb
bookman
booktabs
boondox
breakcites
breakurl
bxbase
bxcjkjatype
bxjalipsum
bxjaprnind
bxjscls
bxorigcapt
c90
caption
carlisle
cell
charter
chbibref
checkcites
chembst
chicago
chicago-annote
chickenize
chscite
citeall
cjk
cjk-gs-integrate
cjkpunct
cjkutils
cleveref
cloze
cm
cm-super
cmap
cmbright
cmextra
cns
collection-basic
collection-bibtexextra
collection-fontsrecommended
collection-fontutils
collection-langcjk
collection-langjapanese
collection-latex
collection-luatex
collref
colortbl
combofont
comment
compactbib
convbkmk
courier
crossrefware
cstypo
ctablestack
custom-bib
dejavu
din1505
dk-bib
dnp
doipubmed
dosepsbin
dvipdfmx
dvipng
dvips
dvipsconfig
dvisvgm
ec
ecobiblatex
economic
enctex
enigma
enumitem
epstopdf
eso-pic
etex
etex-pkg
etoolbox
euenc
euro
euro-ce
eurosym
everyhook
extsizes
fancyhdr
fancyvrb
fbs
figbib
filehook
fira
fix2col
fontaxes
fontinst
fontloader-luaotfload
fontools
fonts-tlwg
fontspec
fontware
footbib
fpl
francais-bst
garuda-c90
geometry
geschichtsfrkl
glyphlist
graphics
graphics-cfg
graphics-def
gsftopk
harvard
harvmac
helvetic
hf-tikz
historische-zeitschrift
hyperref
hyph-utf8
hyphen-base
ietfbibs
ifluatex
ifptex
iftex
ifxetex
ijqc
inlinebib
interpreter
iopart-num
ipaex
ipaex-type1
japanese-otf
japanese-otf-uptex
jfmutil
jknapltx
jlreq
jneurosci
jsclasses
jurabib
kastrup
knuth-lib
knuth-local
koma-script
kpathsea
ksfh_nat
l3kernel
l3packages
lastpage
latex
latex-bin
latex-fonts
latexconfig
latexdiff
latexmk
lcdftypetools
libertine
lineno
listbib
listings
lm
lm-math
logreq
lshort-japanese
ltb2bib
ltxmisc
lua-alt-getopt
lua-check-hyphen
lua-visual-debug
lua2dox
luabibentry
luabidi
luacode
luahyphenrules
luaindex
luainputenc
luaintro
lualatex-doc
lualatex-math
lualibs
luamplib
luaotfload
luapackageloader
luasseq
luatex
luatex85
luatexbase
luatexja
luatexko
luatextra
luatodonotes
luaxml
makeindex
manfnt-font
margbib
marvosym
mathastext
mathdots
mathpazo
mathspec
mathtools
memoir
mendex-doc
metafont
metatype1
mf2pt1
mflogo
mflogo-font
mfnfss
mfware
microtype
mptopdf
ms
multibib
multibibliography
munich
mweights
nag
nar
natbib
ncntrsbk
newtx
newtxsf
nmbib
nodetree
norasi-c90
notes2bib
notex-bst
numprint
oberdiek
odsfile
oscola
palatino
paralist
patchcmd
pbibtex-base
pdfjam
pdfpages
pdftex
perception
pgf
pgfopts
pgfplots
placeat
placeins
plain
platex
platex-tools
platexcheat
pnas2009
preprint
present
preview
ps2pk
pslatex
psnfss
pspicture
pstools
psutils
ptex
ptex-base
ptex-fontmaps
ptex-fonts
ptex2pdf
pxbase
pxchfon
pxcjkcat
pxfonts
pxjahyper
pxrubrica
pxtatescale
pxufont
relsize
rsc
rsfs
sansmath
scheme-basic
scheme-infraonly
scheme-minimal
selnolig
setspace
sfmath
showhyphens
showlabels
showtags
siunitx
sort-by-letters
spelling
splitbib
standalone
stix
sttools
subfigure
svn-prov
symbol
t1utils
tabulary
tetex
tex
tex-gyre
tex-gyre-math
tex-ini-files
tex4ht
texdoc
texlive-common
texlive-docindex
texlive-en
texlive-msg-translations
texlive-scripts
texlive.infra
texliveonfly
textpos
tikz-bayesnet
times
tipa
tocloft
tools
ttfutils
turabian-formatting
txfonts
type1cm
ucharcat
uhc
ulem
uni-wtal-ger
uni-wtal-lin
unicode-data
unicode-math
updmap-map
uplatex
uptex
uptex-base
uptex-fonts
url
urlbst
usebib
utopia
vak
wadalab
wasy
wasy2-ps
wasysym
wrapfig
xcite
xcjk2uni
xcolor
xdvi
xits
xkeyval
xstring
xunicode
zapfchan
zapfding
zxjafbfont
zxjafont
zxjatype
```

# `levenshtein_distance.py`

```py
#!/usr/bin/python

import sys


def main(argv):
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("abc", "ac") == 1
    assert levenshtein_distance("ac", "abc") == 1
    assert levenshtein_distance("abc", "akc") == 1
    assert levenshtein_distance("levenshtein distance", "lvnstn dstnc") == 8


def levenshtein_distance(a, b):
    a, b = list(a), list(b)
    m, n = len(a), len(b)
    row_prev = list(range(n + 1))
    row_now = [None for _ in range(n + 1)]
    for i in range(m):
        row_now[0] = i + 1
        ai = a[i]
        for j in range(n):
            row_now[j + 1] = min(
                row_now[j] + 1,
                row_prev[j + 1] + 1,
                row_prev[j] + (0 if ai == b[j] else 1)
            )
        row_prev, row_now = row_now, row_prev
    return row_prev[n]


if __name__ == '__main__':
    main(sys.argv)
```

# `get_PP.py`

```py
#!/usr/bin/python

import argparse
import sys

import numpy as np

import obspy.taup


def main(argv):
    parser = argparse.ArgumentParser(description='Get P-PP time (s)')
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {version}'.format(version=__version__),
    )
    parser.add_argument(
        '--source_depth_in_km',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--distance_in_degree_1',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--distance_in_degree_2',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--distance_n',
        type=int,
        required=True,
    )
    args = parser.parse_args(argv[1:])

    assert args.distance_in_degree_1 <= args.distance_in_degree_2
    assert args.distance_n > 0

    tpm = obspy.taup.TauPyModel(model='ak135')

    for d in np.linspace(args.distance_in_degree_1, args.distance_in_degree_2, args.distance_n):
        arrivals = tpm.get_travel_times(
            args.source_depth_in_km,
            distance_in_degree=d,
            phase_list=('P', 'PP'),
        )
        P_last = get_P_last(arrivals)
        PP_first = get_PP_first(arrivals)
        print(d, '\t', PP_first - P_last)


def get_PP_first(arrivals):
    ts = [ar.time for ar in arrivals if ar.name == 'PP']
    ts.sort()
    return ts[0]


def get_P_last(arrivals):
    ts = [ar.time for ar in arrivals if ar.name == 'P']
    ts.sort()
    return ts[-1]


def _usage_and_exit(s=1):
    if s == 0:
        fh = sys.stdout
    else:
        fh = sys.stderr
    print('{}'.format(__file__), file=fh)
    exit(s)


if __name__ == '__main__':
    main(sys.argv)
```

# `slide_template.tex`

```tex
% \PassOptionsToPackage{draft}{graphicx}
\PassOptionsToPackage{dvipsnames}{xcolor}
\documentclass{beamer}

\usetheme[numbering=fraction]{metropolis}
\usecolortheme{dove}
\usefonttheme{professionalfonts}
\setbeamercovered{transparent}
% \setbeameroption{show notes}

\usepackage{mathdots}
\usepackage{mathtools}

\usepackage{luatexja-fontspec}
% \setsansfont{TeX Gyre Heros}
% \setsansfont{DejaVu Sans}
\setsansfont{Fira Sans}
\setmainjfont{Noto Sans CJK JP}
\usepackage{newtxsf}
% http://www.feedbackward.com/content/nbs_latex.pdf
\DeclareTextCommand{\nobreakspace}{T1}{\leavevmode\nobreak\ }

\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{cleveref}
\usepackage{tikz}
\usetikzlibrary{overlay-beamer-styles}
\tikzset{invisible/.style={opacity=0.15}}

\usepackage[absolute, overlay]{textpos}
\setlength{\TPHorizModule}{0.01\paperwidth}
\setlength{\TPVertModule}{0.01\paperheight}

\usepackage{bm}
\usepackage{microtype}

\DeclarePairedDelimiter{\abs}{\lvert}{\rvert}
\DeclarePairedDelimiter{\norm}{\lVert}{\rVert}
\DeclarePairedDelimiter{\pr}{(}{)}

\title{Slide Template}
\date{2017}
\author{Me}
\institute{Institute}

\subject{}
% This is only inserted into the PDF information catalog. Can be left
% out.

%\pgfdeclareimage[height=0.4cm]{logo}{img/logo.pdf}
%\logo{\pgfuseimage{logo}}


% Delete this, if you do not want the table of contents to pop up at
% the beginning of each subsection:
\AtBeginSubsection[]
{
\begin{frame}<beamer>{Outline}
\tableofcontents[currentsection,currentsubsection]
\end{frame}
}


% If you wish to uncover everything in a step-wise fashion, uncomment
% the following command:
%\beamerdefaultoverlayspecification{<+->}


\begin{document}

\begin{frame}
\titlepage
\end{frame}

% \begin{frame}{Outline}
% \tableofcontents
% % You might wish to add the option [pausesections]
% \end{frame}

\begin{frame}
  \frametitle{Introduction}
  Hello $42$.
\end{frame}

\end{document}
```

# `mix.jl`

```julia
#!/usr/bin/env julia

using Distributions


const log2pi = log(2*pi)


function main(args)
    if length(args) != 1
        _usage_and_exit()
    end
    srand(48)

    xs, cs = load_data(args[1])

    m_lo = -10.0
    m_hi = 10.0
    m_l = m_hi - m_lo
    s2_lo = 0.0
    s2_hi = 36.0
    s2_l = s2_hi - s2_lo


    n_comp = 3

    ms = m_lo .+ (m_hi - m_lo).*rand(Float64, n_comp)
    ms_new = similar(ms)
    s2s = 0.1 .+ 2.*rand(Float64, n_comp)
    s2s_new = similar(s2s)
    n_xs = length(xs)
    zs = rand(1:n_comp, n_xs)
    zs_new = similar(zs)
    lls = Vector{Float64}(n_xs)
    lls_new = similar(lls)

    loglik!(lls, xs, ms, s2s, zs, m_lo, m_l)
    ll = sum(lls)

    mstep = 0.1
    s2step = 0.1

    n_iter = 20000
    output(STDOUT, 0, ll, ms, s2s, zs)
    for i_iter in 1:n_iter
        randn!(ms_new)
        ms_new .*= mstep
        ms_new .= cyclic.(ms_new .+ ms, m_lo, m_l)
        randn!(s2s_new)
        s2s_new .*= s2step
        s2s_new .= cyclic.(s2s_new .+ s2s, s2_lo, s2_l)

        loglik!(lls_new, xs, ms_new, s2s_new, zs, m_lo, m_l)
        ll_new = sum(lls_new)

        if exp(ll_new - ll) >= rand()
            ms .= ms_new
            s2s .= s2s_new
            lls .= lls_new
            ll = ll_new
        end

        for i in 1:n_xs
            z_new = rand(1:n_comp)
            ll_i = lls[i]
            ll_i_new = loglik(xs[i], ms[z_new], s2s[z_new], m_lo, m_l)
            if exp(ll_i_new - ll_i) >= rand()
                zs[i] = z_new
                lls[i] = ll_i_new
            end
        end
        ll = sum(lls)
        output(STDOUT, i_iter, ll, ms, s2s, zs)
    end
end


function load_data(path)
    xs = Vector{Float64}(0)
    cs = Vector{Int}(0)

    open(path) do io
        for (i, l) in enumerate(eachline(io))
            x, c = split(strip(l))
            push!(xs, parse(eltype(xs), String(x)))
            push!(cs, parse(eltype(cs), String(c)))
        end
    end
    xs, cs
end


function output(io::IO, i, ll, ms, s2s, zs)
    print(io, i, "\t", ll, "\t")
    for m in ms
        print(io, m, "\t")
    end
    for s2 in s2s
        print(io, s2, "\t")
    end
    for i in 1:(length(zs) - 1)
        print(io, zs[i], "\t")
    end
    println(io, zs[end])
end


function loglik!(lls, xs, ms, s2s, zs, m_lo, m_l)
    @assert length(lls) == length(xs) == length(zs) > 0

    for i in 1:length(lls)
        x, z = xs[i], zs[i]
        m = ms[z]
        s2 = s2s[z]
        lls[i] = loglik(x, m, s2, m_lo, m_l)
    end
    lls
end

function loglik(x, m, s2, m_lo, m_l)
    lognormal(cyclic(x - m, m_lo, m_l), s2)
end


function cyclic(x, lo, l)
    n = floor((x - lo)/l)
    x - n*l
end


function lognormal(x, s2)
    -((x^2)/s2 + log2pi + log(s2))/2
end


function _usage_and_exit(s=1)
    io = s == 0 ? STDOUT : STDERR
    println(io, "$PROGRAM_FILE <xc.tsv> > <out.tsv>")
    exit(s)
end


if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
```

# `get_active_app.sh`

```bash
#!/bin/bash

readonly hn="$(hostname)"
if [[ "$(uname -s)" = Darwin ]]; then
   # there seems to be a bug in AppKit which makes loop inside Python useless
   for _ in {1..12}
   do
      /usr/bin/python <<EOF
import datetime
from AppKit import NSWorkspace

hn = "$hn"

# http://apple.stackexchange.com/questions/123730/is-there-a-way-to-detect-what-program-is-stealing-focus-on-my-mac
# print datetime.datetime.now().strftime("%FT%T") + "\t" + str(NSWorkspace.sharedWorkspace().menuBarOwningApplication().bundleURL())
# bundleURL can be nil if the application does not have a bundle structure.
print hn + "\t" + datetime.datetime.now().strftime("%FT%T") + "\t" + str(NSWorkspace.sharedWorkspace().menuBarOwningApplication().executableURL())
EOF
      sleep 4.95
   done
else
   for _ in {1..12}
   do
      t="$(date +'%FT%T')"
      p="$(readlink -f /proc/"$(xdotool getwindowfocus getwindowpid)"/exe)"
      echo "$hn	$t	$p"
      sleep 4.95
   done
fi
```

# `cde`

```bash
cde(){
   local d="$("${MY_EMACSCLIENT:-emacsclient}" -e "
(expand-file-name
 (with-current-buffer
     (window-buffer (get-mru-window))
   default-directory))
" | sed -e 's/^"\(.*\)"$/\1/g')"
   if [[ -z "$d" ]]; then
      :
   else
      pushd "$d"
   fi
}
```

# `plot_active_app.py`

```py3
#!/usr/bin/python

import os
import random
import sys
import datetime

import matplotlib.pyplot as plt


c_of_app = dict(
    Emacs=(106/255, 90/255, 205/255),
    firefox=(229/255, 91/255, 10/255),
    Terminal=(0, 0, 0),
)


def main(argv):
    fig, ax = plt.subplots()
    ts = []
    raw_vs = []
    for t, v in _load(sys.stdin):
        ts.append(t)
        raw_vs.append(v)
    vs = [os.path.basename(v) for v in raw_vs]
    xs = [t.date() for t in ts]
    ys = [t.time().hour + t.time().minute/60 + t.time().second/3600 for t in ts]
    xyc_of_v = {}
    random.seed(44)
    for i, v in enumerate(vs):
        if v in xyc_of_v:
            xyc_of_v[v]["xs"].append(xs[i])
            xyc_of_v[v]["ys"].append(ys[i])
        else:
            if v in c_of_app:
                c = c_of_app[v]
            else:
                c = (random.random(), random.random(), random.random())
            xyc_of_v[v] = dict(
                xs=[xs[i]],
                ys=[ys[i]],
                c=c,
            )
    n_apps = len(xyc_of_v)
    for i, (v, xyc) in enumerate(sorted(xyc_of_v.items(), key=lambda vxyc: len(vxyc[1]["xs"]), reverse=True)):
        ax.scatter(
            xyc["xs"],
            xyc["ys"],
            c=(xyc["c"],),
            marker="_",
            linewidth=0.015,
            label=os.path.basename(v),
        )
        ax.text(
            0.01,
            (n_apps - 1 - i + 0.5)/n_apps,
            v,
            fontsize=100/n_apps,
            color=xyc["c"],
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Time of day (hour)")

    fig.savefig(
        sys.stdout.buffer,
        format="pdf",
        transparent=True,
        bbox_inches="tight",
    )


def _load(fh):
    for l in fh:
        t, v = l.strip().split("\t", 1)
        yield datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S"), v


if __name__ == '__main__':
    main(sys.argv)
```

# `reftex-uniquify-label`

```el
(with-eval-after-load 'reftex-ref
  ;; copied from reftex-ref.el and modified to use time instead of serial number
  (defun reftex-uniquify-label (label &optional force separator)
    ;; Make label unique by appending a number.
    ;; Optional FORCE means, force appending a number, even if label is unique.
    ;; Optional SEPARATOR is a string to stick between label and number.

    ;; Ensure access to scanning info
    (reftex-access-scan-info)

    (cond
     ((and (not force)
           (not (assoc label (symbol-value reftex-docstruct-symbol))))
      label)
     (t (let* ((ti (current-time))
               (s (+ (cadr ti) (* 65536 (car ti)))))
          (concat label
                  (or separator "")
                  (string-reverse (-base36hex (- s 1484800000)))
                  ;; (string-reverse (format "%x" (- s 1484800000)))
                  )))))
  )

(defun -base36hex (n)
  (if (< n 0) (error "n is negative"))
  (let ((r (cl-rem n 36)))
    (--base36hex (/ (- n r) 36)
             (list (---base36hex r)))))

(defun --base36hex (n s)
  (if (<= n 0)
      (apply 'concat s)
    (let ((r (cl-rem n 36)))
      (--base36hex (/ (- n r) 36)
               (cons (---base36hex r) s)))))

(defun ---base36hex (n)
  (nth n '("0" "1" "2" "3" "4" "5" "6" "7" "8"
           "9" "a" "b" "c" "d" "e" "f" "g" "h"
           "i" "j" "k" "l" "m" "n" "o" "p" "q"
           "r" "s" "t" "u" "v" "w" "x" "y" "z")))
```

# `plot_likelihood_1d.py`

```py3
#!/usr/bin/python


from math import sqrt, pi, exp
import os
import sys

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.font_manager


def main(argv):
    if len(argv) != 1:
        _usage_and_exit()

    font_name = next(f.fname for f in matplotlib.font_manager.fontManager.ttflist if os.path.basename(f.fname) == "FiraSans-Regular.otf")
    font_prop = matplotlib.font_manager.FontProperties(fname=font_name, size="large")

    d = np.array((2,))
    sw2 = 1
    se2 = 1/sqrt(2)

    Cb0 = np.eye(1, 1)
    Cb = se2*Cb0
    H = np.array(
        (
            (1,),
        )
    )
    Cw = np.array(
        (
            (sw2,),
        )
    )

    f, a = plt.subplots()

    x1 = -9
    x2 = 7.0
    a.axhline(y=0, linewidth=0.5, color="gray", zorder=-100)
    nx = 201
    xs = np.linspace(x1, x2, nx)
    likelihood_marg = make_marg_likelihood(d, H, Cb, Cw)
    ys = [likelihood_marg((x,)) for x in xs]
    a.plot(
        xs,
        ys,
        c='k',
        linewidth=2,
        zorder=3,
        label=r"$p(d \mid m)$",
    )

    likelihood_bare = make_likelihood(d, H, se2*Cb0)
    ys = [likelihood_bare((x,)) for x in xs]
    a.plot(
        xs,
        ys,
        "--",
        c="k",
        zorder=1,
        label=r"$p(d \mid m, G = H)$",
    )

    likelihood_approx = make_likelihood(d, H, (se2 + sw2*(d[0]/H[0, 0])**2)*Cb0)
    ys = [likelihood_approx((x,)) for x in xs]
    a.plot(
        xs,
        ys,
        c=(0.6, 0.6, 0.6),
        linewidth=1.5,
        zorder=2,
        label=r"$\approx p(d \mid m)$",
    )

    a.legend(
        loc="upper left",
        prop=font_prop,
    )

    a.set_xlabel(r'$m$', fontproperties=font_prop)
    a.set_ylabel('Likelihood', fontproperties=font_prop)
    for ti in a.get_xticklabels() + a.get_yticklabels():
        ti.set_fontproperties(font_prop)

    f.savefig(
        sys.stdout.buffer,
        format='pdf',
        transparent=True,
        bbox_inches='tight',
    )


def _usage_and_exit(s=1):
    print('{} > <pdf>'.format(__file__), file=sys.stderr)
    exit(s)


def make_likelihood(d, G, Cb):
    N, M = np.shape(G)
    assert N > 0
    assert M > 0
    assert np.shape(Cb) == (N, N)
    assert len(d) == N

    coeff = (2*pi)**(-N/2)/sqrt(np.linalg.det(Cb))

    def likelihood(m):
        d_neg_Gm = d - G@m
        return (
            coeff*
            exp(-(d_neg_Gm@(np.linalg.solve(Cb, d_neg_Gm)))/2)
        )
    return likelihood


def make_marg_likelihood(d, H, Cb, Cw):
    N = len(d)
    assert N > 0
    assert np.shape(Cb) == (N, N)
    assert np.shape(Cw) == (N**2, N**2)

    def likelihood(m):
        C = Cb + Cp_of(m, Cw)
        d_neg_Hm = d - H@m
        return (
            (2*pi)**(-N/2)/
            sqrt(np.linalg.det(C))*
            exp(-(d_neg_Hm@(np.linalg.solve(C, d_neg_Hm)))/2)
        )
    return likelihood


def Cp_of(m, Cw):
    M = len(m)
    assert M > 0
    MN = np.shape(Cw)[0]
    assert np.shape(Cw)[1] == MN
    N = MN//M
    assert M*N == MN

    Cp = np.empty((N, N))
    for j in range(N):
        Cp[j, j] = m@corr_of(j, j, N, M, Cw)@m
        for i in range(j, N):
            Cp[i, j] = m@corr_of(i, j, N, M, Cw)@m
            Cp[j, i] = Cp[i, j]
    return Cp


def corr_of(i, j, N, M, Cw):
    assert np.shape(Cw) == (N*M, N*M)
    assert 0 <= i < N
    assert 0 <= j < N

    return Cw[Cw_range_of(i, M), Cw_range_of(j, M)]


def Cw_range_of(i, M):
    return slice(i*M, (i + 1)*M)


if __name__ == '__main__':
    main(sys.argv)
```
