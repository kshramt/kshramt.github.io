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
