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

# `with subplots(**kwargs) as (fig, axs):`

```py
class subplots:

    def __init__(self, **kwargs):
        import matplotlib.pyplot
        self.fig, self.axs = matplotlib.pyplot.subplots(**kwargs)

    def __enter__(self):
        return self.fig, self.axs

    def __exit__(self, typ, val, traceback):
        import matplotlib.pyplot
        matplotlib.pyplot.close(self.fig)
```

# `with rcparams("k", v)`

```py
class rcparams:
    def __init__(self, *args):
        assert len(args)%2 == 0
        self.params = {args[2*i + 0]: args[2*i + 1] for i in range(len(args)//2)}

    def __enter__(self):
        import matplotlib
        self.rcparams_orig = list(matplotlib.rcParams.items())
        for k, v in self.params.items():
            matplotlib.rcParams[k] = v

    def __exit__(self, typ, val, traceback):
        import matplotlib
        matplotlib.rcParams = matplotlib.RcParams()
        for k, v in self.rcparams_orig:
            matplotlib.rcParams[k] = v
```

# `tlmgr` packages

```
adobemapping
algorithm2e
aobs-tikz
authblk
beamer
beamertheme-metropolis
biber
biblatex
bibtex2html
binhex
bm
booktabs
boondox
breakurl
caption
cjkpunct
cleveref
cmap
cmbright
collection-fontsrecommended
comment
ctablestack
dejavu
dot2tex
ec
enumitem
eso-pic
etoolbox
everyhook
everysel
everyshi
expl3
extarticle
extsizes
fancyvrb
filehook
fira
fontaxes
fontspec
hf-tikz
ipaex-type1
jknapltx
jknappen
jsclasses
kastrup
koma-script
l3kernel
l3packages
lastpage
latexdiff
latexmk
libertine
lineno
listings
logreq
lualatex-math
luaotfload
luatex85
luatexbase
luatexja
mathastext
mathdots
mathrsfs
mathspec
mathtools
memoir
microtype
ms
mweights
nag
newtx
newtxsf
noto
numprint
paralist
patchcmd
patchdmd
pdfjam
pdfpages
pgf
pgfopts
pgfplots
placeins
preprint
present
preview
relsize
resize
rsfs
sansmath
setspace
sfmath
showkeys
showlabels
siunitx
standalone
stfloats
stix
sttools
subfigure
svn-prov
tabulary
tex-gyre
tex-gyre-math
tex2html
tex4ht
texdoc
texliveonfly
texpos
textpos
tikz
tikz-bayesnet
times
tocloft
tools
txfonts
type1cm
ucharcat
ulem
unicode-math
uptex-base
wrapfig
xcolor
xits
xkeyval
xparse
xstring
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
