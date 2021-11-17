import seaborn as sb
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot, setp
from numpy import percentile, linspace

color = sb.color_palette()[0]
color_rgb = mpl.colors.colorConverter.to_rgb(color)
colors = [sb.utils.set_hls_values(color_rgb, l=l) for l in linspace(1, 0, 12)]
cmap = sb.blend_palette(colors, as_cmap=True)

def _jplot(xs, y, xlabels=None, ylabel='', figsize=None, nb=30, gs=25, **kwargs):
    nx = len(xs)
    figsize = figsize or (13, 13/nx)
    fig = figure(figsize=figsize)
    gs_ct = GridSpec(2, nx + 1, bottom=0.2, top=1, left=0.1, right=1, hspace=0.05, wspace=0.05,
                     height_ratios=[0.15, 0.85], width_ratios=nx*[1] + [0.2])

    ylim = percentile(y, [0.5, 99.5])
    yper = percentile(y, [50, 75, 95])
    axs_j = []
    axs_m = []
    for i, x in enumerate(xs):
        xlim = percentile(x, [1, 99])
        aj = subplot(gs_ct[1, i])
        am = subplot(gs_ct[0, i])
        aj.hexbin(x, y, gridsize=gs, cmap=cmap, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
        # for yp in yper:
        #    aj.axhline(yp, lw=1, c='k', alpha=0.2)
        am.hist(x, bins=nb, alpha=0.5, range=xlim)
        setp(aj, xlim=xlim, ylim=ylim)
        setp(am, xlim=aj.get_xlim())
        setp(am, xticks=[], yticks=[])

        if i > 0:
            setp(aj.get_yticklabels(), visible=False)
        else:
            setp(aj, ylabel=ylabel)
        if xlabels is not None:
            setp(aj, xlabel=xlabels[i])
        axs_j.append(aj)
        axs_m.append(am)

    am = subplot(gs_ct[1, -1])
    am.hist(y, bins=nb, alpha=0.5, range=ylim, orientation='horizontal')
    setp(am, ylim=ylim, xticks=[])
    setp(am.get_yticklabels(), visible=False)

    [sb.despine(ax=ax, left=True, offset=0.1) for ax in axs_m]
    [sb.despine(ax=ax) for ax in axs_j]
    sb.despine(ax=am, bottom=True)
    return fig