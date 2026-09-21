import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D

from Baselines.get_baseline import get_baseline

def _shade_color(rgb, factor):
    """Tint (factor > 0, toward white) or shade (factor < 0, toward black) an RGB color."""
    r, g, b = rgb
    if factor >= 0:
        r, g, b = (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)
    else:
        r, g, b = (r * (1 + factor), g * (1 + factor), b * (1 + factor))
    return (min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0))

def get_experiment_colors(experiments, exp_names=None):
    """
    Maps each experiment name to a color derived from its baseline's color, spreading
    experiments that share the same underlying baseline into distinguishable shades
    (darker to brighter) instead of reusing the exact same color for all of them.
    """
    if exp_names is None:
        exp_names = list(experiments.keys())

    groups = {}
    baseline_by_exp = {}
    for exp_name in exp_names:
        baseline_by_exp[exp_name] = get_baseline(experiments[exp_name].module)
        groups.setdefault(experiments[exp_name].module, []).append(exp_name)

    exp_colors = {}
    for module, names in groups.items():
        base_rgb = mcolors.to_rgb(baseline_by_exp[names[0]].color)
        if len(names) == 1:
            exp_colors[names[0]] = base_rgb
            continue
        shades = np.linspace(-0.4, 0.4, len(names))
        for exp_name, factor in zip(names, shades):
            exp_colors[exp_name] = _shade_color(base_rgb, factor)

    return exp_colors

# Shared "professional" plot style: muted axes/gridlines, dark-gray text, so every
# figure in plot_functions.py can look consistent. See plot_cum_error for the reference use.
AXIS_GRAY = '#888888'
GRID_GRAY = '#dcdcdc'
TEXT_GRAY = '#333333'

def style_axis(ax, xlabel=None, ylabel=None, title=None, label_fontsize=10, title_fontsize=11):
    """Apply the shared axis look: recessive hairline grid/spines, muted gray text."""
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='-', linewidth=0.8, color=GRID_GRAY, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(AXIS_GRAY)
    ax.spines['bottom'].set_color(AXIS_GRAY)
    ax.tick_params(colors=AXIS_GRAY, labelsize=9)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize, color=TEXT_GRAY)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize, color=TEXT_GRAY)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, color=TEXT_GRAY, pad=8)

def get_legend_handles(exp_colors, exp_names, linewidth=2.5):
    """Line-style legend handles matching the line marks used by style_axis plots."""
    return [Line2D([0], [0], color=exp_colors[experiment_name], linewidth=linewidth, label=experiment_name)
            for experiment_name in exp_names]

def style_figure(fig, title, legend_handles, legend_ncol=None, title_fontsize=14):
    """Bold, muted-gray suptitle plus a frameless legend centered below the figure."""
    fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', color=TEXT_GRAY)
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=legend_ncol if legend_ncol is not None else min(len(legend_handles), 6),
               frameon=False, fontsize=10)
