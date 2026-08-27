import io, base64
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


GREEN_PALETTE = [
    "#2d6a4f", "#40916c", "#52b788", "#74c69d",
    "#95d5b2", "#b7e4c7", "#d8f3dc", "#1b4332",
    "#081c15", "#a8dabd",
]

MULTI_PALETTE = [
    "#e63946", "#f4a261", "#2a9d8f", "#457b9d",
    "#6a4c93", "#f77f00", "#118ab2", "#06d6a0",
    "#ff6b6b", "#a8dadc",
]


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    buf.seek(0)
    svg = buf.read().decode("utf-8")
    return "data:image/svg+xml;base64," + base64.b64encode(
        svg.encode("utf-8")
    ).decode("utf-8")


def render_bar_chart(district_counts: dict) -> str:
    """Horizontal bar chart — FSIs per district."""
    labels = list(district_counts.keys())
    values = list(district_counts.values())

    fig, ax = plt.subplots(figsize=(5.5, max(3, len(labels) * 0.35)))
    bars = ax.barh(labels, values, color=GREEN_PALETTE[0],
                   height=0.6, edgecolor="none")

    # value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left",
                fontsize=8, color="#444")

    ax.set_xlabel("Number of FSIs", fontsize=9, color="#555")
    ax.tick_params(axis="both", labelsize=8, colors="#555")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, color="#e8f5e9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_b64(fig)


def render_donut_chart(type_counts: dict) -> str:
    """Donut chart — FSI type distribution."""
    import math
    labels = list(type_counts.keys())
    values = list(type_counts.values())
    colors = MULTI_PALETTE[:len(labels)]
    total  = sum(values)
    pcts   = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    wedges, _ = ax.pie(
        values, colors=colors,
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5},
    )

    # percentage labels inside each wedge
    for wedge, pct in zip(wedges, pcts):
        angle = (wedge.theta1 + wedge.theta2) / 2
        r = 0.72
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        ax.text(x, y, f"{pct:.0f}%", ha="center", va="center",
                fontsize=5.5, fontweight="bold", color="white")

    # centre label
    ax.text(0, 0, str(total), ha="center", va="center",
            fontsize=16, fontweight="bold", color="#2d6a4f")
    ax.text(0, -0.22, "total", ha="center", va="center",
            fontsize=8, color="#888")

    # legend with count only
    legend = [
        mpatches.Patch(color=c, label=f"{l} ({v})")
        for c, l, v in zip(colors, labels, values)
    ]
    ax.legend(handles=legend, loc="lower center",
              bbox_to_anchor=(0.5, -0.28),
              ncol=2, fontsize=7.5, frameon=False)

    fig.tight_layout()
    return _fig_to_b64(fig)