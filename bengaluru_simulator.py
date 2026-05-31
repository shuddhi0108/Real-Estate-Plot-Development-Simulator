"""
Bengaluru Real Estate Plot Layout Simulator
ADSA Final Project — Interactive Streamlit Presentation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import heapq

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru Plot Simulator",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
h1, h2, h3, .stMetric label { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
code, .mono { font-family: 'DM Mono', monospace; }

.stApp { background: #0d1117; color: #e6edf3; }

.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: #58a6ff; }
.metric-card .label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; margin-bottom: 6px; }
.metric-card .value { font-size: 28px; font-weight: 800; color: #e6edf3; }
.metric-card .sub { font-size: 12px; color: #57ab5a; margin-top: 4px; }
.metric-card .sub.neg { color: #f85149; }

.algo-badge {
    display: inline-block;
    background: #21262d;
    border: 1px solid #388bfd44;
    color: #58a6ff;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    margin: 2px;
}

.section-header {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #8b949e;
    font-weight: 600;
    margin: 24px 0 12px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
}

.stSlider > div { color: #e6edf3; }
.stSelectbox label, .stSlider label { color: #8b949e !important; font-size: 13px !important; text-transform: uppercase; letter-spacing: 1px; }

div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
}
div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 11px !important; }
div[data-testid="metric-container"] div[data-testid="metric-value"] { color: #e6edf3 !important; font-size: 22px !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
PLOT_SIZES       = [500, 1000, 1500]
SIZE_SIDE        = {500: 22, 1000: 32, 1500: 39}
GUTTER           = 5
MAX_WIDTH        = 250
SQFT_TO_M        = 0.3048
COLOR_MAP        = {500: '#57ab5a', 1000: '#388bfd', 1500: '#f0883e'}
LABEL_MAP        = {500: 'Small (500 sqft)', 1000: 'Medium (1000 sqft)', 1500: 'Large (1500 sqft)'}

# ── Embedded real data (medians from the Bengaluru dataset) ────────────────────
PRICE_PER_SQFT = {500: 5_285, 1000: 4_960, 1500: 4_730}   # Rs/sqft (from project notebook)
DEMAND_WEIGHTS = {500: 0.46,  1000: 0.33,  1500: 0.21}    # normalised demand proportions

# ── Core algorithms ────────────────────────────────────────────────────────────

def prim_mst_cost(points):
    """Prim's MST on Euclidean distances (O(n² log n))."""
    n = len(points)
    if n < 2:
        return 0.0
    in_mst   = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0.0
    total_cost  = 0.0
    heap = [(0.0, 0)]
    while heap:
        d, u = heapq.heappop(heap)
        if in_mst[u]:
            continue
        in_mst[u]   = True
        total_cost  += d
        for v in range(n):
            if not in_mst[v]:
                dx   = points[u][0] - points[v][0]
                dy   = points[u][1] - points[v][1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist < min_dist[v]:
                    min_dist[v] = dist
                    heapq.heappush(heap, (dist, v))
    return total_cost


def allocate_plots(total_land, open_space_pct, demand_weights, price_per_sqft):
    """Greedy score-based allocation (demand × price)."""
    buildable     = int(total_land * (1 - open_space_pct / 100))
    total_demand  = sum(demand_weights.values())
    scores = {
        s: price_per_sqft[s] * (demand_weights[s] / total_demand)
        for s in PLOT_SIZES
    }
    ranked     = sorted(PLOT_SIZES, key=lambda s: scores[s], reverse=True)
    allocation = {}
    remaining  = buildable
    for size in ranked:
        count             = remaining // size
        allocation[size]  = int(count)
        remaining        -= count * size
    return allocation, remaining, buildable


def build_layout(allocation):
    plots     = []
    x, y      = 0, 0
    row_max_h = 0
    for size in PLOT_SIZES:
        count = allocation.get(size, 0)
        if count == 0:
            continue
        w = h = SIZE_SIDE[size]
        for _ in range(count):
            if x + w > MAX_WIDTH:
                x         = 0
                y        += row_max_h + GUTTER
                row_max_h = 0
            plots.append({'x': x, 'y': y, 'w': w, 'h': h, 'size': size})
            x        += w + GUTTER
            row_max_h = max(row_max_h, h)
    return plots


def run_simulation(total_land, open_space_pct, cost_per_m_road, demand_weights, price_per_sqft):
    allocation, leftover, buildable = allocate_plots(
        total_land, open_space_pct, demand_weights, price_per_sqft
    )
    plots = build_layout(allocation)
    if not plots:
        return None

    centroids  = [(p['x'] + p['w'] / 2, p['y'] + p['h'] / 2) for p in plots]
    mst_feet   = prim_mst_cost(centroids)
    mst_metres = mst_feet * SQFT_TO_M
    road_cost  = mst_metres * cost_per_m_road

    total_revenue = sum(
        size * price_per_sqft[size] * count
        for size, count in allocation.items()
    )
    profit = total_revenue - road_cost
    roi    = (profit / total_revenue * 100) if total_revenue > 0 else 0

    return {
        'total_land'   : total_land,
        'buildable'    : buildable,
        'allocation'   : allocation,
        'leftover_sqft': leftover,
        'plots'        : plots,
        'num_plots'    : len(plots),
        'mst_metres'   : mst_metres,
        'revenue'      : total_revenue,
        'road_cost'    : road_cost,
        'profit'       : profit,
        'roi_pct'      : roi,
    }

# ── Plotting helpers ───────────────────────────────────────────────────────────

def fig_layout(result, show_mst=True):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    plots     = result['plots']
    centroids = [(p['x'] + p['w'] / 2, p['y'] + p['h'] / 2) for p in plots]

    for p in plots:
        color = COLOR_MAP.get(p['size'], '#8b949e')
        rect  = mpatches.FancyBboxPatch(
            (p['x'], p['y']), p['w'], p['h'],
            boxstyle="round,pad=0.5",
            facecolor=color, edgecolor='#0d1117', linewidth=0.8, alpha=0.88
        )
        ax.add_patch(rect)
        ax.text(p['x'] + p['w'] / 2, p['y'] + p['h'] / 2,
                str(p['size']), ha='center', va='center',
                fontsize=5.5, color='white', fontweight='bold')

    if show_mst and len(centroids) >= 2:
        import heapq as _hq
        n       = len(centroids)
        in_mst  = [False] * n
        min_e   = [(float('inf'), -1, 0)] * n
        min_e[0] = (0, -1, 0)
        heap    = [(0, 0, -1)]
        while heap:
            d, u, parent = _hq.heappop(heap)
            if in_mst[u]:
                continue
            in_mst[u] = True
            if parent >= 0:
                ax.plot([centroids[parent][0], centroids[u][0]],
                        [centroids[parent][1], centroids[u][1]],
                        color='#f0f6fc', linewidth=0.5, alpha=0.35, zorder=5)
            for v in range(n):
                if not in_mst[v]:
                    dx   = centroids[u][0] - centroids[v][0]
                    dy   = centroids[u][1] - centroids[v][1]
                    dist = (dx**2 + dy**2) ** 0.5
                    if dist < (min_e[v][0] if min_e[v][0] != float('inf') else float('inf')):
                        min_e[v] = (dist, u, v)
                        _hq.heappush(heap, (dist, v, u))

    all_x = [p['x'] + p['w'] for p in plots]
    all_y = [p['y'] + p['h'] for p in plots]
    ax.set_xlim(-8, max(all_x) + 12)
    ax.set_ylim(-8, max(all_y) + 12)
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.set_xlabel("feet →", color='#8b949e', fontsize=9)
    ax.set_ylabel("feet ↑", color='#8b949e', fontsize=9)

    handles = [mpatches.Patch(color=COLOR_MAP[s], label=LABEL_MAP[s])
               for s in PLOT_SIZES if result['allocation'].get(s, 0) > 0]
    leg = ax.legend(handles=handles, fontsize=8, loc='upper right',
                    facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')

    ax.set_title(
        f"{result['num_plots']} plots  ·  {result['leftover_sqft']:,} sqft unused",
        color='#e6edf3', fontsize=10, fontweight='bold', pad=10
    )
    plt.tight_layout()
    return fig


def fig_financials(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    labels  = [f"{r['total_land']:,}" for r in results]
    revenue = [r['revenue']   / 1e7 for r in results]
    costs   = [r['road_cost'] / 1e7 for r in results]
    profits = [r['profit']    / 1e7 for r in results]

    x     = np.arange(len(labels))
    width = 0.22

    ax.bar(x - width, revenue, width, label='Revenue',   color='#57ab5a', alpha=0.9, zorder=3)
    ax.bar(x,         costs,   width, label='Road Cost', color='#f85149', alpha=0.9, zorder=3)
    ax.bar(x + width, profits, width, label='Profit',    color='#388bfd', alpha=0.9, zorder=3)

    for i, r in enumerate(results):
        ax.text(x[i] + width, max(profits[i], 0) + 0.04,
                f"ROI {r['roi_pct']:.1f}%",
                ha='center', va='bottom', fontsize=8,
                color='#79c0ff', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l} sqft" for l in labels], color='#e6edf3')
    ax.set_ylabel("Rs Crore", color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"Rs{v:.1f}Cr"))
    ax.grid(axis='y', color='#21262d', linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    leg = ax.legend(fontsize=9, facecolor='#21262d',
                    edgecolor='#30363d', labelcolor='#e6edf3')
    ax.set_title("Financial Comparison", color='#e6edf3', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def fig_allocation_pie(result):
    sizes  = [result['allocation'].get(s, 0) for s in PLOT_SIZES]
    labels = [f"{LABEL_MAP[s]}\n({result['allocation'].get(s, 0)} plots)"
              for s in PLOT_SIZES]
    colors = [COLOR_MAP[s] for s in PLOT_SIZES]

    filtered = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if not filtered:
        return None
    s_vals, l_vals, c_vals = zip(*filtered)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    wedges, texts, autotexts = ax.pie(
        s_vals, labels=l_vals, colors=c_vals,
        autopct='%1.1f%%', startangle=140,
        wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2}
    )
    for t in texts:
        t.set_color('#8b949e'); t.set_fontsize(8)
    for at in autotexts:
        at.set_color('white'); at.set_fontsize(8); at.set_fontweight('bold')
    ax.set_title("Plot Mix", color='#e6edf3', fontsize=10, fontweight='bold')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    st.markdown('<div class="section-header">Land & Infrastructure</div>', unsafe_allow_html=True)
    total_land      = st.slider("Total Land Area (sqft)", 5_000, 30_000, 10_000, step=500)
    open_space_pct  = st.slider("Open Space Setback (%)", 10, 40, 30, step=5)
    cost_per_m_road = st.slider("Road Construction Cost (Rs/m)", 500, 5_000, 2_000, step=100)

    st.markdown('<div class="section-header">Market Demand Weights</div>', unsafe_allow_html=True)
    d_small  = st.slider("Small (500 sqft) demand",  0.05, 0.90, 0.46, step=0.01)
    d_medium = st.slider("Medium (1000 sqft) demand", 0.05, 0.90, 0.33, step=0.01)
    d_large  = st.slider("Large (1500 sqft) demand",  0.05, 0.90, 0.21, step=0.01)

    st.markdown('<div class="section-header">Price Override (Rs/sqft)</div>', unsafe_allow_html=True)
    p_small  = st.number_input("Small plot price/sqft",  value=5285, step=100)
    p_medium = st.number_input("Medium plot price/sqft", value=4960, step=100)
    p_large  = st.number_input("Large plot price/sqft",  value=4730, step=100)

    show_mst = st.checkbox("Show MST Road Network", value=True)

    st.markdown("---")
    st.markdown("**Algorithms used:**")
    st.markdown('<span class="algo-badge">Prim\'s MST</span>', unsafe_allow_html=True)
    st.markdown('<span class="algo-badge">Greedy Allocation</span>', unsafe_allow_html=True)
    st.markdown('<span class="algo-badge">IQR Outlier Filter</span>', unsafe_allow_html=True)

# ── Assemble user parameters ───────────────────────────────────────────────────
user_demand = {500: d_small, 1000: d_medium, 1500: d_large}
user_price  = {500: p_small, 1000: p_medium, 1500: p_large}

# Normalise demand
total_d     = sum(user_demand.values())
user_demand = {k: v / total_d for k, v in user_demand.items()}

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding: 8px 0 24px'>
  <div style='font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:3px;margin-bottom:6px'>ADSA Final Project</div>
  <h1 style='font-size:36px;font-weight:800;color:#e6edf3;margin:0'>🏘️ Bengaluru Plot Layout Simulator</h1>
  <p style='color:#8b949e;font-size:14px;margin-top:8px'>Real-estate subdivision optimizer · Prim's MST road planning · Greedy demand-based allocation</p>
</div>
""", unsafe_allow_html=True)

# ── Run single simulation ──────────────────────────────────────────────────────
result = run_simulation(total_land, open_space_pct, cost_per_m_road, user_demand, user_price)

if result is None:
    st.error("⚠️ Not enough land to generate any plots. Increase total land or reduce open space %.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Plots", result['num_plots'])
with k2:
    st.metric("Buildable Area", f"{result['buildable']:,} sqft")
with k3:
    st.metric("Road Length (MST)", f"{result['mst_metres']:,.0f} m")
with k4:
    st.metric("Revenue", f"Rs {result['revenue']/1e7:.2f} Cr")
with k5:
    roi_delta = f"+{result['roi_pct']:.1f}%" if result['roi_pct'] > 0 else f"{result['roi_pct']:.1f}%"
    st.metric("Profit / ROI", f"Rs {result['profit']/1e7:.2f} Cr", delta=roi_delta)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT + PIE
# ══════════════════════════════════════════════════════════════════════════════
col_layout, col_pie = st.columns([3, 1])

with col_layout:
    st.markdown('<div class="section-header">Plot Layout with MST Road Network</div>', unsafe_allow_html=True)
    st.pyplot(fig_layout(result, show_mst=show_mst))

with col_pie:
    st.markdown('<div class="section-header">Plot Mix</div>', unsafe_allow_html=True)
    pie = fig_allocation_pie(result)
    if pie:
        st.pyplot(pie)

    st.markdown('<div class="section-header">Allocation Breakdown</div>', unsafe_allow_html=True)
    for size in PLOT_SIZES:
        count = result['allocation'].get(size, 0)
        rev   = size * user_price[size] * count / 1e7
        col_a, col_b = st.columns(2)
        col_a.markdown(f"<span style='color:{COLOR_MAP[size]};font-weight:700'>{size} sqft</span>", unsafe_allow_html=True)
        col_b.markdown(f"**{count}** plots · Rs {rev:.2f}Cr")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-SCENARIO COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Multi-Scenario Financial Comparison</div>', unsafe_allow_html=True)
st.caption("Comparing your current land size against neighbouring scenarios")

comparison_lands = sorted(set([
    max(5000, total_land - 3000),
    total_land,
    total_land + 3000
]))
comp_results = [
    r for land in comparison_lands
    if (r := run_simulation(land, open_space_pct, cost_per_m_road, user_demand, user_price)) is not None
]

if len(comp_results) >= 2:
    st.pyplot(fig_financials(comp_results))

    # Summary table
    summary_df = pd.DataFrame([{
        'Land (sqft)'       : f"{r['total_land']:,}",
        'Total Plots'       : r['num_plots'],
        'Road (m)'          : f"{r['mst_metres']:,.0f}",
        'Revenue (Rs Cr)'   : f"{r['revenue']/1e7:.2f}",
        'Road Cost (Rs Cr)' : f"{r['road_cost']/1e7:.2f}",
        'Profit (Rs Cr)'    : f"{r['profit']/1e7:.2f}",
        'ROI %'             : f"{r['roi_pct']:.1f}%",
    } for r in comp_results])

    st.dataframe(
        summary_df.style.set_properties(**{
            'background-color': '#161b22',
            'color'           : '#e6edf3',
            'border-color'    : '#30363d',
        }),
        use_container_width=True,
        hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📐 How the Algorithms Work"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. Data Cleaning")
        st.markdown("""
- Loaded **Bengaluru House Data** CSV
- Converted range sqft (e.g. `1000-1500`) → midpoint
- Removed nulls & zero-value rows
- Applied **IQR outlier filter** on `price_per_sqft`
- Categorised into Small / Medium / Large bins
""")
    with c2:
        st.markdown("#### 2. Greedy Allocation")
        st.markdown("""
- Reserves **open space** (setback %)
- Scores each plot type by `price × demand_share`
- Ranks types by score, fills greedily in order
- Leftover land is reported as unused
""")
    with c3:
        st.markdown("#### 3. Prim's MST (Road Cost)")
        st.markdown("""
- Computes **centroid** of every plot
- Runs Prim's algorithm on Euclidean distances
- MST = minimum total road needed to connect all plots
- Road cost = MST length × cost per metre
""")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:12px'>ADSA Final Project · Bengaluru Real-Estate Plot Simulator · Data: Kaggle Bengaluru House Data</div>",
    unsafe_allow_html=True
)