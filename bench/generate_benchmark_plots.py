"""
High-Resolution Benchmark Plot Generator for Residual-Spec LLM Offloading
Outputs crisp, publication-grade dark-themed graphics for GitHub and Reddit (r/LocalLLaMA).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# Output directory in assets
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Theme configuration: Elegant Dark Theme matching GitHub Dark & Modern ML Papers
STYLE = {
    "bg_dark": "#0d1117",         # Canvas background
    "bg_card": "#161b22",         # Plot area background
    "grid_color": "#21262d",      # Subtle gridlines
    "text_main": "#f0f6fc",       # High-contrast headers
    "text_muted": "#8b949e",      # Subtitles, axis tick labels
    "accent_emerald": "#10b981",  # Residual Q8 (Optimal)
    "accent_cyan": "#06b6d4",     # Residual Q6
    "accent_amber": "#f59e0b",    # Q4_K_M Baselines
    "accent_purple": "#8b5cf6",   # Google MTP / DFlash
    "accent_red": "#ef4444",      # Pure IQ2_M (Quality collapsed)
    "accent_slate": "#6e7681",    # References & Secondary
    "font_family": "sans-serif",
}

plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "figure.facecolor": STYLE["bg_dark"],
    "axes.facecolor": STYLE["bg_card"],
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": STYLE["text_main"],
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.color": STYLE["text_muted"],
    "ytick.color": STYLE["text_muted"],
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "grid.color": STYLE["grid_color"],
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "legend.facecolor": STYLE["bg_card"],
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 9.5,
    "text.color": STYLE["text_main"],
})


def plot_architecture_speedup_comparison():
    """Figure 1: Comprehensive Horizontal Bar Chart comparing Speed, VRAM, and Quality"""
    fig, ax = plt.subplots(figsize=(14.5, 8.5), dpi=300)
    
    models = [
        {"name": "Residual Spec Q8 (Tree D=10)", "speed": 23.5, "vram": 4.20, "quality": "MMLU: 36.7% | PPL: 1024", "color": STYLE["accent_emerald"], "badge": "[OPTIMAL] 2.42x Speedup (Full Quality)"},
        {"name": "Residual Spec Q8 (Tree D=4)",  "speed": 21.7, "vram": 4.20, "quality": "MMLU: 36.7% | PPL: 1024", "color": "#34d399", "badge": "2.24x Speedup"},
        {"name": "Q4_K_M + Google MTP (n=4)",    "speed": 17.7, "vram": 5.64, "quality": "MMLU: 32.4% | PPL: 1232", "color": STYLE["accent_purple"], "badge": "1.82x Speedup"},
        {"name": "Q4_K_M + z-lab DFlash (n=4)",  "speed": 17.5, "vram": 5.97, "quality": "MMLU: 32.4% | PPL: 1232", "color": "#a78bfa", "badge": "1.80x (Near VRAM limit)"},
        {"name": "Residual Spec Q6 (Linear D=2)","speed": 15.8, "vram": 4.20, "quality": "MMLU: 36.0% | PPL: 1048", "color": STYLE["accent_cyan"], "badge": "1.63x Speedup"},
        {"name": "Base Q4_K_M (Sweet spot ngl 37)","speed": 13.9, "vram": 5.82, "quality": "MMLU: 32.4% | PPL: 1232", "color": STYLE["accent_amber"], "badge": "1.43x (Max GPU Offload)"},
        {"name": "Base Q4_K_M (Equal ngl 32)",   "speed": 10.6, "vram": 5.14, "quality": "MMLU: 32.4% | PPL: 1232", "color": "#d97706", "badge": "1.09x"},
        {"name": "Base Q4_K_M (Conserv. ngl 30)","speed": 9.7,  "vram": 4.85, "quality": "MMLU: 32.4% | PPL: 1232", "color": "#b45309", "badge": "1.00x Baseline"},
        {"name": "Pure IQ2_M (Extreme Speed)",   "speed": 34.0, "vram": 4.76, "quality": "MMLU: 29.4% | PPL: 14381", "color": STYLE["accent_red"], "badge": "[ALERT] Quality Collapsed (PPL 14k)"},
    ]
    
    y_pos = np.arange(len(models))
    speeds = [m["speed"] for m in models]
    colors = [m["color"] for m in models]
    
    # Background reference vertical lines for Q4 baselines
    ax.axvline(x=9.7, color="#d97706", linestyle=":", alpha=0.55, linewidth=1.3, zorder=1)
    ax.axvline(x=13.9, color=STYLE["accent_amber"], linestyle="--", alpha=0.65, linewidth=1.5, zorder=1)
    
    # Top tags for reference lines to avoid collision with bars
    ax.text(9.7, -0.65, "Q4 Conserv.\n(9.7 t/s)", color="#d97706", fontsize=8.5, ha="center", va="bottom", weight="bold")
    ax.text(13.9, -0.65, "Q4 Sweet Spot\n(13.9 t/s)", color=STYLE["accent_amber"], fontsize=8.5, ha="center", va="bottom", weight="bold")
    
    bars = ax.barh(y_pos, speeds, color=colors, height=0.60, edgecolor="#30363d", linewidth=1.2, zorder=3)
    
    # Annotations on bars
    for i, (b, m) in enumerate(zip(bars, models)):
        # Speed text
        speed_txt = f"{m['speed']:.1f} tok/s"
        ax.text(m['speed'] + 0.6, i, speed_txt, va='center', ha='left', fontsize=10.5, weight="bold", color=STYLE["text_main"], zorder=5)
        
        # Details badge
        details = f"{m['badge']}   |   {m['quality']}   |   VRAM: {m['vram']:.2f} GB"
        if m['speed'] >= 21.0:
            # Place white/dark text inside bar near start
            text_col = "#091016" if "OPTIMAL" in m["badge"] or "IQ2" in m["name"] else "#ffffff"
            ax.text(0.8, i, details, va='center', ha='left', fontsize=9.0, color=text_col, weight="bold", zorder=5)
        else:
            # Place outside the bar to the right of speed label
            ax.text(m['speed'] + 6.0, i, details, va='center', ha='left', fontsize=8.6, color=STYLE["text_muted"], zorder=5)
            
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m["name"] for m in models], fontsize=10.5, weight="bold")
    ax.invert_yaxis()  # Best on top
    
    ax.set_xlim(0, 48)
    ax.set_ylim(len(models) - 0.2, -1.1)
    ax.set_xlabel("Generation Throughput (Tokens per Second) — Higher is Better", fontsize=11.5, labelpad=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(axis='x', zorder=0)
    
    # Title & Subtitle
    plt.title("Gemma 4 12B Generation Speed & Quality Tradeoff on 6GB VRAM Laptop", 
              fontsize=14.5, weight="bold", color=STYLE["text_main"], pad=28, loc="left")
    fig.text(0.125, 0.945, "AMD Ryzen 5 5600H (DDR4-3200) + RTX 3060 Laptop (GA106 6GB)  •  llama.cpp feature/residual-spec", 
             fontsize=9.8, color=STYLE["text_muted"])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "architecture_speedup_comparison.png")
    plt.savefig(output_path, dpi=300, facecolor=STYLE["bg_dark"], edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_speed_vs_quality_pareto():
    """Figure 2: Pareto Frontier Scatter Plot (Speed vs MMLU-Pro Accuracy and Speed vs Wikitext-2 PPL)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 7.8), dpi=300)
    
    points = [
        {"name": "Residual Spec Q8 (Tree D=10)", "speed": 23.5, "mmlu": 36.71, "ppl": 1024.17, "vram": 4.20, "color": STYLE["accent_emerald"], "marker": "o", "size": 300, "cat": "Residual-Spec"},
        {"name": "Residual Spec Q8 (Tree D=4)",  "speed": 21.7, "mmlu": 36.71, "ppl": 1024.17, "vram": 4.20, "color": "#34d399", "marker": "o", "size": 180, "cat": "Residual-Spec"},
        {"name": "Residual Spec Q6 (Linear D=2)","speed": 15.8, "mmlu": 36.00, "ppl": 1048.46, "vram": 4.20, "color": STYLE["accent_cyan"], "marker": "s", "size": 170, "cat": "Residual-Spec"},
        {"name": "Base Q4_K_M (Sweet spot ngl 37)","speed": 13.9, "mmlu": 32.43, "ppl": 1232.45, "vram": 5.82, "color": STYLE["accent_amber"], "marker": "D", "size": 200, "cat": "Standard Q4"},
        {"name": "Base Q4_K_M (Conserv. ngl 30)","speed": 9.7,  "mmlu": 32.43, "ppl": 1232.45, "vram": 4.85, "color": "#d97706", "marker": "D", "size": 160, "cat": "Standard Q4"},
        {"name": "Q4 + Google MTP (n=4)",        "speed": 17.7, "mmlu": 32.43, "ppl": 1232.45, "vram": 5.64, "color": STYLE["accent_purple"], "marker": "^", "size": 190, "cat": "Speculative Q4"},
        {"name": "Q4 + z-lab DFlash (n=4)",      "speed": 17.5, "mmlu": 32.43, "ppl": 1232.45, "vram": 5.97, "color": "#a78bfa", "marker": "v", "size": 190, "cat": "Speculative Q4"},
        {"name": "Pure Base b2 (No residual)",   "speed": 34.0, "mmlu": 25.14, "ppl": 6613.54, "vram": 4.20, "color": "#f87171", "marker": "P", "size": 170, "cat": "2-bit Draft Only"},
        {"name": "Pure IQ2_M (Extreme Speed)",   "speed": 34.0, "mmlu": 29.43, "ppl": 14381.02, "vram": 4.76, "color": STYLE["accent_red"], "marker": "X", "size": 220, "cat": "2-bit llama.cpp"},
    ]
    
    # --- PANEL 1: Speed vs MMLU-Pro Accuracy ---
    for pt in points:
        ax1.scatter(pt["speed"], pt["mmlu"], color=pt["color"], marker=pt["marker"], s=pt["size"],
                    edgecolors="#ffffff", linewidths=1.2, zorder=4, alpha=0.95)
        
    ax1.annotate("Residual Spec Q8 (Tree D=10)\n23.5 tok/s, 36.71% MMLU\n(4.2 GB VRAM - 1.9 GB Free!)",
                 xy=(23.5, 36.71), xytext=(17.5, 39.2),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_emerald"], lw=1.5),
                 fontsize=9.2, weight="bold", color=STYLE["accent_emerald"],
                 bbox=dict(boxstyle="round,pad=0.4", fc=STYLE["bg_card"], ec=STYLE["accent_emerald"], lw=1.2))
    
    ax1.annotate("Residual Spec Q6 (D=2)\n15.8 tok/s, 36.00%",
                 xy=(15.8, 36.00), xytext=(9.8, 38.0),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_cyan"], lw=1.2),
                 fontsize=8.5, color=STYLE["accent_cyan"])
    
    ax1.annotate("Q4 Baseline\n13.9 tok/s, 32.43%",
                 xy=(13.9, 32.43), xytext=(7.5, 34.0),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_amber"], lw=1.2),
                 fontsize=8.5, color=STYLE["accent_amber"])
    
    ax1.annotate("Google MTP (n=4)\n17.7 tok/s",
                 xy=(17.7, 32.43), xytext=(19.0, 30.8),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_purple"], lw=1.0),
                 fontsize=8.2, color=STYLE["accent_purple"])
    
    ax1.annotate("[ALERT] Pure IQ2_M\n34 tok/s but severe\nknowledge degradation",
                 xy=(34.0, 29.43), xytext=(26.0, 26.5),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_red"], lw=1.2),
                 fontsize=8.5, color=STYLE["accent_red"])

    # Draw Pareto boundary highlighting Residual Spec dominance
    ax1.plot([9.7, 13.9, 15.8, 21.7, 23.5], [32.43, 32.43, 36.00, 36.71, 36.71],
             color=STYLE["accent_emerald"], linestyle="--", linewidth=1.5, alpha=0.6, zorder=2)
    ax1.fill_between([9.7, 13.9, 15.8, 21.7, 23.5], 32.43, [32.43, 32.43, 36.00, 36.71, 36.71],
                     color=STYLE["accent_emerald"], alpha=0.10, label="Residual Spec Superiority Area")
    
    ax1.set_xlim(5, 37)
    ax1.set_ylim(22, 42.5)
    ax1.set_xlabel("Generation Speed (tok/s) — Higher is Better", fontsize=11, labelpad=8)
    ax1.set_ylabel("MMLU-Pro Accuracy (%) — Higher is Better", fontsize=11, labelpad=8)
    ax1.set_title("A) Reasoning Quality (MMLU-Pro 700 Qs) vs Generation Speed", fontsize=12, pad=12)
    ax1.grid(True, zorder=0)

    # Legend for Panel 1
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Residual Spec (Q8/Q6)', markerfacecolor=STYLE["accent_emerald"], markersize=9),
        Line2D([0], [0], marker='D', color='w', label='Standard Q4_K_M Offload', markerfacecolor=STYLE["accent_amber"], markersize=8),
        Line2D([0], [0], marker='^', color='w', label='Speculative Drafter (MTP/DFlash)', markerfacecolor=STYLE["accent_purple"], markersize=8),
        Line2D([0], [0], marker='X', color='w', label='Pure 2-bit (Degraded)', markerfacecolor=STYLE["accent_red"], markersize=8),
    ]
    ax1.legend(handles=legend_elements, loc="lower left", framealpha=0.9, fontsize=8.8)

    # --- PANEL 2: Speed vs Wikitext-2 PPL (Log Scale) ---
    for pt in points:
        ax2.scatter(pt["speed"], pt["ppl"], color=pt["color"], marker=pt["marker"], s=pt["size"],
                    edgecolors="#ffffff", linewidths=1.2, zorder=4, alpha=0.95)
        
    # Unquantized BF16 reference horizontal line
    ax2.axhline(y=596.23, color=STYLE["text_muted"], linestyle=":", linewidth=1.4, alpha=0.7, zorder=2)
    ax2.text(6.0, 630, "Unquantized BF16 Baseline (PPL 596.23, requires 24GB VRAM)", color=STYLE["text_muted"], fontsize=8.2, style="italic")
    
    # Connect Pareto frontier in PPL (Lower is better!)
    ax2.plot([13.9, 15.8, 21.7, 23.5], [1232.45, 1048.46, 1024.17, 1024.17],
             color=STYLE["accent_emerald"], linestyle="--", linewidth=1.5, alpha=0.6, zorder=2)
    
    # Annotations on Panel 2
    ax2.annotate("Residual Q8 (1024.17 PPL)\nLower PPL than Q4!",
                 xy=(23.5, 1024.17), xytext=(17.0, 1850),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_emerald"], lw=1.5),
                 fontsize=9.0, weight="bold", color=STYLE["accent_emerald"],
                 bbox=dict(boxstyle="round,pad=0.3", fc=STYLE["bg_card"], ec=STYLE["accent_emerald"], lw=1.2))
    
    ax2.annotate("Residual Q6 (1048.46 PPL)",
                 xy=(15.8, 1048.46), xytext=(12.0, 750),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_cyan"], lw=1.1),
                 fontsize=8.3, color=STYLE["accent_cyan"])
    
    ax2.annotate("Q4_K_M (1232.45 PPL)",
                 xy=(13.9, 1232.45), xytext=(7.0, 2300),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_amber"], lw=1.2),
                 fontsize=8.5, color=STYLE["accent_amber"])
    
    ax2.annotate("IQ2_M: 14381 PPL\nCatastrophic perplexity loss",
                 xy=(34.0, 14381.02), xytext=(22.0, 8000),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_red"], lw=1.2),
                 fontsize=8.5, color=STYLE["accent_red"])

    ax2.set_yscale("log")
    ax2.set_xlim(5, 37)
    ax2.set_ylim(450, 25000)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_yticks([500, 1000, 2000, 5000, 10000, 20000])
    ax2.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax2.set_xlabel("Generation Speed (tok/s) — Higher is Better", fontsize=11, labelpad=8)
    ax2.set_ylabel("Wikitext-2 Perplexity (PPL) — LOWER is Better (Log Scale)", fontsize=11, labelpad=8)
    ax2.set_title("B) Language Modeling Perplexity (10h Matrix) vs Generation Speed", fontsize=12, pad=12)
    ax2.grid(True, which="both", zorder=0)

    fig.suptitle("Residual Speculative Decoding: Pareto Frontier on 6GB VRAM (Gemma 4 12B)",
                 fontsize=14.5, weight="bold", color=STYLE["text_main"], y=0.98)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "speed_vs_quality_pareto.png")
    plt.savefig(output_path, dpi=300, facecolor=STYLE["bg_dark"], edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_draft_depth_sweep():
    """Figure 3: Draft Depth Sweep Curve (Linear Chain vs Branching Tree W=2)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=300)
    
    depths = np.array([1, 2, 3, 4, 6, 8, 10, 12, 16, 20])
    
    # Linear chain data
    linear_speed = np.array([14.56, 15.80, 15.70, 14.94, 12.83, 10.84, 9.26, 8.12, 6.36, 5.21])
    linear_tau   = np.array([0.82,  1.45,  1.91,  2.22,  2.54,  2.65,  2.67, 2.68, 2.68, 2.68])
    linear_rate  = np.array([82.0,  72.6,  63.7,  55.5,  42.4,  33.1,  26.7, 22.3, 16.7, 13.4])
    
    # Tree draft data (Width W=2)
    tree_speed   = np.array([15.74, 18.67, 20.45, 21.70, 22.99, 23.47, 23.49, 23.24, 22.31, 21.11])
    tree_tau     = np.array([0.97,  1.92,  2.84,  3.74,  5.45,  7.04,  8.51,  9.86,  12.20, 14.11])
    tree_rate    = np.array([96.8,  95.8,  94.7,  93.4,  90.8,  88.0,  85.1,  82.1,  76.2,  70.6])
    
    # --- PANEL 1: Generation Speed vs Depth D ---
    ax1.plot(depths, tree_speed, color=STYLE["accent_emerald"], linewidth=2.6, marker="o", markersize=7,
             label="Branching Tree Draft (W=2)", zorder=4)
    ax1.plot(depths, linear_speed, color="#f97316", linewidth=2.2, marker="s", markersize=6,
             linestyle="--", label="Linear Chain Draft (W=1)", zorder=3)
    
    # Reference lines for baseline Q4
    ax1.axhline(y=13.9, color=STYLE["accent_amber"], linestyle=":", linewidth=1.5, alpha=0.8,
                label="Q4 Sweet Spot Baseline (13.9 tok/s)")
    ax1.axhline(y=9.7, color="#d97706", linestyle="-.", linewidth=1.2, alpha=0.7,
                label="Q4 Conservative Baseline (9.7 tok/s)")
    
    # Peak annotations
    ax1.annotate("Tree Peak: 23.5 tok/s\n(Optimal Depth D=8..10)",
                 xy=(10, 23.49), xytext=(11.5, 24.2),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_emerald"], lw=1.5),
                 fontsize=9.0, weight="bold", color=STYLE["accent_emerald"],
                 bbox=dict(boxstyle="round,pad=0.3", fc=STYLE["bg_card"], ec=STYLE["accent_emerald"], lw=1.2))
    
    ax1.annotate("Linear Peak: 15.8 tok/s (D=2)\nDecays rapidly at D>4",
                 xy=(2, 15.80), xytext=(3.5, 17.5),
                 arrowprops=dict(arrowstyle="->", color="#f97316", lw=1.2),
                 fontsize=8.5, color="#f97316")
    
    # Fill between Tree and Linear
    ax1.fill_between(depths, linear_speed, tree_speed, color=STYLE["accent_emerald"], alpha=0.12,
                     label="Branching Benefit (+49% at D=10)")
    
    ax1.set_xlabel("Draft Depth D (Tokens per Step)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Effective Generation Speed (tok/s)", fontsize=11, labelpad=8)
    ax1.set_title("A) Generation Speed vs Draft Depth (D)", fontsize=12, pad=12)
    ax1.set_xlim(0.5, 21)
    ax1.set_ylim(4, 26)
    ax1.set_xticks([1, 2, 4, 6, 8, 10, 12, 16, 20])
    ax1.grid(True, zorder=0)
    ax1.legend(loc="lower left", framealpha=0.9)
    
    # --- PANEL 2: Accepted Tokens (tau) & Acceptance Rate (%) ---
    color_tau = STYLE["accent_cyan"]
    color_rate = "#ec4899"
    
    ax2.plot(depths, tree_tau, color=color_tau, linewidth=2.4, marker="o", markersize=7,
             label="Tree Accepted Tokens (tau)", zorder=4)
    ax2.plot(depths, linear_tau, color=color_tau, linewidth=1.8, marker="s", markersize=5,
             linestyle="--", label="Linear Accepted Tokens (tau)", zorder=3)
    ax2.set_xlabel("Draft Depth D (Tokens per Step)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Average Accepted Tokens tau per Verifier Pass", color=color_tau, fontsize=11, labelpad=8)
    ax2.tick_params(axis='y', labelcolor=color_tau)
    ax2.set_ylim(0, 16)
    ax2.set_xticks([1, 2, 4, 6, 8, 10, 12, 16, 20])
    
    # Secondary Y axis for Acceptance Rate %
    ax2_twin = ax2.twinx()
    ax2_twin.plot(depths, tree_rate, color=color_rate, linewidth=2.2, marker="^", markersize=6,
                  label="Tree Acceptance Rate (%)", zorder=4)
    ax2_twin.plot(depths, linear_rate, color=color_rate, linewidth=1.8, marker="v", markersize=5,
                  linestyle=":", label="Linear Acceptance Rate (%)", zorder=3)
    ax2_twin.set_ylabel("Verification Acceptance Rate (%)", color=color_rate, fontsize=11, labelpad=8)
    ax2_twin.tick_params(axis='y', labelcolor=color_rate)
    ax2_twin.set_ylim(0, 105)
    ax2_twin.grid(False)
    
    ax2.set_title("B) Accepted Tokens (tau) & Acceptance Rate (%) vs Depth", fontsize=12, pad=12)
    ax2.grid(True, zorder=0)
    
    # Combined legend for ax2
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center left", framealpha=0.9, fontsize=8.8)
    
    fig.suptitle("Linear Chain vs Branching Tree Speculation Mechanics (AVX2 Residual Verifier)",
                 fontsize=14.5, weight="bold", color=STYLE["text_main"], y=0.98)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "draft_depth_sweep.png")
    plt.savefig(output_path, dpi=300, facecolor=STYLE["bg_dark"], edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_avx2_kernel_throughput():
    """Figure 4: AVX2 Micro-Kernel Roofline & Throughput (Bandwidth GB/s & GFLOPS on Zen 3)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=300)
    
    batch_n = np.array([1, 2, 4, 8, 16])
    
    # Bandwidth data (GB/s)
    bw_q8_mlp = np.array([27.3, 26.5, 25.4, 24.0, 22.5])
    bw_q8_att = np.array([15.3, 14.8, 13.4, 12.1, 11.2])
    bw_q6_att = np.array([2.3,  2.4,  2.6,  2.7,  2.8])
    
    # GFLOPS compute throughput
    gflops_q8_mlp = np.array([48.6, 95.0, 215.1, 340.0, 420.0])
    gflops_q8_att = np.array([27.2, 52.8, 113.0, 185.0, 230.0])
    gflops_q6_att = np.array([4.1,  8.2,  18.0,  34.0,  45.0])
    
    # --- PANEL 1: Memory Bandwidth Saturation (GB/s) ---
    ax1.plot(batch_n, bw_q8_mlp, color=STYLE["accent_emerald"], linewidth=2.5, marker="o", markersize=7,
             label="Q8 Residual MLP Kernel (Streaming AVX2)", zorder=4)
    ax1.plot(batch_n, bw_q8_att, color=STYLE["accent_cyan"], linewidth=2.2, marker="s", markersize=6,
             label="Q8 Residual Attention Kernel", zorder=3)
    ax1.plot(batch_n, bw_q6_att, color=STYLE["accent_red"], linewidth=2.2, marker="^", markersize=6,
             linestyle="--", label="Q6 Residual Attention (ALU Stalled, No VBMI)", zorder=3)
    
    # Hardware DDR4 practical peak line
    ax1.axhline(y=34.17, color="#e2e8f0", linestyle=":", linewidth=1.5, alpha=0.8,
                label="DDR4-3200 Measured Streaming Ceiling (34.17 GB/s)")
    
    ax1.annotate("80% of Physical DDR4 Bandwidth\nSaturated by Q8 AVX2 (int8_t)",
                 xy=(1, 27.3), xytext=(2.2, 29.5),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_emerald"], lw=1.2),
                 fontsize=8.8, color=STYLE["accent_emerald"], weight="bold")
    
    ax1.annotate("5x Slowdown in Q6:\nScalar bit shifts stall CPU ports\n(Zen 3 lacks AVX-512 VBMI)",
                 xy=(4, 2.6), xytext=(4.5, 7.5),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_red"], lw=1.2),
                 fontsize=8.5, color=STYLE["accent_red"])

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(batch_n)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xlabel("Verification Token Batch Size N", fontsize=11, labelpad=8)
    ax1.set_ylabel("Effective DRAM Bandwidth (GB/s)", fontsize=11, labelpad=8)
    ax1.set_title("A) Memory Bandwidth Saturation (DDR4-3200)", fontsize=12, pad=12)
    ax1.set_xlim(0.8, 20)
    ax1.set_ylim(0, 38)
    ax1.grid(True, which="both", zorder=0)
    ax1.legend(loc="center right", framealpha=0.9, fontsize=8.8)

    # --- PANEL 2: Compute Density (GFLOPS) & Verification Latency ---
    ax2.plot(batch_n, gflops_q8_mlp, color=STYLE["accent_emerald"], linewidth=2.5, marker="o", markersize=7,
             label="Q8 MLP Compute (GFLOPS)", zorder=4)
    ax2.plot(batch_n, gflops_q8_att, color=STYLE["accent_cyan"], linewidth=2.2, marker="s", markersize=6,
             label="Q8 Attention Compute (GFLOPS)", zorder=3)
    ax2.plot(batch_n, gflops_q6_att, color=STYLE["accent_red"], linewidth=1.8, marker="^", markersize=5,
             linestyle="--", label="Q6 Attention Compute (GFLOPS)", zorder=2)
    
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(batch_n)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xlabel("Verification Token Batch Size N", fontsize=11, labelpad=8)
    ax2.set_ylabel("Arithmetic Throughput (GFLOPS) — Higher is Better", fontsize=11, labelpad=8)
    ax2.set_title("B) Arithmetic Intensity Amortization on AVX2", fontsize=12, pad=12)
    ax2.set_xlim(0.8, 20)
    ax2.set_ylim(0, 460)
    ax2.grid(True, which="both", zorder=0)
    
    # Annotation explaining amortization
    ax2.annotate("Memory-Wall Amortization:\nN=4 tokens verified simultaneously\nachieves 215 GFLOPS at 260 us/token",
                 xy=(4, 215.1), xytext=(2.0, 310),
                 arrowprops=dict(arrowstyle="->", color=STYLE["accent_emerald"], lw=1.2),
                 fontsize=8.8, color=STYLE["accent_emerald"], weight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc=STYLE["bg_card"], ec=STYLE["accent_emerald"], lw=1.1))

    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.8)

    fig.suptitle("AVX2 Residual Kernel Performance & Bottleneck Analysis (AMD Zen 3)",
                 fontsize=14.5, weight="bold", color=STYLE["text_main"], y=0.98)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "avx2_kernel_throughput.png")
    plt.savefig(output_path, dpi=300, facecolor=STYLE["bg_dark"], edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Generating high-resolution benchmark plots...")
    plot_architecture_speedup_comparison()
    plot_speed_vs_quality_pareto()
    plot_draft_depth_sweep()
    plot_avx2_kernel_throughput()
    print("All plots generated successfully!")
