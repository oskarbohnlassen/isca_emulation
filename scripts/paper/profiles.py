"""Final two-variable profile figure, preserving the published layout and style."""
import json
from .common import MODEL_ORDER, LABELS, load_selected, mean_std
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

VARIABLES = ("temp", "ucomp")
REGIMES = ("no_ssw", "with_ssw")
LABEL_OVERRIDES = {"mesh_gnn2d": "Mesh MPNN", "gnn2d_gatedgcn": "Grid MPNN"}


def generate(context):
    global OUTPUT
    from .benchmark import generate as aggregate
    aggregate(context)
    OUTPUT = context.output / "test"
    manifest, groups = load_selected(context)
    with np.load(OUTPUT / "test_aggregate_arrays.npz", allow_pickle=False) as data:
        pressure = data["pressure"]
        with plt.rc_context({"font.size": 9, "axes.labelsize": 9,
                             "axes.titlesize": 10, "xtick.labelsize": 8,
                             "ytick.labelsize": 8, "legend.fontsize": 8.5}):
            fig, axes = plt.subplots(2, 2, figsize=(13 * 2 / 3, 5),
                                     sharex="col", sharey=True)
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for row, regime in enumerate(REGIMES):
                for col, variable in enumerate(VARIABLES):
                    ax = axes[row, col]
                    for model, label, color in zip(MODEL_ORDER, LABELS, colors):
                        label = LABEL_OVERRIDES.get(model, label)
                        mean = data[f"{regime}__{model}__profile_{variable}_mean"]
                        std = data[f"{regime}__{model}__profile_{variable}_std"]
                        check_mean, check_std = mean_std([
                            arrays[f"profile_{variable}"]
                            for _, _, arrays in groups[regime, model]
                        ])
                        np.testing.assert_array_equal(mean, check_mean)
                        np.testing.assert_array_equal(std, check_std)
                        ax.plot(mean, pressure, color=color, label=label, linewidth=1)
                        ax.fill_betweenx(pressure, np.maximum(0, mean - std), mean + std,
                                         color=color, alpha=0.28, linewidth=0)
                    baseline = groups[regime, "cnn2d"][0][2][f"persistence_profile_{variable}"]
                    ax.plot(baseline, pressure, "k--", label="Persistence", linewidth=1.3)
                    ax.set_yscale("log")
                    ax.grid(alpha=0.3)
                    ax.set_xlim(left=0)
                    ax.set_title(f"{'No SSW' if row == 0 else 'With SSW'}: "
                                 f"{'Temperature' if variable == 'temp' else 'Zonal wind'}")
                    assert len(ax.lines) == 11 and len(ax.collections) == 10
                axes[row, 0].set_ylabel("Pressure (hPa)")
            for col, variable in enumerate(VARIABLES):
                axes[1, col].set_xlabel("MAE (K)" if variable == "temp" else "MAE (m/s)")
            axes[0, 0].set_yticks([1000, 100, 10, 1, 0.1, 0.01])
            axes[0, 0].invert_yaxis()
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=4,
                       bbox_to_anchor=(0.5, 0.085), frameon=True)
            fig.tight_layout(rect=(0, 0.21, 1, 1), h_pad=0.7, w_pad=1.2)
            fig.canvas.draw()
            # Preserve the width and top edge, trimming only below the legend.
            renderer = fig.canvas.get_renderer()
            for artist in [fig.legends[0], *[ax.xaxis.label for ax in axes[-1]]]:
                bounds = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
                assert bounds.x0 >= 0 and bounds.x1 <= 1 and bounds.y0 >= 0 and bounds.y1 <= 1
            legend_bottom_inches = fig.legends[0].get_window_extent(renderer).y0 / fig.dpi
            crop_bottom = max(0, legend_bottom_inches - 0.08)
            width, height = fig.get_size_inches()
            export_bbox = Bbox.from_extents(0, crop_bottom, width, height)
            name = "test_mae_profiles_temp_ucomp_mean_std"
            for extension in ("pdf", "png"):
                fig.savefig(OUTPUT / f"{name}.{extension}", dpi=300,
                            bbox_inches=export_bbox, pad_inches=0)
            report = dict(
                selected_runs=len(manifest["runs"]), seeds_per_group=5,
                variables=VARIABLES, figsize_inches=fig.get_size_inches().tolist(),
                exported_size_inches=[float(width), float(height - crop_bottom)],
                bottom_trim_inches=float(crop_bottom), bottom_padding_inches=0.08,
                style="Original test_mae_profiles_mean_std tab10 colours, model order and lines; SD opacity 0.28",
                legend_anchor=[0.5, 0.085], legend_frame=True,
                fonts_points=dict(base=9, titles=10, axis_labels=9, ticks=8, legend=8.5),
                std_definition="Sample SD across five per-seed MAE profiles, ddof=1",
                data_checked="Exact equality with saved aggregate arrays and selected per-seed reductions",
                inference_required=False, selection=manifest["selection"],
            )
            (OUTPUT / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
            plt.close(fig)
    print(OUTPUT / f"{name}.pdf", flush=True)
