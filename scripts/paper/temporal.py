"""Normalized temporal MAE across the ten retained seed-42 models, with observed wind."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from .common import dump, read_manifest

REGIMES = ("no_ssw", "with_ssw")
VARIABLES = ("temp", "ucomp", "vcomp")
TITLES = ("Temperature", "Zonal wind", "Meridional wind")
SPLITS = ("train", "val", "test")
COLORS = {"train": "#2878b5", "val": "#e58a21", "test": "#32965b"}
SPLIT_LABELS = {"train": "Training", "val": "Validation", "test": "Test"}
MODEL_LABELS = {"cnn2d": "Simple CNN2D", "cnn2d_unet": "CNN2D U-Net",
                "cnn3d": "Simple CNN3D", "cnn3d_unet": "CNN3D U-Net",
                "gnn2d_gatedgcn": "Grid MPNN", "mesh_gnn2d": "Mesh MPNN",
                "transformer2d_global_attention": "Transformer 2D GA",
                "transformer2d_swin": "Transformer 2D Swin",
                "transformer3d_global_attention": "Transformer 3D GA",
                "transformer3d_swin": "Transformer 3D Swin"}
WIND_COLOR = "#733c91"


def generate(context):
    selected = read_manifest("selected_checkpoints.json")
    retained = [r for r in selected["runs"] if r["seed"] == 42]
    assert len(retained) == 20
    OUTPUT = context.output / "temporal"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected = np.concatenate([np.arange(1001,3000),np.arange(3001,3250),np.arange(3251,3500)])
    expected_split = np.repeat(SPLITS,[1999,249,249])
    series,winds,stamps,arrays,provenance = {},{},{},{},[]
    for regime in REGIMES:
        rows = [r for r in retained if r["regime"] == regime]
        assert len(rows) == 10 and {r["model"] for r in rows} == set(MODEL_LABELS)
        normalized = []
        for row in rows:
            folder = f"results/paper/inputs/temporal/{regime}/{row['model']}"
            report = context.json(folder+"/report.json")
            for key in ("run_id","checkpoint_digest","config_sha256","test_data_name"):
                assert row[key] == report["run"][key]
            frame = pd.read_csv(context.input(folder+"/errors_by_time.csv"))
            np.testing.assert_array_equal(frame.target_source_index,expected)
            np.testing.assert_array_equal(frame.split,expected_split)
            raw = frame[[f"mae_{v}" for v in VARIABLES]].to_numpy(float).T
            assert np.isfinite(raw).all() and (raw>=0).all()
            denominators = raw.mean(axis=1)
            assert (denominators>0).all()
            values = raw/denominators[:,None]
            np.testing.assert_allclose(values.mean(axis=1),1,rtol=0,atol=1e-14)
            normalized.append(values)
            provenance.append(dict(regime=regime,model=row['model'],seed=42,run_id=row['run_id'],
                checkpoint_digest=row['checkpoint_digest'],denominators=denominators.tolist()))
        raw_stack = np.stack(normalized)
        assert raw_stack.shape == (10,3,2497)
        smooth_stack = np.empty_like(raw_stack)
        for split in SPLITS:
            mask = expected_split == split
            for model in range(10):
                for v in range(3):
                    smooth_stack[model,v,mask] = pd.Series(raw_stack[model,v,mask]).rolling(
                        16,center=True,min_periods=1).mean().to_numpy()
        mean,std = smooth_stack.mean(axis=0),smooth_stack.std(axis=0,ddof=1)
        series[regime] = (mean,std)
        stamps[regime] = (expected,expected_split)
        wind_folder = f"results/paper/inputs/temporal/{regime}"
        winds[regime] = pd.read_csv(context.input(wind_folder+"/observed_u10_60N.csv"))
        grid = context.json(wind_folder+"/wind_overlay_report.json")
        assert grid['actual_pressure_hpa']==10 and abs(grid['actual_latitude_deg']-60)<0.01
        np.testing.assert_array_equal(winds[regime].source_index,np.arange(1000,3500))
        assert np.isfinite(winds[regime].u10_60N_ms).all()
        arrays.update({f"{regime}__normalized_raw":raw_stack,f"{regime}__normalized_smoothed":smooth_stack,
            f"{regime}__mean_smoothed":mean,f"{regime}__sample_std_smoothed":std,
            f"{regime}__mean_raw":raw_stack.mean(axis=0),f"{regime}__sample_std_raw":raw_stack.std(axis=0,ddof=1),
            f"{regime}__model_order":np.asarray([r['model'] for r in rows]),
            f"{regime}__observed_u10_60N_ms":winds[regime].u10_60N_ms.to_numpy()})
    with np.load(context.input('results/paper/inputs/temporal/reference/normalized_curves.npz')) as prior:
        for key,value in arrays.items():
            np.testing.assert_array_equal(value,prior[key])
    arrays.update(target_source_index=expected,split=expected_split,
        observed_source_index=np.arange(1000,3500),variables=np.asarray(VARIABLES))
    np.savez_compressed(OUTPUT/'normalized_curves.npz',**arrays)
    upper = np.ceil(max(float((mean+std).max()) for mean, std in series.values())*4)/4
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12,
                         "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig, axes = plt.subplots(3, 2, figsize=(13, 8.5), sharex=True, sharey=True)
        for col, regime in enumerate(REGIMES):
            x, split_tags = stamps[regime]
            mean, std = series[regime]
            for v, title in enumerate(TITLES):
                ax = axes[v, col]
                for i, split in enumerate(SPLITS):
                    mask = split_tags == split
                    ax.axvspan((1000, 3000, 3250)[i], (3000, 3250, 3500)[i],
                               color=COLORS[split], alpha=0.035, linewidth=0)
                    ax.fill_between(x[mask], mean[v, mask]-std[v, mask], mean[v, mask]+std[v, mask],
                                    color=COLORS[split], alpha=0.22, linewidth=0)
                    ax.plot(x[mask], mean[v, mask], color=COLORS[split], linewidth=1.6)
                for boundary in (3000, 3250):
                    ax.axvline(boundary, color="0.4", linestyle=":", linewidth=1)
                ax.axhline(1, color="0.35", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.set_ylim(0, upper)
                ax.set_xlim(1000, 3499)
                ax.grid(alpha=0.15)
                if col == 0:
                    ax.set_ylabel(f"{title}\nNormalized MAE")
                right = ax.twinx()
                # Draw observed wind behind the MAE and its between-model band.
                right.set_zorder(ax.get_zorder()-1)
                ax.patch.set_visible(False)
                right.plot(winds[regime].source_index, winds[regime].u10_60N_ms,
                           color=WIND_COLOR, linestyle="--", linewidth=1.1, alpha=0.75)
                right.axhline(0, color=WIND_COLOR, linestyle=":", linewidth=0.7, alpha=0.3)
                right.set_ylim(-15, 105)
                right.set_ylabel("Observed zonal wind\n60°N, 10 hPa (m/s)", color=WIND_COLOR, fontsize=9)
                right.tick_params(axis="y", colors=WIND_COLOR, labelsize=9)
                right.spines["right"].set_color(WIND_COLOR)
            axes[0, col].set_title("No SSW" if regime == "no_ssw" else "SSW-enabled")
            axes[-1, col].set_xlabel("Original simulation timestep")
        handles = [Line2D([], [], color=COLORS[s], linewidth=1.6, label=SPLIT_LABELS[s]) for s in SPLITS]
        handles += [Patch(facecolor="0.5", alpha=0.22, label="±1 SD across models"),
                    Line2D([], [], color=WIND_COLOR, linestyle="--", label="Observed wind (right axes)")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True,
                   bbox_to_anchor=(0.5, 0.015), fontsize=9)
        fig.suptitle("Normalized temporal MAE across ten architectures, seed 42", fontsize=13, y=0.98)
        fig.subplots_adjust(left=0.075, right=0.92, bottom=0.125, top=0.91, hspace=0.12, wspace=0.20)
        fig.canvas.draw()
        for extension in ("pdf", "png"):
            fig.savefig(OUTPUT / f"temporal_normalized_mae_wind_overlay.{extension}", dpi=240, bbox_inches="tight")
        plt.close(fig)
    dump(OUTPUT/'report.json',dict(seeds=[42],models_per_regime=10,runs=provenance,
        normalization='Each run/variable divided by its raw full-period mean over 2497 targets',
        smoothing='16-step centered rolling mean separately inside each split, min_periods=1',
        band='Sample SD across architectures at seed 42, ddof=1; not five-seed variability',
        wind='Unsmoothed observed zonal-mean wind, nearest 60N and 10hPa',
        boundaries=[3000,3250],retrospective_final_checkpoint=True,observed_inputs=True,
        inference_required=False,exact_match_to_published_curves=True))
    print('TEMPORAL',OUTPUT,flush=True)
