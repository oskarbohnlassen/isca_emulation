"""Final physical-unit TEST metrics and five-seed profile/map reductions."""
import numpy as np
import pandas as pd

from .common import MODEL_ORDER, REGIMES, VARIABLES, dump, load_selected, mean_std
from . import plotting as pu
import matplotlib.pyplot as plt


def generate(context):
    manifest, groups = load_selected(context)
    output = context.output / "test"
    output.mkdir(parents=True, exist_ok=True)
    reference = next(iter(groups.values()))[0][2]
    arrays = {key: reference[key] for key in ("pressure", "latitude", "longitude")}
    raw, summary, baseline = [], [], []
    for regime in REGIMES:
        for model in MODEL_ORDER:
            rows = groups[regime, model]
            for row, info, arr in rows:
                for key in ("pressure", "latitude", "longitude"):
                    np.testing.assert_array_equal(arr[key], reference[key])
                raw.append(dict(regime=regime, model=model, seed=row["seed"], run_id=row["run_id"],
                    wandb_url=row["wandb_url"], **info["table"][row["model_type"]],
                    **{f"scaled_{k}": v for k, v in info["metrics"]["scaled"].items()},
                    inference_seconds=info["inference_seconds"]))
            record = dict(regime=regime, model=model, n=5, seeds=",".join(str(r[0]["seed"]) for r in rows))
            for metric in rows[0][1]["table"][rows[0][0]["model_type"]]:
                mean, std = mean_std([info["table"][row["model_type"]][metric] for row, info, _ in rows])
                record[metric + "_mean"], record[metric + "_std"] = float(mean), float(std)
            summary.append(record)
            for v in VARIABLES:
                for kind in ("profile", "map"):
                    mean, std = mean_std([a[f"{kind}_{v}"] for _, _, a in rows])
                    arrays[f"{regime}__{model}__{kind}_{v}_mean"] = mean
                    arrays[f"{regime}__{model}__{kind}_{v}_std"] = std
        base = groups[regime, "cnn2d"][0][1]["table"]["Persistence"]
        baseline.append(dict(regime=regime, model="Persistence", **base))
    pd.DataFrame(raw).to_csv(output / "test_metrics_per_seed.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "test_table_mean_std.csv", index=False)
    pd.DataFrame(baseline).to_csv(output / "test_persistence_table.csv", index=False)
    np.savez_compressed(output / "test_aggregate_arrays.npz", **arrays)
    # Exact equality with the published compact numerical figure inputs.
    with np.load(context.input("results/paper/inputs/benchmark/test_aggregate_arrays.npz")) as prior:
        assert set(arrays) == set(prior.files)
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, prior[key])
    old = pd.read_csv(context.input("results/paper/inputs/benchmark/test_table_mean_std.csv"))
    pd.testing.assert_frame_equal(pd.DataFrame(summary).set_index(['regime','model']).sort_index(),
                                 old.set_index(['regime','model']).sort_index(),
                                 check_exact=False, rtol=1e-12, atol=1e-14)
    dump(output / "report.json", dict(test_only=True, checkpoints=100, pairs_per_checkpoint=249,
        seeds_per_architecture_regime=5, sd="Sample SD, ddof=1", predictions_ensembled=False,
        reduction="Per-seed physical-unit absolute error averaged over test time and longitude; equal seed mean afterwards",
        parameter_count="Notebook checkpoint-tensor count, which can include BatchNorm buffers; see trainable counts in hyperparameter export",
        exact_match_to_published_aggregate_arrays=True))
    print("BENCHMARK", output, flush=True)
    return arrays


def spatial(context):
    arrays = generate(context)
    output = context.output / "test"
    display = ['SimpleCNN2D', 'UnetCNN2D_Fixed', 'SimpleCNN3D', 'UnetCNN3D_Fixed',
               'Transformer2D_GA', 'Transformer2D_Swin', 'Transformer3D_GA', 'Transformer3D_Swin',
               'SimpleGNN2D', 'MeshGNN2D']
    scales = {}
    for v in VARIABLES:
        rows = {r: [(m, dict(label=label, map=arrays[f"{r}__{m}__map_{v}_mean"]))
                    for m, label in zip(MODEL_ORDER, display)] for r in REGIMES}
        fig, axes, payload = pu.plot_cached_model_maps(rows['no_ssw'], rows['with_ssw'],
            arrays['pressure'], arrays['latitude'], variable=v, architecture_names=MODEL_ORDER,
            n_pressure_bins=len(arrays['pressure']), n_lat_bins=len(arrays['latitude']),
            panel_titles=('No SSW', 'With SSW'), cmap='Reds')
        assert axes.shape == (10, 2)
        scales[v] = (payload['meta']['vmin'], payload['meta']['vmax'])
        for col, r in enumerate(REGIMES):
            for row, m in enumerate(MODEL_ORDER):
                values = arrays[f"{r}__{m}__map_{v}_mean"]
                np.testing.assert_array_equal(np.asarray(axes[row,col].collections[0].get_array()).reshape(values.shape), values)
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'mae_plevel_lat_all_models_{v}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    dump(output / 'spatial_scales.json', scales)
    print('SPATIAL', output, flush=True)


def persistence(context):
    if not (context.output / 'test/spatial_scales.json').exists():
        spatial(context)
    import json
    scales = json.loads((context.output / 'test/spatial_scales.json').read_text())
    # These six fields were calculated by the original, unmodified notebook function
    # and independently matched to the retained evaluator outputs before publication.
    path = context.input('results/paper/inputs/benchmark/persistence_latitude_pressure_maps.npz')
    with np.load(path) as cache:
        maps = {r: {v: cache[f'{r}__persistence_map_{v}'] for v in VARIABLES} for r in REGIMES}
        fig, axes, payload = pu.plot_cached_persistence(
            {'Persistence': maps['no_ssw']}, {'Persistence': maps['with_ssw']}, cache['pressure'], cache['latitude'],
            no_ssw_model_key='Persistence', with_ssw_model_key='Persistence',
            col_titles=('Temperature', 'Zonal Wind', 'Meridional Wind'), metric='mae', cmap='Reds',
            yticks=[100, 10, 1, 0.1, 0.01], vmin_vmax=scales)
        for row, r in enumerate(REGIMES):
            for col, v in enumerate(VARIABLES):
                np.testing.assert_array_equal(payload['maps'][('No SSW', 'With SSW')[row]][v], maps[r][v])
                np.testing.assert_allclose(axes[row,col].collections[0].get_clim(), scales[v], rtol=0, atol=0)
        for ext in ('png', 'pdf'):
            fig.savefig(context.output / 'test' / f'mae_plevels_lat_persistance_allvars.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    dump(context.output / 'test/persistence_report.json', dict(
        test_only=True, source=str(path.relative_to(context.root)), seed_averaging=False,
        definition='Adjacent evaluated test targets y_true[1:] versus y_true[:-1]; 248 pairs, exactly the notebook convention',
        source_target_indices=[3252,3499], scales=scales,
        saturated_values='Persistence above emulator vmax uses the darkest colour',
        renderer='Original notebook plotting body, accepting its verified saved physical-unit MAE maps'))
    print('PERSISTENCE', context.output / 'test', flush=True)
