"""Publication exports from the pinned, completed TEST evaluations and configs."""
from collections import defaultdict
import csv
import json
import statistics

import numpy as np
import pandas as pd
import yaml

from .common import MODEL_ORDER, REGIMES, dump, load_selected, read_manifest

LABELS = dict(zip(MODEL_ORDER, ('Simple CNN2D', 'CNN2D U-Net', 'Simple CNN3D', 'CNN3D U-Net',
    'Trf.-2D GA', 'Trf.-2D Swin', 'Trf.-3D GA', 'Trf.-3D Swin', 'Grid MPNN', 'Mesh MPNN')))
CONTROLLED = {
    'cnn_width128': 'Simple CNN, 2D-source HPs, shared width 128',
    'cnn_3dbest': 'Simple CNN, 3D-selected HPs, width 64',
    'unet_width128': 'U-Net, 2D-source HPs, shared base width 128',
    'unet_3dbest': 'U-Net, 3D-selected HPs, base width 50',
    'globalattn_2dbest': 'Global attention, 2D-selected HPs',
    'globalattn_3dbest': 'Global attention, 3D-selected HPs',
}


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main_table(context):
    from .benchmark import generate
    generate(context)
    output = context.output / 'tables'
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(context.output / 'test/test_table_mean_std.csv').set_index(['regime','model'])
    baseline = pd.read_csv(context.output / 'test/test_persistence_table.csv').set_index('regime')
    metrics = ('MAE_temp','MAE_ucomp','MAE_strat_temp','MAE_strat_ucomp')
    # Fragment retains manuscript macros and ranking; the manuscript owns the caption.
    lines = [r'% Physical-unit MAE multiplied by 100; sample SD across five retained seeds.',
             r'% Full atmosphere: temp, ucomp; p<=30 hPa: temp, ucomp; checkpoint tensor count.',
             r'\begin{tabular}{l cc cc c cc cc c}', r'\toprule',
             r' & \multicolumn{5}{c}{No SSW} & \multicolumn{5}{c}{With SSW} \\',
             r'Architecture & Temp & Z. wind & Strat. temp & Strat. Z. wind & Params & Temp & Z. wind & Strat. temp & Strat. Z. wind & Params \\',
             r'\midrule']
    persistence = []
    for r in REGIMES:
        persistence += [f'{100*baseline.loc[r,m]:.2f}' for m in metrics] + ['--']
    lines.append('Persistence & ' + ' & '.join(persistence) + r' \\')
    for model in MODEL_ORDER:
        cells = []
        for r in REGIMES:
            for m in metrics:
                row = df.loc[r, model]
                cell = f"{100*row[m+'_mean']:.2f} $\\pm$ {100*row[m+'_std']:.2f}"
                order = df.loc[r, m+'_mean'].sort_values(kind='stable').index.tolist()
                if order.index(model) < 3:
                    cell = '\\' + ('best','second','third')[order.index(model)] + '{' + cell + '}'
                cells.append(cell)
            params = int(round(df.loc[(r, model),'num_parameters_mean']))
            cell = f'{params/1e6:.0f}M'
            order = df.loc[r,'num_parameters_mean'].sort_values(kind='stable').index.tolist()
            if order.index(model) < 3:
                cell = '\\' + ('best','second','third')[order.index(model)] + '{' + cell + '}'
            cells.append(cell)
        lines.append(LABELS[model] + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (output / 'main_benchmark_table.tex').write_text('\n'.join(lines)+'\n')


def hyperparameters(context):
    document = read_manifest('selected_hyperparameters.json')
    manifest, _ = load_selected(context)
    records, wide, long, provenance = document['records'], [], [], []
    assert len(records) == 20
    def flatten(obj, prefix=''):
        result = {}
        for k,v in obj.items():
            if isinstance(v,dict): result.update(flatten(v,prefix+k+'.'))
            else: result[prefix+k] = v
        return result
    for record in records:
        rows = [r for r in manifest['runs'] if (r['regime'],r['model']) == (record['regime'],record['model_key'])]
        assert len(rows) == 5
        for row in rows:
            cfg = yaml.safe_load(context.input('config_train/'+row['config']).read_text())
            reference = dict(record['raw_config'])
            reference['train'] = dict(reference['train'],random_seed=row['seed'])
            assert cfg == reference
            provenance.append({k: row[k] for k in ('regime','model','seed','run_id','wandb_url','selection_reason',
                'config','config_sha256','artifact_name','checkpoint_file','checkpoint_digest','num_parameters')})
        meta = dict(regime=record['regime'], architecture=record['architecture'],
            seeds=','.join(map(str,record['selected_seeds'])),
            trainable_parameters=record['num_trainable_parameters'], config_source=record['base_config_source'])
        fields = flatten(dict(config=record['raw_config'], effective_train=record['effective_train'],
            effective_optimizer=record['effective_optimizer'],effective_scheduler=record['effective_scheduler'],
            effective_loss=record['effective_loss'],implementation=record['architecture_implementation_settings']))
        wide.append({**meta, **{k: json.dumps(v) if isinstance(v,list) else v for k,v in fields.items()}})
        long.extend(dict(**meta,parameter=k,value=json.dumps(v)) for k,v in fields.items())
    output = context.output / 'tables'
    write_csv(output/'selected_20_hyperparameters.csv',wide)
    write_csv(output/'selected_20_hyperparameters_long.csv',long)
    write_csv(output/'selected_100_checkpoint_provenance.csv',provenance)
    dump(output/'selected_20_hyperparameters.json',document)
    (output/'selected_20_configs.yaml').write_text(yaml.safe_dump(
        {r['regime']+'/'+r['model_key']:r['raw_config'] for r in records},sort_keys=False))


def event_inventory(context):
    doc = read_manifest('events.json')
    output = context.output/'tables'
    rows = [dict(**r,elapsed_grid_days=r['onset']/8) for r in doc['events']]
    write_csv(output/'ssw_event_inventory.csv',rows)
    counts = {p:sum(r['period']==p for r in rows) for p in ('training','validation','test','post-test')}
    assert counts == dict(training=1,validation=0,test=0,**{'post-test':5}) and len(rows)==6
    dump(output/'ssw_event_detection.json',doc)
    (output/'ssw_inventory_sentence.tex').write_text(
        'After excluding the initial 1,000 timesteps as spin-up, we detect six SSW events in the SSW-enabled simulation: '
        'one during training, none during validation, and none during the held-out test period, with the remaining five '
        'occurring after the test period and used for the EP-flux robustness analysis.\n')


def controlled(context):
    document = read_manifest('controlled_checkpoints.json')
    assert document['verified'] and len(document['runs']) == 36
    raw, groups, configs = [], defaultdict(list), {}
    for row in document['runs']:
        stem = f"{row['regime']}_{row['model']}_seed{row['seed']}_{row['run_id']}"
        folder = 'results/paper/inputs/controlled/compact/' + stem
        info = context.json(folder+'/summary.json')
        for k in ('run_id','seed','checkpoint_digest','config_sha256','artifact_name'):
            assert row[k] == info['run'][k]
        assert info['samples']==249 and not info['raw_results_retained']
        cfg = yaml.safe_load(context.input('config_train/'+row['config']).read_text())
        configs[row['comparison'],row['representation'],row['seed']] = cfg
        physical = info['table'][row['model_type']]
        rec = dict(comparison=row['comparison'],representation=row['representation'],seed=row['seed'],
            run_id=row['run_id'],split='test',test_pairs=249,test_mae=info['metrics']['scaled']['mae_mean'],
            test_mae_definition='Normalized all-state-element MAE (unitless)',num_parameters=row['num_parameters'],
            **{f'physical_{k}':v for k,v in physical.items() if k.startswith('MAE')},
            checkpoint_file=row['checkpoint_file'],artifact_name=row['artifact_name'],
            checkpoint_digest=row['checkpoint_digest'],wandb_url=row['wandb_url'],config=row['config'],
            result_source=folder+'/summary.json')
        assert np.isfinite(rec['test_mae'])
        raw.append(rec)
        groups[row['comparison'],row['representation']].append(rec)
    summary, audit = [], {}
    for key,label in CONTROLLED.items():
        pair = {r:sorted(groups[key,r],key=lambda x:x['seed']) for r in ('2d','3d')}
        for rep, records in pair.items():
            assert [r['seed'] for r in records] == [42,43,44]
            assert len({r['num_parameters'] for r in records}) == 1
        values = {r:[a['test_mae'] for a in pair[r]] for r in pair}
        a,b = statistics.mean(values['2d']),statistics.mean(values['3d'])
        summary.append(dict(comparison=label,comparison_key=key,metric='test_mae',seeds='42,43,44',n=3,
            test_mae_2d_mean=a,test_mae_2d_sample_sd=statistics.stdev(values['2d']),
            test_mae_3d_mean=b,test_mae_3d_sample_sd=statistics.stdev(values['3d']),
            reduction_3d_percent=100*(a-b)/a,params_2d=pair['2d'][0]['num_parameters'],params_3d=pair['3d'][0]['num_parameters']))
        two,three = [configs[key,r,42] for r in ('2d','3d')]
        assert two['train'] == three['train']
        audit[key] = dict(config_2d=two,config_3d=three,training_identical=True,
            coordinate_difference='3D includes linear/log pressure coordinates in addition to trig spatial inputs',
            unstable_cnn_width128_seed42_retained=key=='cnn_width128')
    old = pd.read_csv(context.input('results/paper/inputs/controlled/summary_test_mae.csv'))
    pd.testing.assert_frame_equal(pd.DataFrame(summary),old,check_exact=False,rtol=1e-12,atol=1e-14)
    output = context.output/'tables'
    write_csv(output/'controlled_test_metrics_per_seed.csv',raw)
    write_csv(output/'controlled_test_summary.csv',summary)
    dump(output/'controlled_configuration_audit.json',audit)
    lines = [r'% Unitless held-out TEST MAE; sample SD across seeds 42,43,44.',
             r'\begin{tabular}{lccccc}',r'\toprule',
             r'Comparison & 2D test MAE & 3D test MAE & 3D reduction & 2D params & 3D params \\',r'\midrule']
    for r in summary:
        lines.append(f"{r['comparison']} & {r['test_mae_2d_mean']:.6f} $\\pm$ {r['test_mae_2d_sample_sd']:.6f} & "
            f"{r['test_mae_3d_mean']:.6f} $\\pm$ {r['test_mae_3d_sample_sd']:.6f} & "
            f"{r['reduction_3d_percent']:.1f}\\% & {r['params_2d']:,} & {r['params_3d']:,}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (output/'controlled_comparisons.tex').write_text('\n'.join(lines)+'\n')


def generate(context):
    main_table(context)
    hyperparameters(context)
    event_inventory(context)
    controlled(context)
    print('TABLES',context.output/'tables',flush=True)
