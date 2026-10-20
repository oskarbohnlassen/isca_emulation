"""Compare regenerated fields, tables and figure pixels with the published outputs."""
import ast
import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import xarray as xr

from .common import dump, read_manifest, sha


def verify(context):
    inventory=read_manifest('outputs.json')
    comparisons=[]
    for row in inventory['figures']:
        original=context.input(row['reference_png'])
        generated=context.output/row['output_png']
        if sha(original)!=row['reference_png_sha256']:
            raise ValueError(f'Published reference changed: {original}')
        old,new=mpimg.imread(original),mpimg.imread(generated)
        same_shape=old.shape==new.shape
        exact=same_shape and np.array_equal(old,new)
        difference=float(np.max(np.abs(old.astype(float)-new.astype(float)))) if same_shape else None
        comparisons.append(dict(figure=row['output_png'],original_shape=list(old.shape),
            generated_shape=list(new.shape),exact_pixel_match=exact,max_channel_difference=difference))
    # Keep a complete comparison report even if one plot needs inspection.
    dump(context.output/'figure_verification.json',dict(figures=comparisons,
        exact_pixel_matches=sum(c['exact_pixel_match'] for c in comparisons),expected=len(comparisons)))
    for name in ('test_metrics_per_seed.csv','test_table_mean_std.csv','test_persistence_table.csv'):
        original=pd.read_csv(context.input('results/paper/inputs/benchmark/'+name))
        generated=pd.read_csv(context.output/'test'/name)
        keys=['run_id'] if name=='test_metrics_per_seed.csv' else (['regime','model'] if name=='test_table_mean_std.csv' else ['regime'])
        pd.testing.assert_frame_equal(generated.set_index(keys).sort_index(),original.set_index(keys).sort_index(),
                                      check_exact=False,rtol=1e-12,atol=1e-14)
    for event in [r['event'] for r in read_manifest('events.json')['events'] if r['period']=='post-test']:
        with xr.open_dataset(context.input(f'results/paper/inputs/reference/ep_flux/si/{event}/with_ssw/plot_fields.nc')) as old, \
             xr.open_dataset(context.output/f'ep_flux/si/{event}/plot_fields.nc') as new:
            assert set(old.data_vars)==set(new.data_vars)
            for key in old.data_vars:
                xr.testing.assert_equal(old[key],new[key])
    # The total and wave main-text figures use their specific retained five/three models.
    with xr.open_dataset(context.output/'ep_flux/main/4618_4698/plot_fields.nc') as new, \
         xr.open_dataset(context.input('results/paper/inputs/reference/ep_flux/main/4618_4698/with_ssw/panel_fields.nc')) as old:
        for key in new:
            if key in old:
                xr.testing.assert_equal(new[key],old[key])
    with xr.open_dataset(context.output/'ep_flux/main/4618_4698/plot_fields.nc') as new, \
         xr.open_dataset(context.input('results/paper/inputs/reference/ep_flux/main/4618_4698/with_ssw/wave_panel_fields.nc')) as old:
        for model in ('mesh_gnn2d','transformer3d_global_attention','cnn3d_unet'):
            for c in ('full','k1','k2','k3','residual'):
                for f in ('ep1','ep2','div'):
                    a=new[f'{model}__{c}__mean_absolute_error_{f}']
                    b=old[f'{model}__{c}__{f}__absolute_mean_error']
                    np.testing.assert_array_equal(a.values,b.values)
    # Original function source was moved verbatim; no need to read a notebook to audit it.
    source_files=[Path(__file__).with_name('plotting.py'),Path(__file__).with_name('climate.py')]
    texts={p:p.read_text() for p in source_files}
    for name,record in read_manifest('source_provenance.json')['functions'].items():
        if record['migration'].startswith('Verbatim'):
            matches=[]
            for path,text in texts.items():
                for node in ast.parse(text).body:
                    if isinstance(node,ast.FunctionDef) and node.name==name:
                        source='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
                        matches.append(hashlib.sha256(source.encode()).hexdigest())
            assert matches==[record['source_text_sha256']],name
    dump(context.output/'numerical_verification.json',dict(
        benchmark_tables_match=True,aggregate_arrays_match=True,
        temporal_curves_match=True,all_five_si_ep_field_sets_match=True,main_ep_fields_match=True,
        verbatim_scientific_functions_match=True))
    failures=[c for c in comparisons if not c['exact_pixel_match']]
    if failures:
        raise AssertionError('Figures differ from the published pixels; inspect figure_verification.json: '+
                             ', '.join(c['figure'] for c in failures))
    print('VERIFIED all numerical outputs and',len(comparisons),'pixel-identical PNG figures',flush=True)
