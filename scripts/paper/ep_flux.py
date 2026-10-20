"""Only the final five-seed EP figures: main event and all-model SI events.

EP physics is already evaluated independently for each seed in the pinned NetCDF
caches. Seed means and pointwise absolute errors are formed after the event mean.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from .common import dump, read_manifest, sha
from . import plotting as pu, climate

MAIN_MODELS = (
    ('mesh_gnn2d','Mesh MPNN'),('transformer2d_global_attention','Transformer 2D GA'),
    ('transformer3d_global_attention','Transformer 3D GA'),('cnn2d_unet','CNN2D U-Net'),
    ('cnn3d_unet','CNN3D U-Net'))
WAVE_MODELS = ('mesh_gnn2d','transformer3d_global_attention','cnn3d_unet')

MODELS = (
    ('cnn2d', 'Simple CNN2D'), ('cnn2d_unet', 'CNN2D U-Net'),
    ('cnn3d', 'Simple CNN3D'), ('cnn3d_unet', 'CNN3D U-Net'),
    ('transformer2d_global_attention', 'Transformer 2D GA'),
    ('transformer2d_swin', 'Transformer 2D Swin'),
    ('transformer3d_global_attention', 'Transformer 3D GA'),
    ('transformer3d_swin', 'Transformer 3D Swin'),
    ('gnn2d_gatedgcn', 'Grid MPNN'), ('mesh_gnn2d', 'Mesh MPNN'),
)
FIELDS = ('ep1', 'ep2', 'div', 'ubar', 'pv')
WAVE_FIELDS = ('ep1', 'ep2', 'div')
COMPONENTS = ('full', 'k=1', 'k=2', 'k=3', 'residual')
LIMITS = dict(ep1=1e-3, ep2=1e-7, div=1e-3, ubar=1e-3, pv=1e-12)

def set_figure_fonts(fig, *, title, label, tick):
    """Style the cached diagnostic figure without altering plotted fields."""
    for ax in fig.axes:
        ax.title.set_fontsize(title)
        ax.xaxis.label.set_fontsize(label)
        ax.yaxis.label.set_fontsize(label)
        ax.tick_params(axis="both", which="both", labelsize=tick)

def load_cached(context, event, regime, model_list=MODELS):
    """Only seed aggregation; all EP physics was completed in the saved caches."""
    first, last = map(int, event.split('_'))
    selected = read_manifest('selected_checkpoints.json')
    assert selected['verified']
    base = context.root / 'results/paper/inputs/ep_flux' / event / regime
    ref = next(r for r in selected['runs'] if r['model']=='cnn3d_unet' and r['regime']==regime and r['seed']==42)
    reference = base / 'cnn3d_unet' / f"seed42_{ref['run_id']}" / 'truth.nc'
    context.input(str(reference.relative_to(context.root)))
    sources = {str(reference.relative_to(context.root)): sha(reference)}
    with xr.open_dataset(reference) as data:
        truth = {f: data['truth_'+f].load() for f in FIELDS}
        wave_truth = {c: {f: data[f"wave_truth_{c.replace('=', '')}_{f}"].load() for f in WAVE_FIELDS}
                      for c in COMPONENTS}
    saved = {f'truth_{f}': v for f,v in truth.items()}
    saved.update({f"truth__{c.replace('=', '')}__{f}": v for c,values in wave_truth.items() for f,v in values.items()})
    models, provenance, deltas = [], {}, {}
    for model,label in model_list:
        rows = sorted([r for r in selected['runs'] if r['model']==model and r['regime']==regime], key=lambda r:r['seed'])
        assert len(rows)==len({r['seed'] for r in rows})==5
        assert any(r['seed']==42 for r in rows)
        provenance[model] = rows
        predictions = []
        for row in rows:
            folder = base / model / f"seed{row['seed']}_{row['run_id']}"
            path = folder/'ep_diagnostics.nc'
            own_truth_path = folder/'truth.nc'
            context.input(str(path.relative_to(context.root)))
            context.input(str(own_truth_path.relative_to(context.root)))
            sources[str(path.relative_to(context.root))] = sha(path)
            sources[str(own_truth_path.relative_to(context.root))] = sha(own_truth_path)
            with xr.open_dataset(path) as data, xr.open_dataset(own_truth_path) as own:
                assert data.attrs['run_id']==row['run_id'] and int(data.attrs['seed'])==row['seed']
                assert (int(data.attrs['target_first']),int(data.attrs['target_last']))==(first+1,last)
                values = {f: data['prediction_'+f].load() for f in FIELDS}
                for f in FIELDS:
                    xr.align(truth[f],values[f],own['truth_'+f],join='exact')
                    assert np.isfinite(values[f]).all()
                    delta=float(np.max(np.abs(truth[f].values-own['truth_'+f].values)))
                    assert delta <= LIMITS[f], (model,row['seed'],f,delta)
                    deltas[f"{model}_seed{row['seed']}_{f}"]=delta
                for c in COMPONENTS:
                    key=c.replace('=','')
                    for f in WAVE_FIELDS:
                        pred = data[f'wave_prediction_{key}_{f}'].load()
                        actual = own[f'wave_truth_{key}_{f}']
                        xr.align(wave_truth[c][f],pred,actual,join='exact')
                        assert np.isfinite(pred).all()
                        np.testing.assert_array_equal(data[f'wave_error_{key}_{f}'],actual-pred)
                        delta=float(np.max(np.abs(wave_truth[c][f].values-actual.values)))
                        assert delta<=LIMITS[f], (model,row['seed'],c,f,delta)
                        deltas[f"{model}_seed{row['seed']}_{key}_{f}"]=delta
                        values[f'{key}__{f}']=pred
            predictions.append(xr.Dataset(values))
        stack=xr.concat(predictions,dim=xr.IndexVariable('seed',[r['seed'] for r in rows])).astype('float64')
        mean={f:stack[f].mean('seed') for f in FIELDS}
        errors={f:truth[f]-mean[f] for f in WAVE_FIELDS}
        for f in WAVE_FIELDS:
            np.testing.assert_allclose(errors[f],(truth[f]-stack[f]).mean('seed'),rtol=1e-10,atol=1e-12)
        wave_mean, wave_mae = {}, {}
        for c in COMPONENTS:
            key=c.replace('=','');wave_mean[c],wave_mae[c]={},{}
            for f in WAVE_FIELDS:
                pred=stack[f'{key}__{f}']
                signed=wave_truth[c][f]-pred
                wave_mean[c][f]=pred.mean('seed')
                wave_mae[c][f]=np.abs(signed).mean('seed')
                tol=64*np.finfo(float).eps*max(1,float(np.max(np.abs(pred))))
                assert (wave_mae[c][f].values+tol>=np.abs(signed.mean('seed').values)).all()
                saved[f'{model}__{key}__mean_prediction_{f}']=wave_mean[c][f]
                saved[f'{model}__{key}__mean_absolute_error_{f}']=wave_mae[c][f]
        for f in WAVE_FIELDS:
            np.testing.assert_allclose(sum(stack[f'{k}__{f}'] for k in ('k1','k2','k3','residual')),
                                       stack[f'full__{f}'],rtol=1e-6,atol=1e-8)
            saved[f'{model}__mean_signed_error_{f}']=errors[f]
        saved.update({f'{model}__mean_prediction_{f}':v for f,v in mean.items()})
        models.append(dict(model=model,label=label,mean=mean,errors=errors,predictions=wave_mean,wave_mae=wave_mae))
    return truth,wave_truth,models,saved,provenance,deltas,sources

def total_figure(truth,models,event,output):
    """Rearrange the unchanged notebook's truth/error panel calls into 1+5+5."""
    fig=plt.figure(figsize=(18,11.6),dpi=140)
    grid=fig.add_gridspec(3,5,left=0.055,right=0.985,bottom=0.15,top=0.925,hspace=0.24,wspace=0.12)
    truth_ax=fig.add_subplot(grid[0,2])
    common=dict(lat_min=0,lat_max=90,vmin_vmax=(-10,10),arrow_scale=4e16,arrow_width=0.002,
                pv_level=2e-6,cmap='PuOr_r')
    payload=pu.ep_flux_panel(fig,truth_ax,*(truth[f] for f in FIELDS),climate,
                             title='True EP-flux',x_label='',y_label='Pressure [hPa]',**common)
    pu._ensure_log_pressure_axis(truth_ax)
    axes=[truth_ax]
    for i,m in enumerate(models):
        row,col=1+i//5,i%5
        ax=fig.add_subplot(grid[row,col],sharex=truth_ax,sharey=truth_ax)
        mean=m['mean']
        rendered=pu.ep_flux_panel(fig,ax,mean['ep1'],mean['ep2'],m['errors']['div'],mean['ubar'],mean['pv'],climate,
                                 add_ep_flux_arrows=False,title=m['label'],
                                 x_label='Latitude' if row==2 else '',y_label='Pressure [hPa]' if col==0 else '',**common)
        np.testing.assert_array_equal(rendered['div'],m['errors']['div'])
        pu._ensure_log_pressure_axis(ax)
        ax.tick_params(labelleft=col==0,labelbottom=row==2)
        axes.append(ax)
    fig.suptitle(f'SSW event {event.replace("_", "–")} · five training seeds',fontsize=19,y=0.982)
    cax=fig.add_axes([0.14,0.058,0.72,0.022])
    bar=fig.colorbar(payload['cf'],cax=cax,orientation='horizontal')
    bar.set_label(r'Divergence / signed error (m s$^{-1}$ day$^{-1}$)')
    set_figure_fonts(fig,title=16,label=15,tick=12)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    assert truth_ax.get_tightbbox(renderer).y0>axes[3].title.get_window_extent(renderer).y1
    assert min(ax.get_tightbbox(renderer).y0 for ax in axes[6:])>bar.ax.get_tightbbox(renderer).y1
    for ext in ('pdf','png'):
        fig.savefig(output/f'ep_flux_seed_mean_signed_error.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return dict(panel_count=11,model_grid=[2,5],truth_location='Centered above model rows',
                figsize_inches=[18,11.6],error_arrows=False,truth_arrows=True,
                wind_and_tropopause_contours=True,range=[-10,10],side_annotations=False,
                panel_function='paper.plotting.ep_flux_panel')

def wave_figure(truth,models,event,output):
    render=pu.plot_cached_wave_grid
    changes="Materialized published cached-field renderer"
    inputs=[dict(label=m['label'],predictions=m['predictions'],errors=m['wave_mae']) for m in models]
    fig,axes,payload=render(truth_fields=truth,model_fields=inputs,pressure_levels=truth['full']['div'].pres.values,
                           lat_levels=truth['full']['div'].lat.values,lon_levels=[],climate=climate,
                           model_row_titles=[m['label'] for m in models],model_rows_as_error=True,
                           model_error_label_suffix='',vmin_vmax=(-10,10),figsize=(15,21),
                           add_model_row_arrows=False,colorbar=False,show=False)
    assert axes.shape==(11,5) and payload['component_keys']==list(COMPONENTS)
    for row in range(11):
        expected=truth if row==0 else models[row-1]['wave_mae']
        for col,c in enumerate(COMPONENTS):
            for f in WAVE_FIELDS:np.testing.assert_array_equal(payload['panel_outputs'][row][col][f],expected[c][f])
            assert axes[row,col].get_yscale()=='log' and axes[row,col].yaxis_inverted()
        assert payload['row_specs'][row]['show_arrows']==(row==0)
        label='True' if row==0 else models[row-1]['label'].replace('Transformer ', 'Transformer\n')
        axes[row,0].set_ylabel(label+'\nPressure [hPa]')
    axes[0,0].set_title('All waves')
    axes[0,4].set_title(r'Small waves ($k\geq4$)')
    fig.set_layout_engine(None)
    left,right=0.15,0.985
    fig.subplots_adjust(left=left,right=right,top=0.96,bottom=0.09,hspace=0.07,wspace=0.08)
    fig.suptitle(f'SSW event {event.replace("_", "–")} · mean absolute divergence error across five seeds',fontsize=18,y=0.988)
    gap=0.035;width=(right-left-gap)/2
    bars=[]
    for x,cf,label in ((left,payload['panel_outputs'][0][0]['cf'],'True divergence'),
                       (left+width+gap,payload['panel_outputs'][1][0]['cf'],'Mean absolute divergence error')):
        cax=fig.add_axes([x,0.042,width,0.013])
        bar=fig.colorbar(cf,cax=cax,orientation='horizontal')
        bar.set_label(label+r' (m s$^{-1}$ day$^{-1}$)')
        bars.append(bar)
    set_figure_fonts(fig,title=16,label=15,tick=12)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=[b.ax.get_tightbbox(renderer) for b in bars]
    assert bounds[0].x1<bounds[1].x0
    assert min(ax.get_tightbbox(renderer).y0 for ax in axes[-1])>max(b.y1 for b in bounds)
    for ext in ('pdf','png'):
        fig.savefig(output/f'ep_flux_waves_seed_mean_absolute_error.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return dict(grid=[11,5],figsize_inches=[15,21],adapter_changes=changes,
                renderer='Unchanged notebook wave-grid renderer with existing cached-field adapter',
                truth_range=[-10,10],error_range=[0,10],truth_arrows=True,error_arrows=False,
                colorbars='Equal-width bars below the plot area; single-line labels')


def main_figures(context):
    event, regime = '4618_4698', 'with_ssw'
    output = context.output / 'ep_flux/main' / event
    output.mkdir(parents=True, exist_ok=True)
    truth, waves, models, saved, runs, deltas, sources = load_cached(context,event,regime,MAIN_MODELS)
    # Same arguments as the published notebook cell-25 call, with the final five model columns.
    model_inputs = [dict(label=m['label'],diagnostics=m['mean'],error_div=m['errors']['div']) for m in models]
    fig, axes, payload = pu.plot_cached_ep_row(truth,model_inputs,truth['div'].pres.values,
        truth['div'].lat.values,[],climate,vmin_vmax=(-10,10),plot_divergence_difference_below=True,
        model_titles_ep_flux=[m['label'] for m in models],
        model_titles_divergence_error=[m['label'] for m in models],show=False)
    assert axes.shape==(2,6)
    np.testing.assert_allclose(fig.get_size_inches(),(19.2,10))
    for i,m in enumerate(models,1):
        for f in FIELDS:
            np.testing.assert_array_equal(payload['panels'][i][f],m['mean'][f])
        np.testing.assert_array_equal(payload['difference_panels'][i]['div'],m['errors']['div'])
    payload['difference_colorbar'].set_label(
        r'Divergence / signed error (truth $-$ prediction; m s$^{-1}$ day$^{-1}$)',
        fontsize=plt.rcParams['axes.titlesize'])
    set_figure_fonts(fig,title=20,label=18,tick=16)
    for ext in ('pdf','png'):
        fig.savefig(output/f'ep_flux_seed_mean_signed_error.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    subset = [m for key in WAVE_MODELS for m in models if m['model']==key]
    inputs = [dict(label=m['label'],predictions=m['predictions'],errors=m['wave_mae']) for m in subset]
    fig,axes,payload = pu.plot_cached_wave_grid(truth_fields=waves,model_fields=inputs,
        pressure_levels=waves['full']['div'].pres.values,lat_levels=waves['full']['div'].lat.values,
        lon_levels=[],climate=climate,model_row_titles=[m['label'] for m in subset],
        model_rows_as_error=True,model_error_label_suffix=' (divergence error)',
        vmin_vmax=(-10,10),show=False,add_model_row_arrows=False,colorbar=False)
    assert axes.shape==(4,5)
    for row in range(4):
        expected = waves if row==0 else subset[row-1]['wave_mae']
        for col,c in enumerate(COMPONENTS):
            for f in WAVE_FIELDS:
                np.testing.assert_array_equal(payload['panel_outputs'][row][col][f],expected[c][f])
        assert payload['row_specs'][row]['show_arrows']==(row==0)
    for row,m in enumerate(subset,1):
        axes[row,0].set_ylabel(f"{m['label']}\nPressure [hPa]")
    fig.set_layout_engine(None)
    fig.set_size_inches(14,10.2)
    left,right=0.10,0.985
    fig.subplots_adjust(left=left,right=right,top=0.955,bottom=0.16,hspace=0.075,wspace=0.09)
    gap=0.025
    width=(right-left-gap)/2
    bars=[]
    for x,cf,label in ((left,payload['panel_outputs'][0][0]['cf'],'True divergence'),
                       (left+width+gap,payload['panel_outputs'][1][0]['cf'],'Mean absolute divergence error')):
        bar=fig.colorbar(cf,cax=fig.add_axes([x,0.075,width,0.022]),orientation='horizontal')
        bar.set_label(label+r' (m s$^{-1}$ day$^{-1}$)')
        bars.append(bar)
    set_figure_fonts(fig,title=16,label=16,tick=14)
    fig.canvas.draw()
    bounds=[b.ax.get_tightbbox(fig.canvas.get_renderer()) for b in bars]
    assert bounds[0].x1<bounds[1].x0
    assert min(ax.get_tightbbox(fig.canvas.get_renderer()).y0 for ax in axes[-1])>max(b.y1 for b in bounds)
    for ext in ('pdf','png'):
        fig.savefig(output/f'ep_flux_waves_seed_mean_absolute_error.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    xr.Dataset(saved,attrs=dict(event=event,regime=regime,seed_count=5,
        aggregation='Per-seed event-mean diagnostics first, seed mean / pointwise MAE second')).to_netcdf(output/'plot_fields.nc')
    dump(output/'report.json',dict(event=event,regime=regime,runs=runs,source_sha256=sources,
        common_truth_roundtrip_max_differences=deltas,inference_required=False,
        signed_error='truth minus the mean of five event-mean diagnostics',
        wave_error='mean across seeds of pointwise absolute error of event-mean diagnostics',
        raw_field_ensemble=False,main_model_order=[m['model'] for m in models],wave_model_order=WAVE_MODELS))
    print('MAIN EP',output,flush=True)


def si_figures(context, event=None):
    events=[r['event'] for r in read_manifest('events.json')['events'] if r['period']=='post-test']
    if event is not None:
        if event not in events:
            raise ValueError(f'Event must be one of {events}')
        events=[event]
    for event in events:
        output=context.output/'ep_flux/si'/event
        output.mkdir(parents=True,exist_ok=True)
        truth,waves,models,saved,runs,deltas,sources=load_cached(context,event,'with_ssw')
        total=total_figure(truth,models,event,output)
        wave=wave_figure(waves,models,event,output)
        xr.Dataset(saved,attrs=dict(event=event,regime='with_ssw',seed_count=5,
            aggregation='Per-seed event-mean diagnostics first, seed mean / pointwise MAE second')).to_netcdf(output/'plot_fields.nc')
        dump(output/'report.json',dict(event=event,regime='with_ssw',runs=runs,source_sha256=sources,
            common_truth_roundtrip_max_differences=deltas,inference_required=False,
            signed_figure=total,wave_figure=wave,
            aggregation='EP diagnostics computed per seed/event first; signed mean and pointwise absolute seed error afterwards'))
        print('SI EP',event,flush=True)
