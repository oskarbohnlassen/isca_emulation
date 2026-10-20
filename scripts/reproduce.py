#!/usr/bin/env python
"""Generate the retained GRL paper/SI figures and tables from completed evaluations."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE','0')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('inventory','tables','profiles','spatial','persistence','temporal','ep-main','ep-si','all','verify'))
    parser.add_argument('--manifest',type=Path,default=Path(__file__).parent/'manifest.json',help='Populated local manifest (not included)')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1],help='Checkout/data root (default: this checkout)')
    parser.add_argument('--output',type=Path,default=None,help='Output directory (default: ROOT/results/paper/generated)')
    parser.add_argument('--event',default=None,help='One cached SI event; default: all five. Applies to ep-si only.')
    args=parser.parse_args()
    root=args.root.resolve()
    output=args.output.resolve() if args.output else root/'results/paper/generated'
    if args.event and args.command!='ep-si': parser.error('--event applies only to ep-si')
    if output==root or root/'results' not in output.parents:
        parser.error('--output must be a directory underneath ROOT/results')
    from paper.manifest import ManifestError, configure_manifest, read_manifest
    try:
        configure_manifest(args.manifest,args.command)
    except ManifestError as exc:
        parser.exit(2,f'{exc}\n')
    if args.command=='inventory':
        inventory=read_manifest('outputs.json')
        print(json.dumps(inventory,indent=2))
        return
    from paper.common import Context, dump
    context=Context(root,output)
    output.mkdir(parents=True,exist_ok=True)
    if args.command in ('all','tables'):
        from paper.tables import generate
        generate(context)
    if args.command in ('all','profiles'):
        from paper.profiles import generate
        generate(context)
    if args.command in ('all','spatial'):
        from paper.benchmark import spatial
        spatial(context)
    if args.command in ('all','persistence'):
        from paper.benchmark import persistence
        persistence(context)
    if args.command in ('all','temporal'):
        from paper.temporal import generate
        generate(context)
    if args.command in ('all','ep-main'):
        from paper.ep_flux import main_figures
        main_figures(context)
    if args.command in ('all','ep-si'):
        from paper.ep_flux import si_figures
        si_figures(context,args.event)
    if args.command in ('all','verify'):
        from paper.verify import verify
        verify(context)
    dump(output/f'inputs_used_{args.command}.json',dict(command=args.command,
        verified_input_files=sorted(context.verified),
        inference_required=False,notebooks_required=False,cluster_scripts_required=False))
    print('DONE',args.command,output,flush=True)


if __name__=='__main__':
    main()
