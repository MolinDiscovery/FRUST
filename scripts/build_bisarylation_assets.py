"""Build exact reviewed-reference figures from an explicit local evidence bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

import frust as ft


def main() -> None:
    """Export the same real local barriers and structures shown in the guide."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_root',type=Path)
    parser.add_argument('--output',type=Path,default=Path('docs/assets/bisarylation'))
    args=parser.parse_args();root=args.evidence_root.resolve().parent;out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=root/'runs/stage04/native_composed_TMP_review_20261029/local_barriers.csv'
    bars=pd.read_csv(source);columns=['ts_type','reference_state','product_state','delta_e_kcal_mol','delta_g_kcal_mol','reverse_delta_e_kcal_mol','reverse_delta_g_kcal_mol','dimer_relative_e_kcal_mol']
    bars[columns].to_csv(out/'local-barriers.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    labels={'TS5':('M','INT4-A'),'TS6':('INT4-B','INT5 + H₂'),'TS7':('INT5 + HBpin','INT6/C7'),'TS8':('INT6/C8','Products')}
    for axis,row in zip(axes.flat,bars.itertuples()):
        start,end=labels[row.ts_type];f=float(row.delta_e_kcal_mol);reverse=float(row.reverse_delta_e_kcal_mol)
        ft.plot_energy_profile([(start,0),(row.ts_type,f),(end,f-reverse)],ax=axis,
            ylabel='Local ΔE / kcal mol⁻¹',show_state_labels=True,hide_y_ticks=False,
            hide_spines=False,annotate_energies=True,show_legend=False,state_label_fontsize=9,energy_fontsize=9)
        axis.set_title(row.ts_type+' · its own neighboring conformers',fontsize=11)
    fig.savefig(out/'local-barriers.svg',bbox_inches='tight');plt.close(fig)
    mols=[];frames=[];sources={'local_barriers_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'references':{}}
    for state in ['TS5','TS6','TS7','TS8']:
        file=args.evidence_root/('stage02/resumed_20261028/ts6_reviewed_native_candidate.json' if state=='TS6' else f'stage03/{state}_reviewed_native_candidate.json')
        record=json.loads(file.read_text());source_file=Path(record['source_dataframe']);source_file=source_file if source_file.is_absolute() else root/source_file
        if not source_file.exists() and 'runs' in source_file.parts:
            source_file=root.joinpath(*source_file.parts[source_file.parts.index('runs'):])
        assert hashlib.sha256(source_file.read_bytes()).hexdigest()==record['source_sha256']
        row=pd.read_parquet(source_file)
        row=row.loc[row.cid==record['cid']].iloc[0];xyz=np.asarray(record['source_geometry']['coords']);atoms=record['source_geometry']['atoms']
        params=Chem.SmilesParserParams();params.removeHs=False;mol=Chem.MolFromSmiles(row.smiles,params)
        assert [a.GetSymbol() for a in mol.GetAtoms()]==atoms
        conformer=Chem.Conformer(len(atoms))
        for i,v in enumerate(xyz):conformer.SetAtomPosition(i,tuple(map(float,v)))
        mol.RemoveAllConformers();mol.AddConformer(conformer);mols.append(mol)
        frames.append(dict(substrate_name=state,atoms=atoms,reviewed_oc=xyz))
        sources['references'][state]=dict(source_sha256=record['source_sha256'],cid=int(record['cid']),reference_json_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    ft.write_xyz(pd.DataFrame(frames),out/'xyz',coords_col='reviewed_oc',overwrite=True)
    ft.MolTo3DGrid(mols,legends=['TS5: second C–H activation','TS6: H₂ release','TS7: bridge entry','TS8: bridge exit'],columns=2,cell_size=(320,270),linked=False,show_confs=False,show_charges=False,background_color=('white',1),export_HTML=str(out/'references.html'))
    vendor=root/'visual/vendor/3Dmol-2.5.5.min.js'
    license_file=vendor.with_name('3Dmol-package-LICENSE.txt')
    html=out/'references.html'
    html.write_text(html.read_text().replace('<script>', '<script>'+vendor.read_text().replace('</script','<\\/script')+'\nvar $3Dmolpromise=Promise.resolve();\n</script>\n<script>',1))
    (out/'3Dmol-LICENSE.txt').write_bytes(license_file.read_bytes())
    for file in [out/'local-barriers.svg',out/'3Dmol-LICENSE.txt']:
        file.write_text('\n'.join(line.rstrip() for line in file.read_text().splitlines()).rstrip()+'\n')
    sources['viewer_sha256']=hashlib.sha256(vendor.read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
