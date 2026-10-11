"""Verify the adopted spatial files; no observations or model fitting."""
from pathlib import Path
import hashlib, json, os, sys, zipfile, tempfile

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
if sys.platform=='win32':
    _dll=os.add_dll_directory(str(Path(sys.executable).parent/'Library/bin'))
    os.environ.setdefault('GDAL_DATA',str(Path(sys.executable).parent/'Library/share/gdal'))
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx

def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    assert manifest['files'], 'EMPTY_MANIFEST'
    for row in manifest['files']:
        p=root/row['path'];b=p.read_bytes()
        assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'], row['path']
    assert manifest['archive_members'], 'EMPTY_ARCHIVE_MEMBERS'
    with zipfile.ZipFile(root/manifest['archive']) as z, tempfile.TemporaryDirectory() as temp:
        assert set(z.namelist())=={r['path'] for r in manifest['archive_members']}
        for row in manifest['archive_members']:
            b=z.read(row['path'])
            assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
            rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
        z.extractall(temp)
        cats=gpd.read_file(Path(temp)/'reach_catchments.shp').sort_values('reach_id')
    lines=gpd.read_file(root/'vectors/reaches_topology.shp')
    assert len(cats)==len(lines)==230
    assert cats.reach_id.astype(int).tolist()==list(range(1,231))
    assert sorted(lines.reach_id.astype(int))==list(range(1,231))
    assert cats.crs==lines.crs and cats.crs.is_projected
    assert cats.geometry.is_valid.all() and not cats.geometry.is_empty.any()
    assert lines.geometry.is_valid.all() and not lines.geometry.is_empty.any()
    area=cats.geometry.area.to_numpy()/1e6
    assert (area>0).all()
    cross=pd.read_csv(root/'tables/geometry_H1_area_crosswalk.csv')
    assert cross.reach_id.tolist()==list(range(1,231)) and len(cross)==230
    assert np.max(abs(area-cross.geometry_area_km2.to_numpy()))<1e-6
    assert np.max(abs(cross.H1_relative_difference-(cross.H1_effective_area_km2/area-1)))<1e-12
    rows=pd.read_csv(root/'tables/topology_edges.csv')
    graph=nx.DiGraph();graph.add_nodes_from(range(1,231))
    for row in rows.itertuples():
        if pd.notna(row.downstream_reach):graph.add_edge(int(row.reach_id),int(row.downstream_reach))
    top=json.loads((root/'tables/H1_FULL24C_topology.json').read_text())
    assert {(int(a)+1,int(b)+1) for a,b in top['downstream'].items()}==set(graph.edges())
    assert graph.number_of_edges()==216 and nx.is_directed_acyclic_graph(graph)
    assert nx.number_weakly_connected_components(graph)==14
    assert sum(graph.out_degree(n)==0 for n in graph.nodes)==14
    summary=json.loads((root/'evidence/spatial_validation.json').read_text())
    assert summary['passed'] and abs(area.sum()-summary['geometry_area_sum_km2'])<1e-6
    assert not summary['H1_area_difference_resolved']
    assert cross.loc[abs(cross.H1_relative_difference)>0.01,'reach_id'].tolist()==summary['H1_mismatch_gt1pct_reaches']
    print(json.dumps(dict(passed=True,verified_files=len(manifest['files']),catchments=230,
        edges=216,geometry_area_km2=float(area.sum()),H1_area_difference_resolved=False,NSE='not_applicable')))

if __name__=='__main__':main()
