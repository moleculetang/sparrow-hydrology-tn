"""Verify the published unified ZIP without accessing experiment data or TN."""
from pathlib import Path
import hashlib, json, os, sys, tempfile, zipfile
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
if sys.platform=='win32':
    _dll=os.add_dll_directory(str(Path(sys.executable).parent/'Library/bin'))
    os.environ.setdefault('GDAL_DATA',str(Path(sys.executable).parent/'Library/share/gdal'))
import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx

def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    assert manifest['files'] and manifest['archive_members'], 'EMPTY_MANIFEST'
    for row in manifest['files']:
        b=(root/row['path']).read_bytes()
        assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
    with zipfile.ZipFile(root/manifest['archive']) as z, tempfile.TemporaryDirectory() as temp:
        assert set(z.namelist())=={r['path'] for r in manifest['archive_members']}
        for row in manifest['archive_members']:
            b=z.read(row['path'])
            assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
            rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
        z.extractall(temp)
        p=Path(temp);c=p/'canonical'
        cats=gpd.read_file(p/'vectors/reach_catchments.shp').sort_values('reach_id')
        assert cats.reach_id.astype(int).tolist()==list(range(1,231))
        assert cats.crs.is_projected and cats.geometry.is_valid.all() and not cats.geometry.is_empty.any()
        area=cats.geometry.area.to_numpy()/1e6;assert (area>0).all()
        cross=pd.read_parquet(c/'training_area_crosscheck.parquet').sort_values('reach_id')
        registry=pd.read_parquet(c/'hydrology_reach_registry_v3.parquet').sort_values('reach_id')
        assert cross.reach_id.tolist()==registry.reach_id.tolist()==list(range(1,231))
        for a in [cross.hydrology_area_m2.to_numpy()/1e6,cross.frozen_area_m2.to_numpy()/1e6,registry.local_area_m2.to_numpy()/1e6]:
            assert np.max(abs(area-a))<1e-9
        edges=pd.read_parquet(c/'hydrology_edges_v3.parquet')
        pairs={(int(r.upstream_reach_id),int(r.downstream_reach_id)) for r in edges.itertuples()}
        top=json.loads((c/'hydrology_model_topology_v3.json').read_text())
        assert pairs=={(int(a)+1,int(b)+1) for a,b in top['downstream'].items()}
        assert pairs=={(int(r.reach_id),int(r.downstream_reach_id)) for r in registry.itertuples() if r.downstream_reach_id>0}
        graph=nx.DiGraph();graph.add_nodes_from(range(1,231));graph.add_edges_from(pairs)
        assert len(edges)==len(pairs)==216 and nx.is_directed_acyclic_graph(graph)
        links=pd.read_parquet(c/'hydrology_reservoir_links_v3.parquet')
        assert len(links)==links.reservoir_id.nunique()==13
        for r in links.itertuples():
            assert set(map(int,r.control_reach_ids))<=set(range(1,231)) and int(r.target_reach_id) in range(1,231)
            assert 0<=r.local_capture_fraction<=1
        import rasterio
        with rasterio.open(c/'hydrology_support_labels_30m_v3.tif') as raster:
            labels=set()
            for _,window in raster.block_windows(1):
                labels.update(map(int,np.unique(raster.read(1,window=window))))
            assert set(range(1,231))<=labels and labels<={0,*range(1,231)}
        proof=json.loads((root/'evidence/topology_v3_completion_audit.json').read_text())
        assert proof['status']=='PASS_HYDROLOGY_ALIGNED_TOPOLOGY_V3' and all(proof['checks'].values())
        lineage=json.loads((root/'evidence/current_area_lineage_readback.json').read_text())
        assert abs(area.sum()-lineage['current_H1']['sum_km2'])<1e-9
        print(json.dumps(dict(passed=True,files=len(manifest['files']),archive_members=len(manifest['archive_members']),
            reaches=230,edges=216,reservoirs=13,area_km2=float(area.sum()),
            maximum_area_difference_km2=float(np.max(abs(area-cross.hydrology_area_m2.to_numpy()/1e6))),
            support_labels_verified=True,NSE='not_applicable')))

if __name__=='__main__':main()
