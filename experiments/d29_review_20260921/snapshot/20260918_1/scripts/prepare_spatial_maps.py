"""Frozen-support maps; colour denotes label exclusion, never physical graph deletion."""
import sys
import pandas as pd,numpy as np
import geopandas as gp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import native_runtime as rt
R=rt.RUN;V=R.parent/'20260915_2'
sys.path.insert(0,str(V/'scripts'))
from release_loader import InputRelease

def main():
 release=InputRelease(V,training_mode=True);p=release.path('training_reaches');line=gp.read_parquet(p);c=gp.read_parquet(release.path('training_catchments'))
 rid=next(n for n in ['reach_id','global_reach_id','ReachID','COMID'] if n in line.columns);cid=next(n for n in ['reach_id','global_reach_id','ReachID','COMID'] if n in c.columns)
 blocks=rt.read(R/'data/spatial_blocks.json');fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
 records=[]
 for ax,(out,b) in zip(axes,blocks.items()):
  c.plot(ax=ax,facecolor='#f5f5f5',edgecolor='#d1d5db',linewidth=.25)
  for role,rr,color in [('held',b['held_reaches'],'#d97706'),('downstream_buffer',b['buffer_reaches'],'#a855f7')]:
   z=c[c[cid].isin(rr)]
   if len(z):z.plot(ax=ax,facecolor=color,edgecolor='white',alpha=.55,linewidth=.3)
  line.plot(ax=ax,color='#55758a',linewidth=.5)
  line[line[rid].eq(int(out))].plot(ax=ax,color='#b91c1c',linewidth=2)
  ax.set_title(f"Outlet {out}: {len(b['held_reaches'])} held reaches\n{len(b['held_stations'])} held / {len(b['buffer_stations'])} buffer stations")
  ax.set_aspect('equal');ax.set_axis_off()
  for role in ['held','buffer']:
   for r in b[role+'_reaches']:records.append(dict(outlet=int(out),reach_id=r,role=role))
 fig.suptitle('Fixed model supports: orange = held labels; purple = downstream exclusion\nFull 230-reach physical routing is retained')
 folder=R/'reports/maps';folder.mkdir(exist_ok=True);fig.savefig(folder/'spatial_outer_blocks.png',dpi=180);fig.savefig(folder/'spatial_outer_blocks.svg');plt.close(fig)
 pd.DataFrame(records).to_csv(folder/'spatial_membership.csv',index=False)
 rt.write(folder/'provenance.json',dict(line_hash=rt.sha(p),catchment_hash=rt.sha(release.path('training_catchments')),crs=str(line.crs),reach_column=rid,description='Frozen v3 model supports; not independent geographical catchment certification'))
 print('SPATIAL_MAPS_COMPLETE',list(line.columns),flush=True)
if __name__=='__main__':main()
