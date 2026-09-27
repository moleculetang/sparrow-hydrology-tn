"""Independent annual research inputs; national trends are not reach observations.

No TN read. Missing fertilizer rates are estimated separately from observed
support, never converted into known zero. A fixed complete national commodity
basket prevents changes in missingness from looking like yield trends.
"""
from pathlib import Path
import sys,ast,re,unicodedata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np,pandas as pd
from d29_platform.source_definitions import crop_bnf_kg
PARENT=ROOT.parent/'20260926_1'
AG=Path('E:/SPARROW/0_reach_topology/data/raw/agriculture')
OUT=ROOT/'outputs/agriculture_reference';OUT.mkdir(parents=True,exist_ok=True)
def key(s):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(s)).lower())
def main():
    tree=ast.parse((ROOT.parent/'20260815_2/scripts/run_stage2.py').read_text(encoding='utf-8'))
    aliases=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALIASES' for t in n.targets))
    aliases.update({'soyabeans':'soybeans','sugarcane':'sugarcane','sugarbeet':'sugarbeet','sweetpotatoes':'sweetpotatoes'})
    def norm(s):return aliases.get(key(s),key(s))
    cp=AG/'crop_n_removal/crop_n_content_coefficients/data'
    tier=pd.read_csv(cp/'Tier_1_and_2_crop_coefficients.csv')
    tier=tier[(tier.tier==2)&(tier.country=='China')&tier.N_kg_per_t_fresh_wt.notna()].copy()
    tier['coefficient_origin']='China_tier2'
    world=pd.read_csv(cp/'World_crop_coefficients_for_UN_FAO.csv');world['coefficient_origin']='world_tier1'
    coef=pd.concat([tier,world]);coef['key']=coef.item.map(norm);coef=coef.dropna(subset=['N_kg_per_t_fresh_wt']).drop_duplicates('key')
    respath=AG/'nitrogen_legacy/global_crop_residue_nutrient_removal_dryad_mgqnk99d1/crop_residue_coefficients.csv'
    res=pd.read_csv(respath);res['key']=res.Item.map(norm)
    res=res.pivot_table(index='key',columns='Element',values='Value',aggfunc='first').reset_index()
    res=res.rename(columns={'Dry matter conent in harvestable organ':'dry_percent','Harvest Index':'hi','N content in oven-dried crop residue':'residue_n_percent'})
    res=res[['key','dry_percent','hi','residue_n_percent']]
    path=PARENT/'outputs/faostat_crops/faostat_china_crop_activity_1961_2024.csv'
    fao=pd.read_csv(path,dtype={'item_cpc':str});fao['key']=fao.crop_item.map(norm)
    # Keep missing values, including all-missing years, through pivot.
    prod=fao[fao.element.eq('Production')][['year','key','crop_item','item_cpc','value']].rename(columns={'value':'production_t'})
    area=fao[fao.element.eq('Area harvested')][['year','key','value']].rename(columns={'value':'national_area_ha'})
    crop=prod.merge(area,on=['year','key'],validate='one_to_one').merge(coef[['key','N_kg_per_t_fresh_wt','coefficient_origin']],on='key',how='left',validate='many_to_one').merge(res,on='key',how='left',validate='many_to_one')
    required=['production_t','national_area_ha','N_kg_per_t_fresh_wt','dry_percent','hi','residue_n_percent']
    crop['complete']=crop[required].notna().all(axis=1)&crop.national_area_ha.gt(0)&crop.hi.gt(0)&crop.hi.le(1)
    good=crop.groupby('key').agg(n=('year','nunique'),all_complete=('complete','all'))
    basket=set(good[(good.n==64)&good.all_complete].index)
    crop['in_fixed_64year_basket']=crop.key.isin(basket)
    crop.to_csv(OUT/'national_commodity_mapping_and_missing.csv',index=False)
    q=crop[crop.in_fixed_64year_basket].copy()
    q['harvest_n_kg']=q.production_t*q.N_kg_per_t_fresh_wt
    q['residue_n_kg']=q.production_t*1000*q.dry_percent/100*(1/q.hi-1)*q.residue_n_percent/100
    named={'Barley':'Barley','Cassava':'Cassava, fresh','Cotton':'Seed cotton, unginned','Groundnut':'Groundnuts, excluding shelled','Maize':'Maize (corn)','Millet':'Millet','Oilpalm':'Oil palm fruit','Potato':'Potatoes','Rapeseed':'Rape or colza seed','Rice':'Rice','Rye':'Rye','Sorghum':'Sorghum','Soybean':'Soya beans','Sugarbeet':'Sugar beet','Sugarcane':'Sugar cane','Sweetpotato':'Sweet potatoes','Wheat':'Wheat','sunflower':'Sunflower seed'}
    audit=ROOT/'outputs/crop_rate_audit/legacy_crop_unknown_exposure.parquet'
    a=pd.read_parquet(audit);records=[];maps=[]
    country={};tot=q.groupby('year')[['national_area_ha','production_t','harvest_n_kg','residue_n_kg']].sum()
    for name in sorted(a.crop.unique()):
        if name in named:sub=q[q.key.eq(norm(named[name]))]
        elif name=='Fruits':sub=q[q.item_cpc.str.startswith('013')]
        elif name=='Vegetables':sub=q[q.item_cpc.str.startswith('012')]
        else:sub=q[~q.key.isin({norm(v) for v in named.values()})&~q.item_cpc.str.startswith(('012','013'))]
        # An explicit proxy, not automatic missing-data suppression.
        fallback=len(sub)==0
        if fallback:sub=q.copy()
        g=sub.groupby('year')[['national_area_ha','production_t','harvest_n_kg','residue_n_kg']].sum()
        if len(g)!=64 or (g.national_area_ha<=0).any():raise ValueError('NATIONAL_BASKET_INCOMPLETE '+name)
        country[name]=g
        maps.append(dict(crop=name,national_items=sorted(sub.crop_item.unique()),whole_basket_proxy=fallback,
            limitation='National yield and activity pattern used as conditional reach proxy, not a local crop observation'))
    fert=pd.read_csv(PARENT/'outputs/faostat_fertilizer_n/faostat_china_agricultural_use_total_n_1961_2024.csv').set_index('year').value_kg_N
    for (name,field,y),g in a.groupby(['crop','field','year'],sort=True):
        known_area=g.known_harvested_area_ha-g.positive_area_without_rate_ha
        donor_area=known_area.sum();donor_mass=g.known_partial_mass_kg.sum()
        donor_identity='same_crop_year'
        if donor_area<=0 and g.known_harvested_area_ha.sum()>0:
            proxy_groups={'Oilpalm':['Fruits'],'Sugarbeet':['Sugarcane'],'Rye':['Wheat','Barley'],
                          'Barley':['Wheat','Maize'],'Millet':['Sorghum','Maize'],'sunflower':['Rapeseed','Groundnut']}
            if name not in proxy_groups:raise ValueError('NO_REGISTERED_RATE_DONOR '+str((name,field,y)))
            dg=a[a.crop.isin(proxy_groups[name])&(a.field==field)&(a.year==y)]
            donor_area=(dg.known_harvested_area_ha-dg.positive_area_without_rate_ha).sum()
            donor_mass=dg.known_partial_mass_kg.sum();donor_identity='crop_group_proxy:'+','.join(proxy_groups[name])
            if donor_area<=0:raise ValueError('NO_REGISTERED_PERENNIAL_PROXY')
        # Zero *represented* area needs no rate; do not certify geographic absence.
        donor=donor_mass/donor_area if donor_area>0 else 0.
        newmass=g.known_partial_mass_kg+g.positive_area_without_rate_ha*donor
        area_factor=1. if y<=2020 else country[name].loc[y,'national_area_ha']/country[name].loc[2020,'national_area_ha']
        for ix,row in g.iterrows():
            records.append(dict(year=int(y),reach_id=int(row.reach_id),crop=name,field=field,
                harvested_area_ha=float(row.known_harvested_area_ha*area_factor),
                mass_kg=float(newmass.loc[ix]*area_factor),rate_gap_estimated_kg=float(row.positive_area_without_rate_ha*donor*area_factor),
                original_area_year=int(row.harvest_area_year),area_trend_factor=float(area_factor),
                rate_year=int(y),rate_donor_kg_ha=float(donor),rate_donor_identity=donor_identity,unknown_area_support=float(row.unknown_harvest_cell_fraction_sum)))
    r=pd.DataFrame(records)
    final=r[r.year.eq(2023)].copy();final.year=2024
    # Fertilizer country trend controls the 2024 TOTAL once; crop area trends
    # only redistribute it. Manure field application has no compatible 2024
    # national product: explicit constant-intensity-by-crop scenario.
    for name in final.crop.unique():
        k=final.crop.eq(name);f=country[name].loc[2024,'national_area_ha']/country[name].loc[2023,'national_area_ha']
        final.loc[k,['mass_kg','rate_gap_estimated_kg','harvested_area_ha']]*=f
        final.loc[k,'area_trend_factor']*=f
    fk=final.field.eq('Nfer');target=float(r[(r.year==2023)&r.field.eq('Nfer')].mass_kg.sum()*fert.loc[2024]/fert.loc[2023])
    normfactor=target/final.loc[fk,'mass_kg'].sum();final.loc[fk,['mass_kg','rate_gap_estimated_kg']]*=normfactor
    r=pd.concat([r,final],ignore_index=True)
    r.to_parquet(OUT/'annual_crop_source_reference.parquet',index=False)
    act=r[r.field.eq('Nfer')][['year','reach_id','crop','harvested_area_ha']].copy()
    act['harvest_n_kg']=0.;act['residue_n_kg']=0.;act['bnf_plant_kg']=0.;act['bnf_soil_kg']=0.
    for (name,y),idx in act.groupby(['crop','year']).groups.items():
        c=country[name].loc[y];area=act.loc[idx,'harvested_area_ha'].to_numpy()
        production=area*c.production_t/c.national_area_ha
        act.loc[idx,'harvest_n_kg']=area*c.harvest_n_kg/c.national_area_ha
        act.loc[idx,'residue_n_kg']=area*c.residue_n_kg/c.national_area_ha
        if name in ('Rice','Sugarcane'):act.loc[idx,'bnf_soil_kg']=crop_bnf_kg(name.lower(),area,production)
        elif name in ('Soybean','Groundnut'):act.loc[idx,'bnf_plant_kg']=crop_bnf_kg(name.lower(),area,production)
    act.to_parquet(OUT/'annual_crop_activity_reference.parquet',index=False)
    write_json(OUT/'receipt.json',dict(status='conditional_reference_not_independent_reach_observations',TN_used=False,
        complete_national_basket_items=len(basket),crop_mapping=maps,
        rate_gap_method='same crop and same year valid harvested-area weighted HaNi rate; oilpalm without any valid basin support uses explicit fruit-perennial proxy; estimated mass separately reported',
        area_nan_policy='Only represented positive harvested support is quantified; NaN combines author-encoded zero and possible unknown. No extra area imputed. National trend is an explicit conditional support scenario.',
        post2020_area='2020 within-crop spatial pattern times actual year national matched-basket area trend; not copied as observed year',
        fertilizer_2024='2023 corrected total times FAOSTAT N ratio; one normalization; area patterns only redistribute',
        manure_2024='2023 field-applied crop intensity held as declared management scenario, scaled by independently declared area; pasture EMN not added',
        bnf='FAOSTAT S2 nonlinear soybean/groundnut to plants; rice/sugarcane 25 kg/harvested ha to soil; other crop and natural BNF unknown and excluded from this conditional boundary',
        residue='Dryad DM*yield*(1/HI-1)*residue-N. No residue is external nitrogen.',
        hashes={str(p):sha(p) for p in (path,audit,respath,OUT/'annual_crop_source_reference.parquet',OUT/'annual_crop_activity_reference.parquet')},
        source_2024_kg=r[r.year==2024].groupby('field').mass_kg.sum().to_dict(),estimated_rate_gap_2024_kg=r[r.year==2024].groupby('field').rate_gap_estimated_kg.sum().to_dict(),
        official_data_complete=False,NSE='not applicable'))
    print('agriculture reference complete',len(r),'basket',len(basket),flush=True)
if __name__=='__main__':main()
