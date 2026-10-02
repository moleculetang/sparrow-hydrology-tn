"""Execute the complete diagnostic readout on a known synthetic amplitude case."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import diagnostics as d
from mltn.common import ROOT,write


def main():
    previous=d.ROOT
    with tempfile.TemporaryDirectory(prefix='synthetic_diagnostics_',dir=ROOT/'transfer') as temp:
        root=Path(temp);(root/'outputs/evaluation').mkdir(parents=True);(root/'data').mkdir()
        y=np.array([1.,3.,2.,4.,3.,5.]);p=y.mean()+.5*(y-y.mean())
        q=pd.DataFrame(dict(station_key=['fixture']*6,date=pd.date_range('2023-01-01',periods=6),observed=y,prediction=p,read_count=[2]*6))
        q.to_parquet(root/'outputs/evaluation/fixture_daily_frozen.parquet',index=False)
        r=pd.DataFrame(dict(station_key=['fixture']*12,date=np.repeat(q.date.to_numpy(),2),eligible=True,adopted_value=np.repeat(y,2)+np.tile([-1.,1.],6)))
        r.to_parquet(root/'data/hf_readings.parquet',index=False)
        try:d.ROOT=root;d.main()
        finally:d.ROOT=previous
        table=pd.read_csv(root/'outputs/evaluation/error_decomposition_station.csv')
        row=table[table.scope.eq('level')].iloc[0]
        np.testing.assert_allclose([row.NSE,row.correlation,row.amplitude_ratio],[.75,1.,.5],atol=1e-12)
        np.testing.assert_allclose(row.bias_squared+row.amplitude_mismatch_squared+row.decorrelation_error,row.mse,atol=1e-12)
        within=table[table.scope.eq('month_centered')].iloc[0]
        np.testing.assert_allclose([within.NSE,within.correlation,within.amplitude_ratio],[.75,1.,.5],atol=1e-12)
        assert len(pd.read_parquet(root/'outputs/evaluation/unresolved_four_hour_variation.parquet'))==6
    write(ROOT/'evidence/diagnostic_readout_acceptance.json',dict(passed=True,known_NSE=.75,known_correlation=1.,known_amplitude_ratio=.5,scope='whole diagnostic main, level and month-centered decomposition and reading support; isolated synthetic input, no real TN or fitting'))
    print('DIAGNOSTIC_READOUT_PASS')


if __name__=='__main__':main()
