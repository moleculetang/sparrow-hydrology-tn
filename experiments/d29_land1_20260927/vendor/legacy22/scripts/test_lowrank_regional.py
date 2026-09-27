import native_runtime as rt
from lowrank_regional import *

def main():
    rng=np.random.default_rng(1729);u=rng.normal(size=(80,2));v=rng.normal(size=(9,2));y=u@v.T;w=np.ones_like(y);w[rng.random(y.shape)<.1]=0
    masked=y.copy();masked[w==0]=np.nan
    a,b,h=masked_modes(masked,w,2);error=np.max(abs((a@b.T-y)[w>0]));assert error<1e-7,error
    other=masked.copy();other[w==0]=1e20;aa,bb,hh=masked_modes(other,w,2);assert np.array_equal(a,aa) and np.array_equal(b,bb)
    x=np.column_stack([np.ones(80),rng.normal(size=(80,4))]);basis=np.linalg.qr(rng.normal(size=(9,2)))[0];coef=coefficient_ridge(x,y,np.ones_like(y),basis);expected=np.column_stack([ridge(x,(y@basis)[:,k],np.ones(80)) for k in range(2)]);assert np.max(abs(coef-expected))<1e-10
    one=coefficient_ridge(x,masked,w,basis);two=coefficient_ridge(x,other,w,basis);assert np.array_equal(one,two)
    dates=pd.date_range('2021-01-01',periods=730);s=season(dates);truth=s@rng.normal(size=5);coef=wls(s,truth,np.ones(len(s)));assert np.max(abs(truth-s@coef))<1e-10
    rt.write(RUN/'reports/lowrank_fixture.json',dict(status='PASS',masked_rank2_error=float(error),missing_values_ignored=True,seasonal_recovery=True,evaluation_not_argument=True))
    print('PASS lowrank fixture',error)
if __name__=='__main__':main()
