"""Read-only probe: confirm the FCT8 implementation surface against the real H1 fold.

Answers, in one run:
  P1  geometry of data.mid / starts / months  (month gather is well-posed)
  P2  dynamic_basis shape and normalisation   (phi in the same units as u)
  P3  Endpoints parameter spec, and where delta_0..7 must be appended
  P4  h_preD29 at the published theta_ref       (the frozen FCT8 weight w)
  P5  w zero-weight reaches, and phi_bar
  P6  literal-vs-closed-form identity on REAL data at real phi_bar
  P7  h^D29 and h^FCT8 at delta=0, bitwise
"""
import sys, json, math
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUN / 'scripts')]
import numpy as np, torch
import campaign_model as CM

FOLD = 'T24_G_D_H1'
REF = RUN / 'outputs' / f'{FOLD}_s0' / 'model.json'

fold = json.loads((RUN / 'configs/folds.json').read_text(encoding='utf-8'))[FOLD]
print('P0 fold', {k: fold[k] for k in ('domain', 'train_years', 'evaluation_year')})

data = CM.load_data(fold['domain'], verify=True)
train = __import__('pandas').read_parquet(RUN / 'data/folds' / FOLD / 'train.parquet')
design = json.loads((RUN / 'data/designs' / f'{FOLD}.json').read_text(encoding='utf-8'))

# ---- P1 ---------------------------------------------------------------------
print('\nP1 ---')
starts = np.asarray(data.starts)
mid = np.asarray(data.mid)
nmon = len(data.months)
print('  dates', len(data.dates), 'months', nmon, 'starts', starts.shape, 'mid', mid.shape)
print('  mid range', int(mid.min()), int(mid.max()), 'dtype', mid.dtype)
cnt = np.bincount(mid, minlength=nmon)
print('  days-per-month min/max/sum', int(cnt.min()), int(cnt.max()), int(cnt.sum()))
print('  starts strictly increasing', bool(np.all(np.diff(starts) > 0)),
      '| starts[0]', int(starts[0]))
# the reduceat identity that phi_bar will use
assert np.array_equal(np.add.reduceat(np.ones(len(mid)), starts).astype(int),
                      np.bincount(mid, minlength=nmon)), 'reduceat/bincount disagree'
assert np.array_equal(np.repeat(np.arange(nmon), cnt), mid), 'mid is not the month of each day'
print('  reduceat(ones,starts) == bincount(mid) == repeat-block  OK')

# ---- P2 ---------------------------------------------------------------------
print('\nP2 ---')
m0 = CM.make_model(data, train, 'D29_BE', design)
db = m0.dynamic_basis
print('  dynamic_basis', tuple(db.shape), db.dtype)
print('  design dynamic_norm', design['scientific']['dynamic_norm'], 'sqrt8', math.sqrt(8))
print('  basis all-cols', tuple(m0.basis.shape))
print('  names', len(m0.names), 'bounds', len(m0.bounds),
      'variable_scale', m0.variable_scale().shape)
print('  tail names', m0.names[-4:], 'tail bounds', m0.bounds[-4:])

# ---- P4 ---------------------------------------------------------------------
print('\nP4 ---')
t = torch.tensor(np.asarray(json.loads(REF.read_text(encoding='utf-8'))['parameters'], float))
print('  theta_ref len', len(t), 'pg', json.loads(REF.read_text(encoding='utf-8'))['pg'])
print('  design in model.json == design file',
      json.loads(REF.read_text(encoding='utf-8'))['design'] == design)
h, s, f = CM.Predictor.hazard(m0, t[:29])
shift = m0.pi * (t[29] - t[1])
h_pre = h * torch.exp(shift[None, :] * m0.logcontact)
u = torch.einsum('trj,j->tr', db, t[21:29])
h_d29 = h_pre * torch.exp(math.log(10) * torch.tanh(u / math.log(10)))
h_direct, _, _, _ = m0.flux_parameters(t)
print('  h_preD29', tuple(h_pre.shape), 'finite', bool(torch.isfinite(h_pre).all()),
      'nonneg', bool((h_pre >= 0).all()))
print('  flux_parameters == h_pre*mult  max|d|', float((h_direct - h_d29).abs().max()))

# ---- P5 ---------------------------------------------------------------------
print('\nP5 ---')
w = h_pre.detach().numpy()
phi = db.numpy()                                   # (T,R,8) == phi/sqrt8
sw = np.add.reduceat(w, starts, axis=0)            # (M,R) monthly weight mass
bad = sw == 0                                      # the Sigma-w = 0 cells (cells, not reaches)
print('  w shape', w.shape, '| exact zeros %.2f%%' % (100.0 * (w == 0).mean()))
print('  reaches with a zero-mass month', int(bad.any(0).sum()),
      '| zero-mass (month,reach) cells', int(bad.sum()), 'of', sw.size)
print('  min w-mass over cells with mass %.6e' % float(sw[~bad].min()))
uni = np.add.reduceat(phi, starts, axis=0) / cnt[:, None, None]
phibar = np.divide(np.add.reduceat(w[:, :, None] * phi, starts, axis=0), sw[:, :, None],
                   out=np.zeros((nmon, phi.shape[1], 8)), where=~bad[:, :, None])
if bad.any():
    phibar[bad] = uni[bad]
print('  phi_bar', phibar.shape, 'finite', bool(np.isfinite(phibar).all()),
      'max|phi_bar| %.6e' % float(np.abs(phibar).max()))
print('  uniform fallback applied to %d cells' % int(bad.sum()))
# timing term must be w-zero-mean within month, on every cell with positive mass
res = phibar[mid] - phi
num = np.add.reduceat(w[:, :, None] * res, starts, axis=0)
agg = np.abs(np.where(~bad[:, :, None], num / np.where(sw == 0, 1, sw)[:, :, None], 0)).max()
print('  w-weighted within-month mean of (phibar-phi) max|.| %.3e' % float(agg))
print('  unweighted within-month mean of (phibar-phi) max|.| %.3e'
      % float(np.abs(np.add.reduceat(res, starts, axis=0) / cnt[:, None, None]).max()))

# ---- P6 ---------------------------------------------------------------------
print('\nP6 ---')
pb = torch.tensor(phibar)
midt = torch.tensor(mid)
b = t[21:29]
for d in (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 0.0):
    dv = torch.full((8,), d)
    bL, bT = b + dv / 2, b - dv / 2
    muL = torch.einsum('mrj,j->mr', pb, bL)[midt]
    muT = torch.einsum('mrj,j->mr', pb, bT)[midt]
    uT = torch.einsum('trj,j->tr', db, bT)
    literal = muL + (uT - muT)
    closed = torch.einsum('trj,j->tr', db, b) + torch.einsum('mrj,j->mr', pb, dv)[midt] \
             - 0.5 * torch.einsum('trj,j->tr', db, dv)
    g = float((literal - torch.einsum('trj,j->tr', db, b)).abs().max())
    print('  delta=%-7g literal-vs-closed %.3e   gap_u %.6e   gap/|d| %.6e'
          % (d, float((literal - closed).abs().max()), g, g / d if d else 0.0))

# ---- P7 ---------------------------------------------------------------------
print('\nP7 ---')
dv = torch.zeros(8)
bL, bT = b + dv / 2, b - dv / 2
muL = torch.einsum('mrj,j->mr', pb, bL)[midt]
muT = torch.einsum('mrj,j->mr', pb, bT)[midt]
uT = torch.einsum('trj,j->tr', db, bT)
uf = muL + (uT - muT)
a0 = math.log(10) * torch.tanh(u / math.log(10))
af = math.log(10) * torch.tanh(uf / math.log(10))
print('  max|u_fct(0)-u|      %.3e' % float((uf - u).abs().max()))
print('  max|mult_fct(0)-mult| %.3e' % float((torch.exp(af) - torch.exp(a0)).abs().max()))
print('  max|h_fct(0)-h_D29|   %.3e' % float((h_pre * torch.exp(af) - h_d29).abs().max()))
