"""Read-only remote bibliography/code provenance; never execute external code."""
import json,hashlib,time
import requests
import native_runtime as rt
R=rt.RUN
PAPERS=[
('dPL','10.1038/s41467-021-26107-z','Tsai et al.',2021),
('differentiable regional hydrology','10.1029/2022WR032404','Feng et al.',2022),
('MCP','10.1029/2023WR036461','Wang and Gupta',2024),
('input bias and hard conservation','10.1002/hyp.14847','Frame et al.',2023),
('POD DeepONet','10.1016/j.cma.2022.114778','Lu et al.',2022),
('sparse sensor reconstruction','10.1109/MCS.2018.2810460','Manohar et al.',2018),
('conservative ROM','10.1016/j.jcp.2018.05.019','Carlberg et al.',2018),
('FBPINNs','10.1007/s10444-023-10065-9','Moseley et al.',2023),
('differentiable geosciences perspective','10.1038/s43017-023-00450-9','Shen et al.',2023),
('deep learning water quality review','10.1038/s44221-024-00202-z','Zhi et al.',2024)]
REPOS=[
('ml-jku/mc-lstm',None,'mclstm.py'),
('YuanHWang/Mass-Conserving-Perceptron',None,'MCPBRNN_lib_tools/InputBiasCorr.py'),
('mhpi/hydrodl2',None,'src/hydrodl2/models/hbv/hbv.py'),
('lululxvi/deepxde','6e7aff8babcef184480bdc5f7dafee241aa28835','deepxde/nn/pytorch/deeponet.py'),
('dynamicslab/pysensors','8ea18cac79dde99ae401b0f0a096ce80bc0181c6','pysensors/reconstruction/_sspor.py'),
('benmoseley/FBPINNs','84ee6e1eaa0c6802698af46ed04978e97bb02f0a','fbpinns/trainers.py')]
def get(url):
    v=requests.get(url,timeout=25,headers={'User-Agent':'SPARROW-scientific-review/1.0'});v.raise_for_status();return v
def main():
    dest=R/'evidence/literature';dest.mkdir(parents=True,exist_ok=True);papers=[];repos=[]
    for topic,doi,authors,year in PAPERS:
        row=dict(topic=topic,doi=doi,authors_prior_verified=authors,year_prior_verified=year,url='https://doi.org/'+doi)
        try:
            m=get('https://api.crossref.org/works/'+doi).json()['message'];rt.write(dest/(topic.replace(' ','_')+'.json'),m);row.update(title=m.get('title'),metadata_verified=True)
        except Exception as e:row.update(metadata_verified=False,error=str(e),status='prior-session verified; this fetch incomplete')
        papers.append(row)
    papers.append(dict(topic='MC-LSTM',authors='Hoedt et al.',year=2021,url='https://proceedings.mlr.press/v139/hoedt21a.html',status='formal PMLR page verified in planning'))
    for repo,commit,path in REPOS:
        row=dict(repository=repo,file=path,code_use='reference only; not imported or executed')
        try:
            meta=get('https://api.github.com/repos/'+repo).json();branch=meta['default_branch'];commit=commit or get('https://api.github.com/repos/'+repo+'/commits/'+branch).json()['sha']
            body=get(f'https://raw.githubusercontent.com/{repo}/{commit}/{path}').content
            local=dest/(repo.replace('/','__')+'__core.py.txt');local.write_bytes(body)
            row.update(commit=commit,sha256=hashlib.sha256(body).hexdigest(),url=f'https://github.com/{repo}/blob/{commit}/{path}',license=meta.get('license'),archived=meta.get('archived'),verified=True)
        except Exception as e:row.update(verified=False,error=str(e))
        repos.append(row)
    rt.write(R/'reports/literature_code_registry.json',dict(papers=papers,repositories=repos,created=time.time(),limitations='Repository license metadata is recorded; no third-party core is incorporated. Abstract-only access is not represented as full-text reading.'))
    print('LITERATURE',len(papers),len(repos),flush=True)
if __name__=='__main__':main()
