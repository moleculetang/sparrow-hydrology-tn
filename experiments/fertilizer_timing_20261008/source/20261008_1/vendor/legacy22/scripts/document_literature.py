"""Turn verified citation/commit records into the experiment's inspectable adoption table."""
import native_runtime as rt
R=rt.RUN

def main():
    registry=rt.read(R/'reports/literature_code_registry.json')
    uses={'dPL':'共享映射产生过程参数，保留完整物理递推；不外推大样本效果。','differentiable regional hydrology':'属性与动态响应交互；不继承float32或截断热身梯度。','MCP':'输入估计可校正，同时保留质量账本；不解释为真实排放识别。','input bias and hard conservation':'守恒不能修复输入偏差；需求与源校正分开。','POD DeepONet':'训练期低维基底和少数系数；普通重建不自动守恒。','sparse sensor reconstruction':'稀疏站点的模式可观测性；区分借助实测重建与无TN输入预测。','conservative ROM':'局地守恒不能被全域总量约束替代。','FBPINNs':'局部表示有潜力，不用PINN替换已有库存核。','differentiable geosciences perspective':'明确可学习映射与物理过程的职责边界。','deep learning water quality review':'水质学习须考虑输入、采样支持和泛化限制。','MC-LSTM':'质量输入与调节信息分离；不引入新的神经质量池。'}
    text='# 文献—代码证据与本轮采用边界\n\n本轮没有移植或执行外部模型核心。文献支持有限方法探索，不保证15个HF站上的预测收益。元数据查证不冒充逐篇全文阅读。\n\n|主题|已核验题名与链接|本轮采用及边界|\n|---|---|---|\n'
    for p in registry['papers']:
        title=p.get('title',p['topic']);title='；'.join(title) if isinstance(title,list) else title
        text+=f"|{p['topic']}|[{title}]({p.get('url','')})；查证状态 {p.get('metadata_verified',p.get('verified','见登记'))}|{uses.get(p['topic'],'仅作方法背景，不视作本流域效能证据。')}|\n"
    text+='\n## 固定代码版本及许可登记\n\n|仓库|核心文件／提交|许可元数据|使用方式|\n|---|---|---|---|\n'
    for p in registry['repositories'] if 'repositories' in registry else registry.get('code',[]):
        lic=p.get('license') or {};text+=f"|{p.get('repository')}|[{p.get('file')}]({p.get('url')})；`{p.get('commit')}`|{lic.get('spdx_id','未核实')}|只读参考，未导入执行；字节哈希见JSON|\n"
    text+='\nNOASSERTION表示GitHub未给出可确定的SPDX许可，不能据此宣称许可已明确。本轮不对外发布参考代码。完整网络核查状态、固定提交和文件SHA256见 reports/literature_code_registry.json。\n'
    (R/'文献与代码证据.md').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
