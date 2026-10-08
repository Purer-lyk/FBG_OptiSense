"""Generate Chinese report strictly from executed benchmark outputs."""
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from manual_touch_sensor_token import metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/model_comparisons/sensor_token26_two_sessions_v1'
DOC=OUT/'SensorToken注意力与稀疏CNN_26类位置及三维力对比报告.docx'


def main():
    comparison=json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
    split=json.loads((OUT/'split.json').read_text(encoding='utf-8'))
    data=np.load(OUT/'dataset.npz',allow_pickle=False)
    readonly_path=Path('C:/Users/Administrator/AppData/Local/hermes/cache/scratch/fbg_touch_dataset_audit.json')
    readonly=json.loads(readonly_path.read_text(encoding='utf-8'))
    for source,session in zip(audit['sources'],readonly['sessions']):
        frame=next(f for f in session['jsonl_files'] if Path(f['path']).name=='frames.jsonl')
        assert source['sha256']==frame['sha256'] and source['lines']==frame['frame_count']
    shutil.copy2(readonly_path,OUT/'readonly_dataset_audit.json')
    old=json.loads((ROOT/'outputs/manual_touch_sequences/press_grid_20260927_113931/models/manual_touch_sparse_cnn.report.json').read_text(encoding='utf-8'))
    kinds=['sensor_token','sparse_cnn26']; titles={'sensor_token':'Sensor Token 注意力','sparse_cnn26':'稀疏CNN（26类重训）'}
    reports={}; predictions={}
    for k in kinds:
        seed=comparison['selected_models'][k]['seed']
        reports[k]=next(r for r in comparison['runs'] if r['architecture']==k and r['seed']==seed)
        predictions[k]=np.load(OUT/f'{k}_seed{seed}_test_predictions.npz',allow_pickle=False)
        assert np.array_equal(predictions[k]['indices'],split['test'])
    assert len(comparison['runs'])==6 and all(r['checkpoint_reload_verified'] for r in comparison['runs'])
    train=np.asarray(split['train']); test=np.asarray(split['test'])
    mean_force=data['forces'][train][data['valid'][train]].mean(0)
    with np.errstate(invalid='ignore',divide='ignore'):
        baseline=metrics(data['labels'][test],np.full(len(test),int(np.bincount(data['labels'][train]).argmax())),data['forces'][test],np.tile(mean_force,(len(test),1)),data['valid'][test])
    baseline['force_pearson']=[None,None,None]  # Constant predictions have undefined correlation.
    (OUT/'constant_baseline.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'logical_cpus':os.cpu_count(),'files_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['manual_touch_sensor_token.py','manual_touch_cnn.py','run_sensor_token_comparison.py','sensor_token_subprocess_tests.py']},'docx':'python-docx','input_source_hashes':{s['path']:s['sha256'] for s in audit['sources']}}
    (OUT/'experiment_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','Arial Unicode MS'] if f in available),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10})
    chart_dir=OUT/'figures'; chart_dir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,k in zip(axes,kinds):
        cm=np.asarray(reports[k]['metrics']['test']['confusion_matrix'])
        image=ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(1,cm.max()))
        ax.set(xticks=np.arange(26),yticks=np.arange(26),xlabel='预测标签（0=空载）',ylabel='真实标签',title=titles[k]); ax.tick_params(labelsize=7)
        fig.colorbar(image,ax=ax,fraction=.046)
    fig.tight_layout(); fig.savefig(chart_dir/'confusion.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    indices=np.asarray(split['test']); xs=np.arange(len(indices)); force=data['forces'][indices]
    cut=np.flatnonzero((np.diff(indices)!=1)|(np.diff(data['run'][indices])!=0)|(np.diff(data['session'][indices])!=0))+1
    segments=np.split(np.arange(len(indices)),cut)
    for j,(ax,name) in enumerate(zip(axes,['Fx','Fy','Fz'])):
        for seg_index,seg in enumerate(segments):
            ax.plot(xs[seg],force[seg,j],color='#172B4D',linewidth=1.4,label='真实力' if seg_index==0 else None)
            for k,color,style in [('sensor_token','#087E8B','--'),('sparse_cnn26','#D95F02',':')]:
                ax.plot(xs[seg],predictions[k]['predicted_forces'][seg,j],color=color,linestyle=style,linewidth=1.1,label=titles[k] if seg_index==0 else None)
        ax.set_ylabel(name+' / N'); ax.grid(alpha=.2); ax.legend(loc='upper right',fontsize=8)
    axes[0].set_title('测试集实测与预测力：跨采集段和被隔离区间断线，不插值连接')
    axes[-1].set_xlabel('测试帧序号（不是连续真实时间）')
    fig.tight_layout(); fig.savefig(chart_dir/'force_curves.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k,color in [('sensor_token','#087E8B'),('sparse_cnn26','#D95F02')]:
        h=reports[k]['history']; x=np.arange(1,len(h)+1)
        axes[0].plot(x,[v['validation_accuracy']*100 for v in h],color=color,label=titles[k])
        axes[1].plot(x,[v['validation_standardized_force_mae'] for v in h],color=color,label=titles[k])
    axes[0].set(xlabel='Epoch',ylabel='验证位置准确率 / %'); axes[1].set(xlabel='Epoch',ylabel='验证标准化力MAE')
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(chart_dir/'learning.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4)); x=np.arange(26)
    for k,offset,color in [('sensor_token',-.18,'#087E8B'),('sparse_cnn26',.18,'#D95F02')]:
        recalls=np.array([r['recall'] for r in reports[k]['metrics']['test']['per_class']])*100
        ax.bar(x+offset,recalls,width=.36,color=color,label=titles[k])
    ax.set(xticks=x,xlabel='真实位置标签（0=空载）',ylabel='逐类召回率 / %',ylim=(0,105)); ax.legend(); ax.grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(chart_dir/'per_class.png',dpi=180); plt.close(fig)

    doc=Document(); section=doc.sections[0]
    section.top_margin=Cm(1.8); section.bottom_margin=Cm(1.8); section.left_margin=Cm(1.9); section.right_margin=Cm(1.9)
    normal=doc.styles['Normal']; normal.font.name='Calibri'; normal.font.size=Pt(10)
    normal._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    normal.paragraph_format.space_after=Pt(6)
    for h in ['Title','Heading 1','Heading 2']:
        doc.styles[h]._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
        doc.styles[h].font.color.rgb=RGBColor.from_string('17365D')
    section.header.paragraphs[0].text='三号机 FBG 触觉识别 | 两数据集重新训练与对比'
    footer=section.footer.paragraphs[0]; footer.text='开发评估 · 未完成跨会话现场泛化验证  |  页 '
    field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)
    def p(text): doc.add_paragraph(str(text))
    def heading(text,level=1): doc.add_heading(text,level)
    def table(headers,rows):
        t=doc.add_table(rows=1, cols=len(headers)); t.style='Table Grid'
        for c,text in zip(t.rows[0].cells,headers): c.text=str(text)
        repeat=OxmlElement('w:tblHeader'); t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in rows:
            for c,value in zip(t.add_row().cells,row): c.text=str(value)
        for row in t.rows:
            for c in row.cells:
                for para in c.paragraphs:
                    for run in para.runs: run.font.size=Pt(8)
        return t
    def scalar(v,digits=3,percent=False):
        if v is None: return '无参考'
        return f'{v*100 if percent else v:.{digits}f}'+('%' if percent else '')
    def agg(kind,key,index=None,percent=False):
        a=comparison['aggregate_test_metrics'][kind][key]; m=a['mean']; s=a['std_sample']
        if index is not None: m=m[index]; s=s[index]
        scale=100 if percent else 1
        return f'{m*scale:.2f} ± {s*scale:.2f}'+(' %' if percent else '')
    doc.add_heading('Sensor Token 注意力与稀疏CNN\n26类按压位置及三维力回归对比报告',0)
    p('输入：九个光栅，每个光栅一个五维Token；共享注意力特征提取器，26类互斥位置分类头与Fx/Fy/Fz回归头。')
    p('报告全部数值来自本次实际训练输出；原始JSONL、删除备份和旧25类CNN模型均未改写。训练过程中未使用旧模型权重或测试集选择最优种子。')
    heading('1. 结果摘要')
    accuracy_token=comparison['aggregate_test_metrics']['sensor_token']['position_accuracy']['mean']
    accuracy_cnn=comparison['aggregate_test_metrics']['sparse_cnn26']['position_accuracy']['mean']
    force_token=comparison['aggregate_test_metrics']['sensor_token']['force_vector_mae_n']['mean']
    force_cnn=comparison['aggregate_test_metrics']['sparse_cnn26']['force_vector_mae_n']['mean']
    p(f'三个固定种子的测试集平均位置准确率：Sensor Token {accuracy_token*100:.2f}%，稀疏CNN {accuracy_cnn*100:.2f}%；注意力相对CNN差值为{(accuracy_token-accuracy_cnn)*100:+.2f}个百分点。三维向量平均误差：注意力{force_token:.3f} N，CNN {force_cnn:.3f} N（越小越好）。这是同一开发划分的实测结果，不能概括为架构在所有数据上更优。')
    p('重要限制：空载只来自10-08、按压只来自09-27；会话和接触状态完全混淆。多数位置只有一次连续按压段。本次未实现覆盖26类的独立会话测试，deployment_ready=false。')
    rows=[]
    for key,name,percent in [('position_accuracy','位置准确率',True),('macro_f1_present_classes','26类宏平均F1',False),('balanced_accuracy_present_classes','平衡准确率',True),('pressed_only_accuracy','仅1～25号准确率',True),('no_press_recall','0号空载召回率',True),('no_press_precision','0号空载精确率',True),('force_vector_mae_n','三维向量MAE / N',False)]:
        rows.append([name,agg(kinds[0],key,percent=percent),agg(kinds[1],key,percent=percent)])
    table(['测试指标：均值 ± 种子标准差','Sensor Token','26类稀疏CNN'],rows)
    p('标准差为三个随机种子的样本标准差，不是独立按压轮次的置信区间；三次训练使用相同测试帧，因此不能将它解释为跨会话稳定性。')
    p('综合解释：本次注意力的平均准确率仅小幅高于CNN，差异小于两种模型跨种子的标准差，不能宣布显著提升。CNN宏F1与Fx/Fy回归更好；注意力Fz/向量MAE略小、空载精确率更高且种子波动较小。不同指标各有取舍，没有全面胜出。')
    group_rows=[]
    for low,high in [(1,19),(20,25)]:
        values=[]
        for k in kinds:
            pred=predictions[k]; mask=(pred['labels']>=low)&(pred['labels']<=high)
            values.append(scalar(float(np.mean(pred['labels'][mask]==pred['predicted_labels'][mask])),2,True))
        group_rows.append([f'{low}～{high}号',int(mask.sum())]+values)
    table(['交付模型测试子集','帧数','Sensor Token准确率','CNN准确率'],group_rows)
    p('特别警示：20～25号测试帧全部来自后补采段，注意力与CNN都明显低于1～19号同段尾部表现。这提示轮次漂移/分布变化风险，而不是已经证实具体硬件或标签原因。注意力训练准确率99.40%，测试仅61.36%，存在显著过拟合/泛化差距；不建议把此模型视为已验收的现场识别模型。')
    heading('最终交付的单个checkpoint（按验证集选中）',2)
    table(['测试指标','Sensor Token选中模型','CNN选中模型'],[[name,scalar(reports[kinds[0]]['metrics']['test'][key],digits,percent),scalar(reports[kinds[1]]['metrics']['test'][key],digits,percent)] for key,name,digits,percent in [('position_accuracy','位置准确率',2,True),('macro_f1_present_classes','宏F1',3,False),('pressed_only_accuracy','仅按压位置准确率',2,True),('no_press_precision','空载精确率',2,True),('no_press_recall','空载召回率',2,True),('force_vector_mae_n','三维向量MAE/N',3,False)]])
    table(['交付模型三轴力指标','Sensor Token','CNN'],[[axis+' '+name,scalar(reports[kinds[0]]['metrics']['test'][key][j]),scalar(reports[kinds[1]]['metrics']['test'][key][j])] for j,axis in enumerate(['Fx','Fy','Fz']) for key,name in [('force_mae_n','MAE/N'),('force_rmse_n','RMSE/N'),('force_r2','R²')]])
    p('两个交付模型均选中seed=20261010；注意力最佳第113轮，CNN第36轮。该选择依据验证分数，测试集不参与选择；单个交付模型的分数不等于三个种子的均值。')

    heading('2. 数据审计与力标签')
    table(['批次','主文件帧数','位置类别','有效45维ADC','有效三轴力'],[['09-27',2420,'1～25',2420,2420],['10-08',100,'仅0号空载',100,100],['合计',2520,'0～25共26类',2520,2520]])
    p('只使用两批frames.jsonl；frames.removed_label17_last18.jsonl是已删除的错误标注备份，未混回。主文件无重复source_frame_sequence、无完全相同45维ADC向量。两批45点波长路线、九峰索引顺序与25点物理布局完全一致；这只确认存储契约一致，不证明物理传感器身份一致。')
    p('输入来自quality.stable_adc（同帧45个second_code），按光栅1～9每组五点组织；不使用first_code、不插值、不为不存在的光栅制造Token。')
    f=data['forces'][data['labels']==0]
    p(f'空载真实Fx/Fy/Fz均值为{f.mean(0).round(6).tolist()} N，标准差为{f.std(0).round(6).tolist()} N。空载Fz约−0.658 N，并非零；09-27 official_zero_count=1，10-08为0，零偏和协议/采集会话差异可能影响力任务。未对两批人为归零，也未将无按压强制定义为零力。')
    p('所有帧stability_verified=false、optical_accuracy_verified=false；09-27的2419帧及10-08的100帧还标记unsettled/pair_difference_large。因此“stable_adc字段有效”只表示数值完整，不等于通过稳定性认证。遵循现有采集策略保留并说明质量标记，未静默删掉这些帧。')
    table(['位置标签','总数','训练','验证','测试','隔离帧'],[[i,int(np.sum(data['labels']==i))]+[int(np.sum(data['labels'][np.asarray(split[k])]==i)) for k in ['train','validation','test','purged']] for i in range(26)])
    heading('3. 模型结构与优化目标')
    table(['模块','Sensor Token注意力','稀疏CNN对照'],[['原始输入','B×9×5（九个真实光栅）','B×6×4×3（五通道+存在性Mask）'],['特征组织','5→48线性嵌入+可学习传感器身份编码','物理稀疏布局，空格填标准化后的0'],['共享backbone','2层Transformer Encoder；4头注意力；FFN=128；Pre-LN、GELU','保留旧CNN：Conv 6→24→48→48，3×3，ReLU'],['共享特征','九Token均值池化→Linear 48→128，GELU','Flatten→Linear 576→128，ReLU'],['Dropout','0.12','0.12'],['位置头','Linear 128→26 + softmax','Linear 128→26 + softmax'],['力头','Linear 128→3，无激活限制','Linear 128→3，无激活限制']])
    p('位置标签为互斥26维one-hot bit语义：第0维对应无按压，第1～25维对应位置编号。实现用整数标签0～25和加权CrossEntropy，其数学目标等价于one-hot交叉熵；不是26个独立sigmoid多标签任务。softmax概率之和为1。')
    p('力预测对Fx/Fy/Fz做回归，使用训练集真实力均值/标准差标准化后SmoothL1。总损失=位置交叉熵+力回归损失；若未来出现力缺失，保留分类样本、用mask屏蔽该帧回归，绝不补零。当前2520帧全部有有效力。')
    p('两个任务反向传播到同一个特征提取器；仅预测头不同。Sensor Token可用全局跨光栅注意力建模相关性；CNN有局部二维几何归纳偏置。数据少、会话漂移和标签分布会影响两者，不预设注意力一定优于CNN。')
    p('为公平比较，两个模型使用相同五通道标准化方式：每个五点通道跨训练样本及九个光栅统计均值/标准差。全部信号和力统计只拟合训练集，验证/测试仅变换。Sensor Token采用可学习光栅身份编码；CNN沿用原4×3布局与Mask。')
    heading('4. 划分与可复现训练条件')
    table(['划分','帧数','用途'],[['训练',len(split['train']),'拟合参数、归一化与类别权重'],['验证',len(split['validation']),'早停和选择checkpoint/种子'],['测试',len(split['test']),'只做最终指标计算，不用于参数拟合/选种子'],['隔离',len(split['purged']),'排除边界相邻帧，既不训练也不评估']])
    p('逐类按采集文件顺序：前60%训练，约5%隔离，约15%验证，约5%隔离，余下约15%测试；取整造成小幅比例差异。未随机交错相邻帧。标签1～19与0的三份仍来自同一连续段；20～25的训练主要来自第一段、验证与测试来自补采段，但验证和测试仍可同属一次按压。隔离区仅减少邻帧耦合，不能创造独立物理轮次。')
    p('统一设置：AdamW，学习率0.001，weight_decay=0.0001，batch=64；最大180轮，验证分数连续30轮未改善则早停；梯度范数裁剪1.0；类别权重=train_count/(26×class_train_count)；CPU Torch线程=2。种子20261008、20261009、20261010。')
    p('checkpoint/种子选择分数=验证位置准确率−0.05×验证标准化力MAE。最终交付模型是在三个种子中按该验证分数选择，不是按测试准确率挑选；没有在2520帧上重新拟合，以保留报告测试的可追溯性。')
    table(['模型','种子','最佳轮次','已跑轮次','验证准确率','测试准确率','测试向量MAE/N'],[[titles[r['architecture']],r['seed'],r['best_epoch'],r['epochs_completed'],scalar(r['metrics']['validation']['position_accuracy'],2,True),scalar(r['metrics']['test']['position_accuracy'],2,True),scalar(r['metrics']['test']['force_vector_mae_n'])] for r in comparison['runs']])
    doc.add_picture(str(chart_dir/'learning.png'),width=Inches(6.7))
    heading('5. 三维力指标与误差解释')
    table(['测试三轴指标','Sensor Token均值±SD','CNN均值±SD','常数力预测基线'],[[f'{axis} {metric}'+(' / N' if metric!='R²' else ''),agg(kinds[0],key,index=j),agg(kinds[1],key,index=j),scalar(baseline[key][j])] for j,axis in enumerate(['Fx','Fy','Fz']) for metric,key in [('MAE','force_mae_n'),('RMSE','force_rmse_n'),('R²','force_r2')]])
    p('MAE为平均绝对误差，RMSE对大误差更敏感；R²=1−SSE/SST，负值表示在该测试轴上不如“测试均值”参照。表内常数预测基线则使用训练力均值，严格不读取测试力拟合。向量MAE=mean(||预测力−真实力||₂)，不是三个分轴MAE简单相加。')
    doc.add_picture(str(chart_dir/'force_curves.png'),width=Inches(6.7))
    p('曲线使用按验证分数选出的单个checkpoint；跨标签、采集段或隔离帧断线。横轴是测试帧顺序，不是连续现场时间。力曲线重合程度可直观看效果，但不能替代MAE/RMSE/R²。')
    heading('6. 位置分类、空载与易混淆位置')
    doc.add_picture(str(chart_dir/'confusion.png'),width=Inches(6.7))
    doc.add_picture(str(chart_dir/'per_class.png'),width=Inches(6.7))
    rows=[]
    for label in range(26):
        row=[label,reports[kinds[0]]['metrics']['test']['per_class'][label]['support']]
        for k in kinds:
            c=reports[k]['metrics']['test']['per_class'][label]
            row.extend([scalar(c['precision'],2,True),scalar(c['recall'],2,True),scalar(c['f1'])])
        rows.append(row)
    table(['标签','测试数','Token精确率','Token召回率','Token F1','CNN精确率','CNN召回率','CNN F1'],rows)
    for k in kinds:
        cm=np.asarray(reports[k]['metrics']['test']['confusion_matrix']); pairs=[(int(cm[a,b]),a,b) for a in range(26) for b in range(26) if a!=b and cm[a,b]>0]; pairs.sort(reverse=True)
        p(titles[k]+'主要混淆（真实→预测）：'+('；'.join(f'{a}→{b}: {n}帧' for n,a,b in pairs[:10]) if pairs else '无'))
    p('0号测试仅15帧，且空载状态与会话完全绑定；即使空载精确率/召回率达到100%，也不能声称可靠的跨天接触判定。建议在每次完整按压轮次前后交错采集空载，统一归零流程并记录零偏、设备协议与温度。')
    heading('7. 模型体积、推理耗时及选中checkpoint')
    table(['模型','选中种子','参数量','单帧中位数/ms','P95/ms','模型文件/KB'],[[titles[k],reports[k]['seed'],reports[k]['parameters'],scalar(reports[k]['cpu_single_frame_latency_ms']['median']),scalar(reports[k]['cpu_single_frame_latency_ms']['p95']),f"{Path(comparison['selected_models'][k]['model_path']).stat().st_size/1024:.1f}"] for k in kinds])
    p('单帧CPU耗时：固定2线程，预热20次后测量200次，包含输入标准化与模型计算；不含模型载入、串口、子进程IPC和Qt重绘，不能直接解释为GUI交互延迟。所有六个模型保存后均重载，与原模型预测逐项一致性检查通过。')
    for k in kinds: p(titles[k]+': '+comparison['selected_models'][k]['model_path'])
    heading('8. 与前一次25类稀疏CNN的关系')
    table(['指标','旧25类CNN历史验证','本次Token测试（三种子均值）','能否直接比较'],[['位置准确率',scalar(old['position_accuracy'],2,True),scalar(accuracy_token,2,True),'不能：类别/划分不同'],['Fx MAE/N',scalar(old['force_mae_n'][0]),scalar(comparison['aggregate_test_metrics']['sensor_token']['force_mae_n']['mean'][0]),'不能直接归因架构'],['Fy MAE/N',scalar(old['force_mae_n'][1]),scalar(comparison['aggregate_test_metrics']['sensor_token']['force_mae_n']['mean'][1]),'不能直接归因架构'],['Fz MAE/N',scalar(old['force_mae_n'][2]),scalar(comparison['aggregate_test_metrics']['sensor_token']['force_mae_n']['mean'][2]),'不能直接归因架构'],['向量MAE/N',scalar(old['force_vector_mae_n']),scalar(force_token),'不能直接归因架构']])
    p(f"历史模型仅09-27的1～25号：训练{old['train_count']}帧、验证{old['validation_count']}帧、最佳第{old['best_epoch']}轮；采用逐类尾部20%开发验证。旧模型没有0号输出。本次训练包含空载且有独立验证/测试角色、隔离区、26类头；历史73.14%只能作为背景记录。真正同口径对照是本文重训的26类CNN，不把历史分数变化当作架构收益。")
    heading('9. 局限、建议与使用方式')
    p('现阶段最重要的提升路径不是继续追求同轮次高分，而是补采多个完整且随机顺序的轮次：每个会话覆盖0～25，0号在各按压之间交错出现，多个力方向/幅值，记录press_cycle_id/pass_id、归零状态、温度和设备协议；训练/验证/测试按完整轮次或会话隔离。建议至少5个完整轮次，3训练/1验证/1测试。')
    p('注意力权重不自动等于传感器重要性或物理因果关系；本模型学习的跨光栅关联仍需独立数据验证。当前结果只作为开发模型，尚未进行真实设备在线按压和新会话验收。')
    p('本次新增独立Torch推理接口，未修改第五页默认训练/加载逻辑，也未覆盖正在使用的旧模型。新格式fbg_touch26_v1须使用新的ModelBundle/serve入口，不能直接当旧25类模型格式加载。')
    p('命令（在项目Python目录运行）：')
    p('..\\.venv\\Scripts\\python.exe -m manual_touch_sensor_token --serve "'+comparison['selected_models']['sensor_token']['model_path']+'"')
    p('stdin每行JSON输入：{"adc": [同一帧45个稳定ADC值], "request_id": 1}；输出含position=0～25、confidence、26维probabilities及三轴force。0号是空载，不能传给仅接受1～25号的旧位置显示代码。')
    heading('10. 输出文件与审计追踪')
    p(str(OUT.resolve()))
    p('comparison.json：六次训练与均值/标准差；split.json：每个样本的训练/验证/测试/隔离索引；dataset.npz：从原始JSONL提取的紧凑输入；data_audit.json：来源/行号/标签/run/sample_index；readonly_dataset_audit.json：独立只读审计；各seed的.pt、.report.json和预测NPZ；figures/：混淆矩阵、曲线、逐类召回率、学习曲线；experiment_manifest.json：软件版本与源码哈希。')
    for source in audit['sources']:
        p('来源：'+source['path']); p('SHA-256：'+source['sha256'])
    doc.save(DOC)
    # Re-open actual package and confirm headings/table counts/data strings.
    check=Document(DOC); text='\n'.join([p.text for p in check.paragraphs]+[c.text for t in check.tables for r in t.rows for c in r.cells])
    for term in ['26类','2520','1512','383','会话','73.14','Sensor Token']:
        assert term in text,term
    validation={'docx_path':str(DOC.resolve()),'paragraphs':len(check.paragraphs),'tables':len(check.tables),'inline_figures':len(check.inline_shapes),'bytes':DOC.stat().st_size,'verified_data_terms':True}
    assert len(check.inline_shapes)==4
    (OUT/'docx_readback_verification.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
