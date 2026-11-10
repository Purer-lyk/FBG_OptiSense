"""Torch-free model discovery for the Qt process."""
import json
from pathlib import Path


def find_latest_touch_model(output_root):
    root=Path(output_root)
    candidates=[]
    for comparison in root.glob('model_comparisons/*/comparison.json'):
        try:
            metadata=json.loads(comparison.read_text(encoding='utf-8'))
            selected=metadata['selected_models']['sensor_token']
            model=Path(selected['model_path'])
            if not model.is_file():
                model=comparison.parent/model.name
            if model.is_file() and model.with_suffix('.report.json').is_file():
                candidates.append(model.resolve())
        except (KeyError,ValueError,TypeError,OSError):
            continue
    candidates.extend(p.resolve() for p in root.glob('manual_touch_sequences/*/models/manual_touch_sparse_cnn.pt') if p.with_suffix('.report.json').is_file())
    return max(candidates,key=lambda p:p.stat().st_mtime_ns) if candidates else None


def model_description(report):
    kind=report.get('architecture')
    if report.get('schema') in ('fbg_touch_temporal26_v1', 'fbg_touch_separate26_v1'):
        config=report.get('config') or report.get('configuration') or {}
        title={'tcn':'TCN','sensor_token':'Sensor Token','sparse_cnn':'稀疏CNN','inception_1d':'Inception 1D'}.get(kind,str(kind))
        force_mode='历史反馈仅用于力回归' if config.get('force_history') else '力回归仅当前帧'
        return f'{title}：输入B×9×5实测dBm；位置仅当前45点，26类（0空载 / 1～25按压）；位置/力独立骨干、优化与选模。{force_mode}；实时ADC按模型跨阻档位转换，缺失参考力不补零。'

    if kind=='sensor_token_dbm':
        config=report.get('configuration') or {}
        layers=config.get('layers','?'); heads=config.get('heads','?')
        return f'Sensor Token dBm注意力模型：9个真实光栅，每光栅5维实测dBm，输入 B×9×5；{layers}层{heads}头自注意力，共享有序展平特征；26类位置分类（0=无按压）与Fx/Fy/Fz回归。实时ADC先按模型保存的跨阻档位转换为dBm，再复用训练标准化。'
    if kind=='sensor_token':
        return 'Sensor Token 注意力模型：9个真实光栅，每光栅5维实测值；输入 B×9×5。共享两层4头自注意力backbone；26类Softmax位置分类（0=无按压，1～25=按压位置）与Fx/Fy/Fz回归双头。'
    if kind=='sparse_cnn26':
        return '稀疏CNN（26类）：9个光栅按4×3布局，五个信号通道+存在性Mask，输入 B×6×4×3；共享卷积backbone，26类位置分类（0=无按压）与Fx/Fy/Fz回归。'
    return '稀疏CNN（25类）：9个光栅按4×3布局，五个信号通道+存在性Mask，输入 B×6×4×3；共享卷积backbone，1～25号位置分类与Fx/Fy/Fz回归（不支持无按压类别）。'
