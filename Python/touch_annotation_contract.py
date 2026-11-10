"""Torch-free annotation guards for mutually exclusive legacy training readers."""
import json
from pathlib import Path


def require_single_position_annotation(metadata):
    """Fail closed for every multi-mode record, including empty/one-point sets."""
    if (metadata.get('dataset_kind') == 'manual_touch_delta9'
            or metadata.get('input_contract') == 'delta_wavelength9'
            or metadata.get('feature_kind') == 'delta_wavelength9'
            or metadata.get('schema') == 'fbg-touch-delta9/v1'):
        raise ValueError('Unsupported delta_wavelength9 / 9峰Δλ采集数据：当前旧训练只支持45点输入；需单独适配训练。')
    if (metadata.get('annotation_mode') == 'multi'
            or metadata.get('dataset_kind') == 'manual_touch_multi45'
            or str(metadata.get('position_label_schema', '')).startswith('multi_hot')
            or 'position_multihot' in metadata or 'position_labels' in metadata):
        raise ValueError('Unsupported multi-contact annotation: 当前训练仅支持单点/空载互斥标签；请使用单点数据集。')


def require_single_position_session(directory):
    """Inspect declared task before cache access or parsing any training samples."""
    manifest = Path(directory) / 'manifest.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        require_single_position_annotation(data)
        require_single_position_annotation(data.get('metadata') or {})
