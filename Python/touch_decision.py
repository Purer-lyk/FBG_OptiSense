"""Torch-free 26-class/contact agreement and explicit abstention."""
import math

DEFAULT_POLICY={'press_confidence':.65,'press_contact':.8,'press_margin':.10,'no_press_confidence':.60,'no_press_contact_max':.20}


def decide_contact(probabilities,contact_probability=None,policy=None):
    p=[float(v) for v in probabilities]
    if len(p)!=26 or not all(math.isfinite(v) and 0<=v<=1 for v in p):
        raise ValueError('Expected 26 finite mutually exclusive probabilities')
    if abs(sum(p)-1)>1e-4: raise ValueError('Probabilities must sum to 1')
    cfg=dict(DEFAULT_POLICY); cfg.update(policy or {})
    ranked=sorted(range(26),key=lambda i:p[i],reverse=True); raw=ranked[0]; confidence=p[raw]; margin=confidence-p[ranked[1]]
    contact=1-p[0] if contact_probability is None else float(contact_probability)
    if not math.isfinite(contact) or not 0<=contact<=1: raise ValueError('Invalid contact probability')
    status='uncertain'; position=None; reason='low_confidence_or_contact_disagreement'
    if raw==0 and confidence>=cfg['no_press_confidence'] and contact<=cfg['no_press_contact_max']:
        status='no_press'; position=0; reason='classification_and_contact_agree_no_press'
    elif raw>0 and confidence>=cfg['press_confidence'] and margin>=cfg['press_margin'] and contact>=cfg['press_contact']:
        status='pressed'; position=raw; reason='classification_and_contact_agree_pressed'
    return {'position':position,'raw_position':raw,'confidence':confidence,'contact_probability':contact,'no_press_probability':p[0],'probability_margin':margin,'decision_status':status,'decision_reason':reason}
