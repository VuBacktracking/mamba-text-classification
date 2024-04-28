import json
from dataclasses import field, asdict

class MambaConfig:
    d_model: int  = 2560
    n_layer: int = 64
    vocab_size: int  = 50277
    ssm_cfg: dict = field(default_factory = dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    padd_vocab_size_multiple: int = 8
    
    def to_json_string(self):
        return json.dumps(asdict(self))
    
    def to_dict(self):
        return asdict(self)