import torch as th
from dataclasses import fields, is_dataclass
from typing import Any

def dataclass_to_loggable_dict(dc_instance) -> dict:
    if not is_dataclass(dc_instance):
        raise ValueError("Input must be a dataclass instance")

    result = {}
    
    for field in fields(dc_instance):
        value = getattr(dc_instance, field.name)
        if isinstance(value, th.Tensor):
            if value.ndim == 0:  # scalar
                result[field.name] = value.item()
            else:
                # Avoid .item() error for non-scalars
                result[field.name] = value.detach().cpu().numpy()
        else:
            result[field.name] = value  # could be float, int, or None, etc.

    return result