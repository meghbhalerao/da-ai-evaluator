def merge_dicts_strict(d1: dict, d2: dict) -> dict:
    overlap = d1.keys() & d2.keys()
    if overlap:
        raise ValueError(f"Duplicate keys found: {overlap}")
    return {**d1, **d2}