import torch

def data_parser(
    array: torch.tensor=None,
    offset: float=1,
    scale: float=1,
    rain_th: float=0.01,
    rain2dbz: bool=True,
    min_value: float=None,
    *args,
    **kwargs
    ):
    array = array / scale + offset
    if kwargs:
        for k, v in kwargs.items():
            exec(f"{k}={v}")
    if rain2dbz:
        array[array>rain_th] = torch.log10(((array[array>rain_th] ** (8 / 5)) * 200)) * 10
        array[array<=rain_th] = min_value
    return array

def min_max_normalizer(
    array: torch.tensor=None,
    min_value: float=None,
    max_value: float=None,
    *args,
    **kwargs
    ):
    if kwargs:
        for k, v in kwargs.items():
            exec(f"{k}={v}")
    return (array - min_value) / (max_value - min_value)