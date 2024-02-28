import megatron_utils.parallel_state as parallel_state
import megatron_utils.tensor_parallel
import megatron_utils.utils

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
    "utils",
]