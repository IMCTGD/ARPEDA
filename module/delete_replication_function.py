import numpy as np

def delete_replication_function(A):
    if not isinstance(A, (list, tuple, np.ndarray)):
        raise ValueError(f"A 必须是 list, tuple 或 np.ndarray，收到类型 {type(A)}")
    A = np.asarray(A)
    # print(f"delete_replication_function: A shape = {A.shape}, values = {A}")
    if A.size == 0:
        return A
    A = A.flatten()
    B = np.unique(A)
    return B