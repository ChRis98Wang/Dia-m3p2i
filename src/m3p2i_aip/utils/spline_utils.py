import numpy as np
import torch

# -------- 通用自然三次样条细化 (n 节点 -> L 细分) --------
def natural_cubic_refine(y_nodes: torch.Tensor,
                         subdivisions: list[int],
                         device=None) -> torch.Tensor:
    """
    y_nodes: (M, nu)  — M 个节点，每节点 nu 维
    subdivisions: len = M-1, 每个原区间要细分多少份
                   总长度 L = sum(subdivisions) + 1
    返回: (L, nu) 细化后的序列
    """
    if device is None:
        device = y_nodes.device
    y_np = y_nodes.detach().cpu().numpy()       # (M, nu)
    M, nu = y_np.shape
    h = np.ones(M-1, dtype=float)               # 等距 1
    # -------- 先算自然样条二阶导 M_i --------
    # 端点二阶导为 0 → 求 (M-2) 元三对角
    a = h[:-1]
    b = 2*(h[:-1]+h[1:])
    c = h[1:]
    d = 6*((y_np[2:]-y_np[1:-1])/h[1:,None] - (y_np[1:-1]-y_np[:-2])/h[:-1,None])
    # 解三对角
    def solve_tri(a,b,c,d):
        n = len(b)
        cp = np.zeros((n,nu)); dp = np.zeros((n,nu))
        cp[0] = c[0]/b[0]
        dp[0] = d[0]/b[0]
        for i in range(1,n):
            denom = b[i]-a[i-1]*cp[i-1]
            cp[i] = c[i]/denom if i<n-1 else 0.
            dp[i] = (d[i]-a[i-1]*dp[i-1])/denom
        x = np.zeros((n,nu))
        x[-1]=dp[-1]
        for i in range(n-2,-1,-1):
            x[i]=dp[i]-cp[i]*x[i+1]
        return x
    M_inner = solve_tri(a,b,c,d)
    M = np.zeros((M,nu))
    M[1:-1] = M_inner

    # -------- 细化采样 --------
    L = sum(subdivisions)+1
    y_fine = np.zeros((L,nu))
    t_out = np.zeros(L)
    idx = 0
    for i in range(M-1):
        m = subdivisions[i]
        x0=i; x1=i+1; h_i=1
        for k in range(m):
            x = x0 + k/m
            A=(x1-x)/h_i; B=(x-x0)/h_i
            term  = (M[i]*( (x1-x)**3 - h_i**2*(x1-x) )
                    +M[i+1]*( (x-x0)**3 - h_i**2*(x-x0) ))/(6*h_i)
            y_fine[idx]=A*y_np[i]+B*y_np[i+1]+term
            t_out[idx]=x
            idx+=1
    # 最后一个节点
    y_fine[idx]=y_np[-1]; t_out[idx]=M-1
    return torch.tensor(y_fine, device=device, dtype=y_nodes.dtype)
