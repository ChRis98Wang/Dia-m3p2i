import torch

# 定义参数
sigma0 = 1e-2
sigma1 = 1.0
Ndiffuse = 2
horizon_diffuse_factor = 0.9
Hnode = 11

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

A = sigma0
B = torch.log(torch.tensor(sigma1 / sigma0, device=device)) / Ndiffuse

# 在GPU上计算sigmas
sigmas = A * torch.exp(B * torch.arange(Ndiffuse, device=device, dtype=torch.float32))
t_range = torch.arange(Hnode + 1, device=device, dtype=torch.float32)
sigma_control = horizon_diffuse_factor ** (Hnode - t_range)
# 在GPU上计算sigma_control
# 使用flip进行逆序，而非[::-1]
#sigma_control = horizon_diffuse_factor ** torch.flip(torch.arange(Hnode + 1, device=device, dtype=torch.float32), dims=[0])
print(B)
print(sigmas)
print(sigma_control)
A = torch.ones(12,9)
print("A:",A)
B= torch.zeros(200,12,9)
C=A+B
print("C:",C.shape)
def reverse_once(i,action):
    print("the action is ", action,i)
    action_seq=action + i
    print("the new action is", action_seq)

    return action
def reverse(Ndiffe,):
    init_action = 2
    for i in range(Ndiffe):
        reverse_once(init_action,i)

reverse(4)
