import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from functorch import vmap, grad, jacrev, make_functional
import functools
import warnings
import time
from scipy.io import savemat
torch.manual_seed(40)
warnings.filterwarnings('ignore')
# Device
device = torch.device('cpu')

# Model
class Plain(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.ln1 = nn.Linear(in_dim, h_dim).double()
        self.act1 = nn.Tanh()
        self.ln2 = nn.Linear(h_dim, h_dim).double()
        self.act2 = nn.Tanh()
        self.ln3 = nn.Linear(h_dim, out_dim).double()

    def forward(self, x):
        out = self.ln1(x)
        out = self.act1(out)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.ln3(out)
        return out

# needs. loss, training points, sign, interface, training, optimizer

def chebyshev_first_kind(dim, n):
    X = []
    x = []
    X = (np.mgrid[[slice(None, n), ] * dim])
    XX = np.cos(np.pi * (X + 0.5) / n)
    for i in range(len(X)):
        x.append(np.array(XX[i].tolist()).reshape(n ** dim, 1))
    return np.hstack(np.array(x))

def inner(N):
    X = chebyshev_first_kind(2, N)
    x = X[:,0:1]
    y = X[:,1:2]
    X_inner = np.hstack((x, y))
    return torch.tensor(X_inner, requires_grad=True, device=device)

def exact_u(x, y):
    u = (x**2 + y**2) / k
    return u

def boundary_d(N):
    cheby_point = np.linspace(-1, 1, N+2)[1:N+1].reshape(-1, 1)
    dumy_one = np.ones((N, 1))
    xx1 = np.hstack((cheby_point, -1.0 * dumy_one, dumy_one)) # 아래
    
    xx2 = np.hstack((-1.0 * dumy_one, cheby_point, dumy_one)) # 왼쪽 
    
    xx3 = np.hstack((dumy_one, cheby_point, dumy_one)) # 오른쪽
    
    xx4 = np.hstack((cheby_point, dumy_one, dumy_one)) # 위
    
    X_bd = np.vstack([xx1, xx2, xx3, xx4])
    x = X_bd[:, 0:1]
    y = X_bd[:, 1:2]
    g_d = exact_u(x, y)
    g_d = torch.tensor(g_d, requires_grad=False, device=device)
    ## U_bd: function values on the boundary, totally 4*N_inner points
    x = X_bd[:, 0:1]
    y = X_bd[:, 1:2]
    X_bd = np.hstack([x, y])
    return torch.tensor(X_bd, requires_grad=True, device=device), g_d

def gradient(y, x):
    grad = torch.autograd.grad(
    y, x,
    grad_outputs=torch.ones_like(y),
    retain_graph=True,
    create_graph=True
    )[0]
    return grad

def multi_dot(x, y):
    return torch.sum(torch.mul(x, y), dim=1, keepdim=True)

def loss_residual_function(func_params, x_inner):
    def f(x, func_params):
        output = func_model(func_params, x)
        return output.squeeze(0)    
    grad2_f = - k * (jacrev(grad(f)))(x_inner, func_params)
    dudX2 = (torch.diagonal(grad2_f))
    
    laplace = (dudX2[0] + dudX2[1])
    loss_Res = laplace + 4.0
    
    return loss_Res.flatten()

def loss_boundary_d_function(func_params, x_bd_d, g_d):
    def f(x, func_params):
        output = func_model(func_params, x)
        return output.squeeze(0)

    u_pred = f(x_bd_d, func_params)
    loss_b = u_pred - g_d
    
    return loss_b.flatten()

r_big = 0.5
N = 32
num_neuron = 20
k = 1
alpha = 1
model = Plain(2, num_neuron, 1).to(device)
func_model, func_params = make_functional(model)

x_inner = inner(N)
x_bd_d, g_d = boundary_d(N)

x_inner_valid = inner(3*N)
x_bd_d_valid, g_d_valid = boundary_d(3*N)

LM_iter = 300
mu_update = 2 # update \mu every mu_update iterations
div_factor = 1.3 # \mu <- \mu/div_factor when loss decreases
mul_factor = 3. # \mu <- mul_factor*\mu when loss incerases

mu = 10**5
loss_sum_old = 10**5
itera = 0

savedloss = []
savedloss_valid = []
start_time = time.time()
for step in range(LM_iter+1):
    # Put into loss functional to get L_vec
    L_vec_res = vmap(loss_residual_function, (None, 0))(func_params, x_inner)
    L_vec_b_d = vmap(loss_boundary_d_function, (None, 0, 0))(func_params, x_bd_d, g_d)
    L_vec_res = L_vec_res/np.sqrt(N**2)
    L_vec_b_d = L_vec_b_d/np.sqrt(4.0*N)
    loss = torch.sum(L_vec_res**2) + torch.sum(L_vec_b_d**2)

    L_valid_vec_res = vmap(loss_residual_function, (None, 0))(func_params, x_inner_valid)
    L_valid_vec_b_d = vmap(loss_boundary_d_function, (None, 0, 0))(func_params, x_bd_d_valid, g_d_valid)
    L_valid_vec_res = L_valid_vec_res/np.sqrt((3*N)**2)
    L_valid_vec_b_d = L_valid_vec_b_d/np.sqrt(3*4.0*N)
    loss_valid = torch.sum(L_valid_vec_res**2) + torch.sum(L_valid_vec_b_d**2)

    # Consturct J for domain points
    # (None, 0 ,0): func_params: no batch. data_d: batch wrt shape[0] (data[i, :]). force_value: batch wrt shape[0] (force_value[i,:])
    
    per_sample_grads = vmap(jacrev(loss_residual_function), (None, 0))(func_params, x_inner)
    cnt = 0
    for g in per_sample_grads: 
        g = g.detach()
        J_d_res = g.view(len(g), -1) if cnt == 0 else torch.hstack([J_d_res, g.view(len(g), -1)])
        cnt = 1
    
    per_sample_grads = vmap(jacrev(loss_boundary_d_function), (None, 0, 0))(func_params, x_bd_d, g_d)
    cnt = 0
    for g in per_sample_grads: 
        g = g.detach()
        J_d_b_d = g.view(len(g), -1) if cnt == 0 else torch.hstack([J_d_b_d, g.view(len(g), -1)])
        cnt = 1

    # cat J_d and J_b into J
    J_mat = torch.cat((J_d_res, J_d_b_d))
    L_vec = torch.cat((L_vec_res, L_vec_b_d))

    # update lambda
    I = torch.eye((J_mat.shape[1])).to(device)

    with torch.no_grad():
        J_product = J_mat.t()@J_mat
        
        rhs = -J_mat.t()@L_vec
        with torch.no_grad():
            dp = torch.linalg.solve(J_product + mu*I, rhs)

        # update parameters
        cnt=0
        for p in func_params:
            mm=torch.Tensor([p.shape]).tolist()[0]
            num=int(functools.reduce(lambda x,y:x*y,mm,1))
            p+=dp[cnt:cnt+num].reshape(p.shape)
            cnt+=num

        itera += 1
        if step % mu_update == 0:
            #if loss_sum_check < loss_sum_old:
            if loss < loss_sum_old:
                mu = max(mu/div_factor, 10**(-9))
            else:
                mu = min(mul_factor*mu, 10**(8))
            loss_sum_old = loss
                
        if step%100 == 0:
            print(
                    'Iter %d, Loss: %.5e, mu: %.5e' % (itera, loss.item(), mu)
                )     
            print(
                    'Iter %d, Loss_Valid: %.5e, mu: %.5e' % (itera, loss_valid.item(), mu)
                )     
        savedloss.append(loss.item())
        savedloss_valid.append(loss_valid.item())
end_time = time.time()
x_, y_ = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_, y_)
X, Y = X.flatten(), Y.flatten()
x, y = X.reshape(-1, 1), Y.reshape(-1, 1)

X_combine = torch.tensor(np.hstack((x, y)), device=device)
u = func_model(func_params, X_combine).detach().numpy()
# set up a figure twice as wide as it is tall

u_test = exact_u(x, y)

def surf_plot(x, y, u):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.scatter(x, y, u, c=u)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# surf_plot(x, y, u)
# surf_plot(x, y, u_test)

error = np.absolute(u - u_test)
error_u_inf = np.linalg.norm(error, np.inf)
print('Error u (absolute inf-norm): %e' % (error_u_inf))
error_u_2 = np.linalg.norm(error, 2) / 100
print('Error u (absolute 2-norm): %e' % (error_u_2))
# print(end_time-start_time)
