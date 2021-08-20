import time
import torch
import torch.autograd as autograd
import functools
import os, shutil
import numpy as np


def compute_noise(stoc_grads, true_grads):
    total_noise_sq = 0 
    total_grad_sq = 0
    for k in stoc_grads.keys():
        total_noise_sq += (stoc_grads[k]- true_grads[k]).norm(2).item() ** 2
        total_grad_sq += stoc_grads[k].norm(2).item() ** 2
    return total_noise_sq, total_grad_sq

def compute_norm(grads):
    total_grad_sq = 0
    for k in grads.keys():
        total_grad_sq += grads[k].norm(2).item() ** 2
    return total_grad_sq


def compute_l1norm(grads):
    total_grad_l1 = 0
    for k in grads.keys():
        total_grad_l1 += grads[k].norm(1).item()
    return total_grad_l1

def compute_linfnorm(grads):
    total_grad_inf = 0
    for k in grads.keys():
        total_grad_inf = max(total_grad_inf, torch.max(torch.abs(grads[k])))
    return total_grad_inf


def clone_grad(net, true_grads):
    for name, param in net.named_parameters():
        if param.grad is None:
            continue
        true_grads[name] = torch.clone(param.grad.data).detach()
        
def param_weights(net):
    weight_names = []
    weights = []
    for name, param in net.named_parameters():
        if param.requires_grad:
            weight_names.append(name)
            weights.append( param.norm(2).item())
    return weight_names, weights
        

def coord_noise(stoc_grads, true_grads):
    coordnoise = []
    for k in stoc_grads.keys():
        coordnoise.extend((stoc_grads[k]-
                           true_grads[k]).cpu().numpy().flatten().tolist()[:])
    coordnoise = np.array(coordnoise)
    return coordnoise


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))

def save_checkpoint(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))



def eigen_variance(net, criterion, dataloader, n_iters=10, tol=1e-2, verbose=False):
    n_parameters = num_parameters(net)
    v0 = torch.randn(n_parameters)

    Av_func = lambda v: variance_vec_prod(net, criterion, dataloader, v)
    mu = power_method(v0, Av_func, n_iters, tol, verbose)
    return mu


def eigen_hessian(net, dataloader, criterion, batches, n_iters=10, tol=1e-2, verbose=False):
    n_parameters = num_parameters(net)
    v0 = torch.randn(n_parameters)

    Av_func = lambda v: hessian_vec_prod(net, dataloader, criterion, batches, v)
    mu = power_method(v0, Av_func, n_iters, tol, verbose)
    return mu


def variance_vec_prod(net, criterion, dataloader, v):
    X, y = dataloader.X, dataloader.y
    Av, Hv, n_samples = 0, 0, len(y)

    for i in range(n_samples):
        bx, by = X[i:i+1].cuda(), y[i:i+1].cuda()
        Hv_i = Hv_batch(net, criterion, bx, by, v)
        Av_i = Hv_batch(net, criterion, bx, by, Hv_i)
        Av += Av_i
        Hv += Hv_i
    Av /= n_samples
    Hv /= n_samples
    H2v = hessian_vec_prod(net, criterion, dataloader, Hv)
    return Av - H2v


def hessian_vec_prod(net, dataloader, criterion, batches, v):
    Hv_t = 0
    n_batchs = batches #len(dataloader)
    for i, (bx, by) in enumerate(dataloader):
        Hv= Hv_batch(net, criterion, bx.cuda(), by.cuda(), v)
        Hv_t += Hv
        if i == batches - 1:
            break
    return Hv_t/n_batchs


def Hv_batch(net, criterion, batch_x, batch_y, v):
    """
    Hessian vector multiplication
    """
    net.zero_grad()
    net.eval()
    output = net(batch_x)
    loss = criterion(output, batch_y)

#     loss = criterion(logits, batch_y)
#     loss = loss.float().mean().type_as(loss)

    grads = autograd.grad(loss, net.parameters(), create_graph=True, retain_graph=True)
    idx, res = 0, 0
    for grad_i in grads:
        ng = torch.numel(grad_i)
        v_i = v[idx:idx+ng].cuda()
        res += torch.dot(v_i, grad_i.view(-1))
        idx += ng

    Hv = autograd.grad(res, net.parameters()).detach()
    net.zero_grad()
    Hv = [t.data.cpu().view(-1) for t in Hv]
    Hv = torch.cat(Hv)
    return Hv


def power_method(v0, Av_func, n_iters=10, tol=1e-3, verbose=False):
    mu = 0
    v = v0/v0.norm()
    for i in range(n_iters):
        time_start = time.time()

        Av = Av_func(v)
        mu_pre = mu
        mu = torch.dot(Av,v).item()
        v = Av/Av.norm()

        if abs(mu-mu_pre)/abs(mu) < tol:
            break
        if verbose:
            print('%d-th step takes %.0f seconds, \t %.2e'%(i+1,time.time()-time_start,mu))
    return mu


def num_parameters(net):
    """
    return the number of parameters for given model
    """
    n_parameters = 0
    for para in net.parameters():
        n_parameters += para.data.numel()

    return n_parameters




