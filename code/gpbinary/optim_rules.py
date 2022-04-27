import torch

FTOL = 1e-04
GTOL = 1e-03


def convergence_f(f_prev, f_curr):
    with torch.no_grad():
        f_prev = torch.tensor((f_prev,))
        f_curr = torch.tensor((f_curr,))
        d = f_curr - f_prev
        return torch.abs(d) / torch.max(torch.tensor((torch.abs(f_prev), torch.abs(f_curr), torch.tensor((1,))))) <= FTOL


def convergence_g(param_generator):
    g = [p.grad for p in param_generator]
    # print('max g: ', torch.max(torch.abs(g[0])))
    return torch.max(torch.abs(g[0])) <= GTOL
