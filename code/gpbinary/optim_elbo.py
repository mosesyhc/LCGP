import torch
import optim_rules
from optim_rules import convergence_f, convergence_g


def optim_elbo(model, maxiter=500, lr=8e-3):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=lr)

    epoch = 0
    flag = None
    negelbo_prev = torch.inf
    while True:
        optim.zero_grad(set_to_none=True)
        negelbo = model.negelbo()
        negelbo.backward()
        optim.step()

        if convergence_f(negelbo_prev, negelbo):
            print('FTOL <= {:.3E}'.format(optim_rules.FTOL))
            flag = 'F_CONV'
            break
        elif convergence_g(model.parameters()):
            print('GTOL <= {:.3E}'.format(optim_rules.GTOL))
            flag = 'G_CONV'
            break
        elif epoch >= maxiter:
            flag = 'MAX_ITER'
            break

        epoch += 1
        negelbo_prev = negelbo.clone().detach()
    return model, epoch, flag