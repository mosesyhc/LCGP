import torch
import optim_rules
from optim_rules import convergence_f, convergence_g


def optim_elbo(model, ftr, thetatr, fte, thetate, maxiter=2500, lr=8e-3):
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr)

    epoch = 0
    flag = None
    negelbo_prev = torch.inf
    while True:
        optim.zero_grad(set_to_none=True)  # from guide: Alternatively, starting from PyTorch 1.7, call model or optimizer.zero_grad(set_to_none=True).
        negelbo = model.negelbo()
        negelbo.backward()
        optim.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                model.create_MV()
                trainmse = model.test_mse(thetatr, ftr)
                mse = model.test_mse(thetate, fte)

                print('{:<5d} {:<12.3f} {:<12.6f} {:<12.6f}'.format
                      (epoch, negelbo, mse, trainmse))

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