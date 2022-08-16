import torch
import time
from optim_rules import convergence_f, convergence_g, convergence_f_abs


def optim_elbo_lbfgs(model,
                     maxiter=250, lr=1e-3,
                     ftol=None):

    if ftol is None:
        ftol = model.n / 1e4
    optim = torch.optim.FullBatchLBFGS(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    def closure():
        model.compute_MV()
        optim.zero_grad(set_to_none=True)
        negelbo = model.negelbo()
        return negelbo
    loss_prev = torch.inf
    loss = closure()
    loss.backward()
    # raise

    epoch = 0
    while True:
        options = {'closure': closure, 'current_loss': loss}
        loss, grad, lr, _, _, _, _, _ = optim.step(options)

        epoch += 1
        if epoch > maxiter:
            flag = 'MAX_ITER'
            break
        # if epoch >= 3:
        if convergence_f_abs(loss_prev, loss, ftol=ftol):
            print('exit after epoch {:d}, FTOL <= {:.3E}'.format(epoch, ftol))
            flag = 'F_CONV'
            break

        loss_prev = loss.clone().detach()

    return model, epoch, flag


# print(epoch, grad, loss)
# for p in model.parameters():
#     print(p)
# if epoch % 1 == 0:
#     with torch.no_grad():
#         model.create_MV()
#         trainmse = model.test_mse(thetatr, ftr)
#         mse = model.test_mse(thetate, fte)
#
#         print('{:<5d} {:<12.3f} {:<12.3f} {:<12.6f} {:<12.6f}'.format
#               (epoch, loss, loss_prev - loss, mse, trainmse))

def optim_elbo(model,
               # ftr, thetatr, fte, thetate,
               maxiter=2500, lr=8e-3):
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr)

    epoch = 0
    flag = None
    negelbo_prev = torch.inf
    while True:
        optim.zero_grad(set_to_none=True)   # from guide: Alternatively, starting from PyTorch 1.7, call model or optimizer.zero_grad(set_to_none=True).
        negelbo = model.negelbo()
        if torch.isnan(negelbo):
            print('go here')
            negelbo = model.negelbo()
            break
        negelbo.backward()
        optim.step()
        #
        # if epoch % 10 == 0:
        #     with torch.no_grad():
        #         model.create_MV()
        #         trainmse = model.test_mse(thetatr, ftr)
        #         mse = model.test_mse(thetate, fte)
        #
        #         print('{:<5d} {:<12.3f} {:<12.6f} {:<12.6f}'.format
        #               (epoch, negelbo, mse, trainmse))

        if convergence_f(negelbo_prev, negelbo, ftol=1e-6):
            print('FTOL <= {:.3E}'.format(1e-6))
            flag = 'F_CONV'
            break
        elif convergence_g(model.parameters(), gtol=1e-03):
            print('GTOL <= {:.3E}'.format(1e-03))
            flag = 'G_CONV'
            break
        elif epoch >= maxiter:
            flag = 'MAX_ITER'
            break

        epoch += 1
        negelbo_prev = negelbo.clone().detach()
    return model, epoch, flag
