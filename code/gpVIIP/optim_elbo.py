import torch
from optim_rules import convergence_f, convergence_g, convergence_f_abs


def optim_elbo_lbfgs(model,
                     maxiter=500, lr=1e-1,
                     gtol=1e-2,
                     thetate=None, fte=None,
                     verbose=False):

    optim = torch.optim.FullBatchLBFGS(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    def closure():
        model.compute_MV()
        optim.zero_grad(set_to_none=True)
        negelbo = model.negelbo()
        return negelbo
    loss_prev = torch.inf
    loss = closure()
    loss.backward()

    epoch = 0

    header = ['iter', 'grad.mean()', 'lr', 'negelbo', 'diff.', 'test mse']
    if verbose:
        print('{:<5s} {:<12s} {:<12s} {:<12s} {:<12s} {:<12s}'.format(*header))
        print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f} {:<12.3f} {:<12.3f}'.format
              (epoch, 0, lr, loss, loss_prev - loss, model.test_mse(thetate, fte)))
    while True:
        options = {'closure': closure, 'current_loss': loss,
                   'c1': 1e-2, 'c2': 0.7,
                   'max_ls': 10, 'damping': True}
        loss, grad, lr, _, _, _, _, _ = optim.step(options)

        epoch += 1
        if epoch > maxiter:
            flag = 'MAX_ITER'
            print('exit after maximum epoch {:d}'.format(epoch))
            break
        if epoch >= 10:
            if grad.abs().max() <= gtol:
                print('exit after epoch {:d}, GTOL <= {:.3E}'.format(epoch, gtol))
                flag = 'G_CONV'
                break
        if verbose and thetate is not None and fte is not None:
            print('{:<5d} {:<12.3f} {:<12.3f} {:<12.3f} {:<12.3f} {:<12.3f}'.format
                  (epoch, grad.abs().mean(), lr, loss, loss_prev - loss, model.test_mse(thetate, fte)))

        with torch.no_grad():
            loss_prev = loss.clone()
    return model, epoch, flag
