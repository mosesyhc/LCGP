import torch

LS_FAIL_MAX = 5

def optim_elbo_lbfgs(model,
                     maxiter=500, lr=1e-1,
                     max_ls=15, c1=1e-4, c2=0.9,
                     gtol=1e-1,
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
    ls_fail_count = 0
    reset_optim = 0

    header = ['iter', 'grad.absmax()', 'lr', 'negelbo', 'diff.']
    if verbose:
        print('{:<5s} {:<12s} {:<12s} {:<12s} {:<12s}'.format(*header))
    while True:
        options = {'closure': closure, 'current_loss': loss,
                   'history_size': 4,
                   'c1': c1, 'c2': c2,
                   'max_ls': max_ls, 'damping': True}
        loss, grad, lr, _, _, _, _, _ = optim.step(options)
        ls_fail_count += (lr < 1e-12)

        epoch += 1
        if epoch > maxiter:
            flag = 'MAX_ITER'
            print('exit after maximum epoch {:d}'.format(epoch))
            break
        if epoch >= 10:
            if grad.abs().max() <= gtol:
                # stopping rules relaxed
                print('exit after epoch {:d}, GTOL <= {:.3E}'.format(epoch, gtol))
                flag = 'G_CONV'
                break
        if ls_fail_count > LS_FAIL_MAX:
            if reset_optim > 0:
                flag = 'LS_FAIL_MAX_REACHED'
                print('exit at epoch {:d}, line searches failed for {:d} iterations'.format(epoch, LS_FAIL_MAX))
                break
            else:
                optim = torch.optim.FullBatchLBFGS(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                reset_optim += 1
                ls_fail_count = 0
                print('reset optimizer')
        if verbose:
            print('{:<5d} {:<12.3f} {:<12.3E} {:<12.3f} {:<12.3f}'.format
                  (epoch, grad.abs().max(), lr, loss, loss_prev - loss))

        with torch.no_grad():
            loss_prev = loss.clone()
    return model, epoch, flag
