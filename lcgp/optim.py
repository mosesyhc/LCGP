import torch

LS_FAIL_MAX = 3
PG_CONV_FLAG = 0


def optim_lbfgs(model,
                maxiter=1000, lr=1e-1, history_size=4,
                max_ls=15, c1=1e-2, c2=0.9,
                pgtol=1e-6, ftol=2e-12,
                verbose=False):
    def closure():
        model.compute_aux_predictive_quantities()
        optim.zero_grad(set_to_none=True)
        nlp = model.neglpost()
        return nlp

    #  precheck learning rate
    while True:
        optim = torch.optim.FullBatchLBFGS(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                           # debug=True,
                                           dtype=torch.float64)
        optim.zero_grad(set_to_none=True)
        loss = closure()
        loss.backward()

        if torch.isnan(loss):
            print('go here')

        options = {'closure': closure, 'current_loss': loss,
                   'history_size': history_size,
                   'c1': c1, 'c2': c2,
                   'max_ls': max_ls, 'damping': False,
                   # 'ls_debug': True
                   }
        loss, grad, lr, _, _, _, _, _ = optim.step(options)
        if torch.isfinite(loss) and torch.isfinite(grad).all():
            break

        model.init_params()
        lr /= 10
    loss_prev = torch.inf
    epoch = 0
    ls_fail_count = 0
    reset_optim = False  # True if reset

    header = ['iter', 'grad.absmax()', 'pgrad.absmax()', 'lsigma2', 'lr', 'neglpost', 'diff.']
    if verbose:
        print('{:<5s} {:<12s} {:<12s} {:<12s} {:<12s} {:<12s} {:<12s}'.format(*header))
    while True:
        options = {'closure': closure, 'current_loss': loss,
                   'history_size': history_size,
                   'c1': c1, 'c2': c2,
                   'max_ls': max_ls, 'damping': True}
        loss, grad, lr, _, _, _, _, _ = optim.step(options)
        d = optim.state['global_state'].get('d')
        pg = d.dot(grad) / grad.norm() ** 2 * grad
        ls_fail_count += (lr < 1e-16)

        epoch += 1
        if epoch > maxiter:
            flag = 'MAX_ITER'
            print('exit after maximum epoch {:d}'.format(epoch))
            break
        if epoch >= 10:
            if pg.abs().max() <= pgtol:
                # stopping rules relaxed
                print('exit after epoch {:d}, PGTOL <= {:.3E}'.format(epoch, pgtol))
                flag = 'PG_CONV'
                break
            #  if line search fails, this criterion can be triggered
            if (lr > 1e-8) and ((loss_prev - loss) / torch.max(
                    torch.tensor((loss_prev.abs(), loss.abs(), torch.tensor(1, ))))) <= ftol:
                print('exit after epoch {:d}, FTOL <= {:.3E}'.format(epoch, ftol))
                flag = 'F_CONV'
                break

        if ls_fail_count >= LS_FAIL_MAX:
            if not reset_optim:
                flag = 'LS_FAIL_MAX_REACHED'
                print('exit at epoch {:d}, line searches failed for {:d} iterations'.format(epoch, LS_FAIL_MAX))
                break
            else:
                optim.__init__(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                reset_optim = False
                ls_fail_count = 0
                print('reset optimizer')
        if verbose and epoch % 10 == 0:
            print('{:<5d} {:<12.3f} {:<12.3E} {:<12.3f} {:<12.3E} {:<12.3f} {:<12.3f}'.format
                  (epoch, grad.abs().max(), pg.abs().max(), model.lsigma2s.max(), lr, loss, loss_prev - loss))

        with torch.no_grad():
            loss_prev = loss.clone()
    return model, epoch, flag
