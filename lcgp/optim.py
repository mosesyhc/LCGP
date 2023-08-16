import torch
import importlib
import inspect

LS_FAIL_MAX = 3
PG_CONV_FLAG = 0


def which_lbfgs():
    """
    Determines the implementation of LBFGS.  FullBatchLBFGS implementation at
    github.com/hjmshi/PyTorch-LBFGS is recommended.
    """
    optim_list = \
        [name for name, _ in inspect.getmembers(importlib.import_module('torch.optim'),
                                                inspect.isclass)]
    if 'FullBatchLBFGS' in optim_list:
        lbfgs = torch.optim.FullBatchLBFGS
        fblbfgs = True
    elif 'LBFGS' in optim_list:
        lbfgs = torch.optim.LBFGS
        fblbfgs = False
    else:
        return ImportError('No LBFGS implementation found.')
    return lbfgs, fblbfgs


def custom_step(optim, fullbatch_flag, closure, loss,
                **kwargs):
    """
    Takes LBFGS step given implementation.
    """
    if fullbatch_flag:
        options = {'closure': closure, 'current_loss': loss,
                   'history_size': kwargs.get('history_size'),
                   'c1': kwargs.get('c1'), 'c2': kwargs.get('c2'),
                   'max_ls': kwargs.get('max_ls'), 'damping': False,
                   # 'ls_debug': True
                   }
        loss, grad, lr, _, _, _, _, _ = optim.step(options)
        d = optim.state['global_state'].get('d')
    else:
        loss = optim.step(closure)
        grad = optim._gather_flat_grad()
        lr = optim.state_dict()['param_groups'][0]['lr']
        d = optim.state_dict()['state'].get('d')
        if d is None:
            d = -grad
    return loss, grad, lr, d


def optim_lbfgs(model,
                maxiter=1000, lr=1e-1, history_size=4,
                max_ls=15, c1=1e-2, c2=0.9,
                pgtol=1e-5, ftol=2e-10,
                verbose=False):
    """
    Main optimization runner for LCGP model.

    :param model: LCGP class.
    :param maxiter: Maximum iteration of optimization steps to take.  Defaults to 1000.
    :param lr: Initial step size.  Defaults to 0.1.
    :param history_size:  History buffer kept for Hessian estimation.  Defaults to 4.
    :param max_ls: Maximum number of weak Wolfe line searches.  Defaults to 15.
    :param c1:  Constant for Armijo condition.  Defaults to 1e-2.
    :param c2:  Constant for weak Wolfe condition.  Defaults to 0.9.
    :param pgtol:  Projected gradient tolerance.  Defaults to 1e-5.
    :param ftol:  Absolute function difference tolerance.  Defaults to 2e-10.
    :param verbose:  Print progress if True.
    :return: LCGP class, number of iterations taken, and convergence flag.
    """
    def closure():
        optim.zero_grad(set_to_none=True)
        loss = model.loss()
        return loss

    lbfgs, fullbatch_flag = which_lbfgs()

    #  precheck learning rate
    while True:
        if not fullbatch_flag:
            optim = lbfgs(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                          max_iter=maxiter, history_size=history_size,
                          line_search_fn='strong_wolfe')
        else:
            optim = lbfgs(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                          dtype=torch.float64, debug=False)
        optim.zero_grad(set_to_none=True)
        loss = closure()
        loss.backward()

        loss, grad, lr, d = custom_step(optim, fullbatch_flag, closure, loss,
                                        history_size=history_size, c1=c1, c2=c2,
                                        max_ls=max_ls)

        if torch.isfinite(loss) and torch.isfinite(grad).all():
            break

        model.init_params()
        lr /= 10
    loss_prev = torch.inf
    epoch = 0
    ls_fail_count = 0
    reset_optim = False  # True if reset

    header = ['iter', 'grad.absmax()', 'pgrad.absmax()', 'lsigma2', 'lr',
              'neglpost', 'diff.']
    if verbose:
        print('{:<5s} {:<12s} {:<12s} {:<12s} {:<12s} '
              '{:<12s} {:<12s}'.format(*header))
    while True:
        loss, grad, lr, d = custom_step(optim, fullbatch_flag, closure, loss,
                                        history_size=history_size, c1=c1, c2=c2,
                                        max_ls=max_ls)
        if grad.isnan().any():
            flag = 'GRAD_NAN'
            print('exit after epoch {:d}, invalid gradient'.format(epoch)) 
            break

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
            if (lr > 1e-8) and \
                    ((loss_prev - loss) / torch.max(
                        torch.tensor((loss_prev.abs(), loss.abs(), torch.tensor(1, )))
                    )) <= ftol:
                print('exit after epoch {:d}, FTOL <= {:.3E}'.format(epoch, ftol))
                flag = 'F_CONV'
                break

        if ls_fail_count >= LS_FAIL_MAX:
            if not reset_optim:
                flag = 'LS_FAIL_MAX_REACHED'
                print('exit at epoch {:d}, line searches failed for '
                      '{:d} iterations'.format(epoch, LS_FAIL_MAX)) 
                break
            else:
                optim.__init__(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr)
                reset_optim = False
                ls_fail_count = 0
                print('reset optimizer')
        if verbose and epoch % 1 == 0:
            print('{:<5d} {:<12.3f} {:<12.3E} {:<12.3f} '
                  '{:<12.3E} {:<12.3f} {:<12.3f}'.format
                  (epoch, grad.abs().max(), pg.abs().max(), model.lsigma2s.max(),
                   lr, loss, loss_prev - loss))

        with torch.no_grad():
            loss_prev = loss.clone()
    return model, epoch, flag
