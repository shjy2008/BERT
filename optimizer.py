from typing import Callable, Iterable, Tuple
import math
import torch
from torch.optim import Optimizer

#### VLJ: For Task 1, step() need to be implemented !####

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]
                m_prev = state.get("m_prev", 0)
                v_prev = state.get("v_prev", 0)
                t = state.get("t", 1)

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                betas = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                # Update first and second moments of the gradients
                m_t = betas[0] * m_prev + (1 - betas[0]) * grad
                v_t = betas[1] * v_prev + (1 - betas[1]) * (grad * grad)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if correct_bias:
                    alpha_t = alpha * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                else:
                    alpha_t = alpha

                # Update parameters
                p.data = p.data - alpha_t * m_t / (torch.sqrt(v_t) + eps)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if weight_decay != 0:
                    p.data = p.data - alpha * weight_decay * p.data # This is equivalent to adding ((weight_decay/2) * weights^2) to loss function, because the derivative of this formula = weight_decay * weights

                state["m_prev"] = m_t
                state["v_prev"] = v_t
                state["t"] = t + 1
                

        return loss
