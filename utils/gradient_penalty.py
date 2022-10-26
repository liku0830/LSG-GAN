from utils.perceptual_loss import *
from losses import *
from torch.autograd import Variable
import torch.autograd as autograd
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



def compute_gradient_penalty(D, X):
    """Calculates the gradient penalty loss for DRAGAN"""
    # Random weight term for interpolation
    alpha = Tensor(np.random.random(size=X.shape)).cuda()

    interpolates = alpha * X + ((1 - alpha) * (X + 0.5 * X.std() * torch.rand(X.size()).cuda()))
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = D(interpolates)

    fake = Variable(Tensor(X.shape[0], 1, 30, 20).fill_(1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    lambda_gp = 15
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty