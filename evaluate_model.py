import torch
import numpy as np

from data import prepare_data_loaders
from rejection_sampling import correlation_conditional, mmd_prior, mmd_posterior

# Specify device to evaluate on
device = 'cuda'

# Change this if another instance of Fourier shapes should be used
from data import LensShapeModel as data_model

# Specify whether trained model is conditional or unconditional
conditional = True
test_condition = data_model.test_condition
# conditional = False
# test_condition = None


# Load and prepare the trained model to be evaluated
def prepare_model():

    # YOUR
    # CODE
    # HERE

    return model.to(device)


# Draw N samples from the trained conditional density model
# for conditional y input (i.e. observation) 'condition',
# to be used once per observation during evaluation
def generate_samples(model, N, condition=None, latent=None):

    # YOUR
    # CODE
    # HERE

    return x


# Infer the corresponding latent variables 'z' and Jacobian
# log determinants 'jac' for the given data samples 'x'
def inference_and_jacobian(model, x, condition=None):

    # YOUR
    # CODE
    # HERE

    return z, jac




# Compute the maximum likelihood loss of the trained model on the test data
def evaluate(model, test_loader):
    with torch.no_grad():

        losses = []
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x += 0.01 * torch.randn_like(x)

            if conditional:
                z, jac = inference_and_jacobian(model, x, condition=y)
            else:
                z, jac = inference_and_jacobian(model, x)

            loss = 0.5 * torch.sum(z**2, dim=1).mean() - jac.mean()
            losses.append(loss.item())

        return np.mean(losses)


# Evaluate the trained model in terms of log-likelihood
# Mode can be 'standard' or 'hint'
def evaluate_log_likelihood(model, test_loader, mode='standard'):
    # Evaluate likelihood
    n_dims = data_model.n_parameters
    if mode == 'standard':
        likelihood = - evaluate(model, test_loader, conditional) - np.log(2*np.pi) * (n_dims/2)
    elif mode == 'hint_paper':
        likelihood = - evaluate(model, test_loader, conditional) / n_dims
    else:
        likelihood = None

    print(f'Test log-likelihood: {likelihood:.3f}')


# Evaluate the trained model in terms of maximum mean discrepancy (MMD)
# between model samples and true prior/posterior
def evaluate_mmd(model, condition=None):
    if condition is None:
        mmd_prior([model], data_model)
    else:
        mmd_posterior([model], data_model)


# Evaluate the trained model in terms of correlation structure in the x domain
def evaluate_parameter_correlations(model, condition=None, N=4000 if conditional else 10000)
    # Generate sample and compute parameter correlations
    with torch.no_grad():
        x = generate_samples(model, N, condition=condition).cpu().numpy()
    corr = np.corrcoef(x.T)

    # Estimate true parameter correlations from data set
    if not conditional:
        corr_true = np.corrcoef(data_model.sample_prior(N, flat=True).T)
    else:
        corr_true = correlation_conditional(data_model, condition, N)

    # Calculate the mean squared error
    corr_mse = np.nanmean(np.square(corr - corr_true))
    print(f'Parameter correlation MSE: {corr_mse:.4f}')


# Evaluate the trained model in terms of geometric fidelity of generated shapes
def evaluate_generated_shapes(model, condition=None, N=1000):
    # Fetch correct eval script for data model
    if data_model.name == 'lens':
        from evaluate_shapes import evaluate_lens_shape as eval_shape
    elif data_model.name == 'cross:
        from evaluate_shapes import evaluate_cross_shape as eval_shape
    else:
        return

    # Generate data sample and apply
    with torch.no_grad():
        x = generate_samples(model, N, condition=condition).cpu().numpy()
    eval_shape(x)



if __name__ == '__main__':
    pass

    model = prepare_model()
    test_loader = prepare_data_loaders(data_model, 10000, 10000, 1000)[1]

    evaluate_log_likelihood(model, test_loader)
    evaluate_mmd(model, condition=test_condition)
    evaluate_parameter_correlations(model, condition=test_condition)
    evaluate_generated_shapes(model, condition=test_condition)
