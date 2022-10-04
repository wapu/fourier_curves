import numpy as np
import torch
import time
import os
import pickle
from tqdm import tqdm
from scipy.spatial import distance_matrix


# Specify device to evaluate on
device = 'cuda'



# Apply MMD (Gretton et al) to two sample sets x and y,
# averaging over inverse multiquadratic kernels;
# approaches zero for two large samples from the same distribution
def multi_mmd(x, y, widths_exponents=[(0.5, 1), (0.2, 1), (0.2, 0.5)]):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2*xy, 0, np.inf)

    XX, YY, XY = (torch.zeros_like(xx), torch.zeros_like(xx), torch.zeros_like(xx))

    for C,a in widths_exponents:
        XX += C**a * ((C + dxx) / a)**-a
        YY += C**a * ((C + dyy) / a)**-a
        XY += C**a * ((C + dxy) / a)**-a

    return torch.mean(XX + YY - 2*XY)



# Generate a very large pool of paired data
# for the Approximate Bayesian Computation baseline
def prepare_ABC_samples(data_model, N=int(1e8)):
    print(f'Drawing {N:,} samples from "{data_model.name}" prior...', end=' ')
    t = time.time()
    x, y, = [], []
    for i in tqdm(range(int(N/1e4))):
        x.append(data_model.sample_prior(int(1e4)).astype(np.float32))
        y.append(data_model.forward_process(x[-1]).astype(np.float32))
    np.save(f'abc/{data_model.name}/x_huge', np.concatenate(x, axis=0))
    np.save(f'abc/{data_model.name}/y_huge', np.concatenate(y, axis=0))
    print(f'Done in {time.time()-t:.1f} seconds.')



# Sort the given paired data set (x, y) by how near y is to y_target
# and return the n nearest samples x from that ordering
def quantile_ABC(x, y, y_target, n=4000):
    print(f'Evaluating ABC to obtain {n:,} samples closest to {y_target[0]} from set of {len(y):,}...', end=' ')
    t = time.time()
    d = distance_matrix(y_target, y)[0]
    sort = np.argsort(d)[1:]
    sample = x[sort][:n]
    threshold = d[sort[n]]
    print(f'Done in {time.time()-t:.1f} seconds, tolerance is {threshold:.3f}.')
    return sample, threshold



# Apply the forward process to all samples x and average the
# Euclidean distance between the outcomes and y_target
def resimulation_error(data_model, y_target, x):
    y = data_model.forward_process(x.cpu().numpy())
    dists = torch.sum((y - y_target)**2, dim=1).sqrt()
    return dists.mean()



# Compute the correlation matrix of the x-posterior for a given y_target
# Posterior is estimated with ABC, either in threshold or in quantile mode
def correlation_conditional(data_model, y_target=None, n=4000):
    if y_target is None:
        y_target = data_model.test_condition

    try:
        # Check whether a posterior sample was already created
        with open(f'abc/{data_model.name}/posterior_y_target.pkl', 'rb') as f:
            y_target_old, sample = pickle.load(f)
        assert y_target_old == y_target

    except:
        # If no suitable posterior sample exists, make a new one
        print(f'Obtaining {n} conditional {data_model.name} shapes for {y_target} using {abc_mode} ABC...')
        y_target = np.array(y_target)

        # For cross shapes, use threshold ABC with threshold 0.05 (trade-off between precision and efficiency)
        if data_model.name == 'cross':
            xs = []
            ys = []
            while len(xs) < n:
                coords, y = data_model.generate_cross_shape(forward=True, target=y_target)
                d = np.sqrt(np.sum(np.square(y_target - y)))
                if d < 0.05:
                    x = data_model.fourier_coeffs(coords, n_coeffs=CrosssShapeModel.n_parameters//4)
                    xs.append(x)
                    ys.append(y)
                    print(f' {len(xs)} ', end='', flush=True)
                print('.', end='')
            print()
            xs = np.stack(xs)
            ys = np.stack(ys)
            sample = data_model.flatten_coeffs(xs)

        # For Lens shapes or in general, use more sample efficient quantile ABC
        # To do so, must first call prepare_ABC_samples() with sufficiently large N
        else:
            x, y = np.load(f'abc/{data_model.name}/x_huge.npy'), np.load(f'abc/{data_model.name}/y_huge.npy')
            sample, _ = quantile_ABC(x, y, [y_target], n=n)

        # Save the new posterior sample together with y_target
        with open(f'abc/{data_model.name}/posterior_y_target.pkl', 'wb') as f:
            pickle.dump((y_target, sample), f)

    # Compute correlation and return it
    corr = np.corrcoef(sample.T)
    # np.save(f'abc/{data_model.name}/correlation_y_target.npy', corr)
    return corr



# Compare unconditional samples from generative models
# to samples from the data prior, using MMD
# Results are averaged over 'n_runs' independent runs
# Each model in the list 'models' must have a '.name' attribute
def mmd_prior(models, data_model, n_runs=100, sample_size=4000):
    # Load prior
    prior = np.load(f'abc/{data_model.name}/x_huge.npy')

    # Reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare lists
    samples = {model.name:[] for model in models}
    mmds = {model.name:[] for model in models}
    times = {model.name:[] for model in models}

    # Perform runs
    for i in tqdm(range(n_runs)):
        # Ground truth sample and shared latent sample for all models
        x_gt = torch.tensor(prior[np.random.choice(prior.shape[0], sample_size, replace=False)], device=device)
        z = torch.randn(sample_size, data_model.n_parameters, device=device)
        # Generate samples from all models
        with torch.no_grad():
            for model in models:
                t = time.time()
                x = generate_samples(model, sample_size, condition=None, latent=z)
                times[model.name].append(time.time() - t)
                samples[model.name].append(x)
                mmds[model.name].append(multi_mmd(x, x_gt).item())

    # Save results
    with open(f'abc/{data_model.name}/mmd_prior_results.pkl', 'wb') as f:
        pickle.dump((mmds, times), f)

    # Print averaged results
    print('\nAverage over all runs:')
    for model in models:
        print(f"{model.name+':':45} {np.mean(mmds[model.name]]):.5f}     ({np.mean(times[model.name]]):.3f}s)")



# Compare conditional samples from generative models
# to samples from the relevant ABC posteriors, using MMD and resimulation error
# Results are averaged over 'n_runs' independent runs
# Each model in the list 'models' must have a '.name' attribute
def mmd_posterior(models, data_model, n_runs=1000, sample_size=4000):
    # Load prior over x and corresponding y values
    prior = np.load(f'abc/{data_model.name}/x_huge.npy')
    y = np.load(f'abc/{data_model.name}/y_huge.npy')

    # Reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare lists
    samples = {model.name:[] for model in models}
    mmds = {model.name:[] for model in models}
    times = {model.name:[] for model in models}
    resim = {model.name:[] for model in models}
    targets = []

    # Perform runs
    for i in range(n_runs):
        # Obtain ABC posterior sample to compare against
        try:
            with open(f'abc/{data_model.name}/{i:05}.pkl', 'rb') as f:
                y_target, x_gt, threshold = pickle.load(f)
            assert x_gt.shape[0] >= sample_size
        except:
            if not os.path.exists(f'abc/{data_model.name}'):
                os.mkdir(f'abc/{data_model.name}')
            y_target = data_model.forward_process(data_model.sample_prior(1)).astype(np.float32)
            x_gt, threshold = quantile_ABC(prior, y, y_target, n=sample_size)
            with open(f'abc/{data_model.name}/{i:05}.pkl', 'wb') as f:
                pickle.dump((y_target, x_gt, threshold), f)
        targets.append(y_target[0])
        x_gt = torch.from_numpy(x_gt).to(device)

        print(f'Run {i+1:04}/{n_runs:04} | y* = {np.round(y_target[0], 3)}')

        # Shared latent sample and target observation for all models
        z = torch.randn(sample_size, data_model.n_parameters, device=device)
        y_target = torch.tensor(y_target).to(device).expand(sample_size, data_model.n_observations)

        # Generate samples from all models
        with torch.no_grad():
            for model in models:
                t = time.time()
                x = generate_samples(model, sample_size, condition=y_target, latent=z)
                times[model.name].append(time.time() - t)
                samples[model.name].append(x)
                mmds[model.name].append(multi_mmd(x, x_gt).item())
                resim[model.name].append(resimulation_error(data_model, y_target, x).item())

    # Save results
    with open(f'abc/{data_model.name}/mmd_posterior_results.pkl', 'wb') as f:
        pickle.dump((targets, mmds, resim, times), f)

    # Print averaged results
    print('\nAverage over all runs:')
    for model in models:
        print(f"{model.name+':':45} {np.mean(mmds[model.name]]):.5f}     {np.mean(resim[model.name]]):.5f}     ({np.mean(times[model.name]]):.3f}s)")




if __name__ == '__main__':
    pass

    # prepare_ABC_samples(FourierCurveModel())
    # correlation_conditional(PlusShapeModel(), (0.75, 0.0, 1.0, 3.0), n=4000)

    # prepare_ABC_samples(LensShapeModel())
    # correlation_conditional(LensShapeModel(), (2.0, -1.0), n=4000)
