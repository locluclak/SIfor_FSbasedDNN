import torch
from captum.attr import IntegratedGradients
import numpy as np
import model_train




# Custom Integrated Gradients
def custom_integrated_gradients(model, input_samples, baseline, target, n_steps=50, device = torch.device("cpu")):
    input_samples = input_samples.clone().to(device).requires_grad_(True)
    baseline = baseline.to(device)  # Ensure baseline is on the same device
    attributions = torch.zeros_like(input_samples).to(device)
    alphas = torch.linspace(0, 1, n_steps, device=device)

    model.eval()
    for alpha in alphas:
        # Interpolate between baseline and input
        interpolated = baseline + alpha * (input_samples - baseline)
        interpolated.requires_grad_(True)
        # Compute model output
        outputs = model(interpolated)

        # Select the target class logits for gradient computation
        target_outputs = outputs[torch.arange(outputs.size(0)), target]

        # Compute gradients with respect to the interpolated input
        gradients = torch.autograd.grad(
            target_outputs.sum(), interpolated, create_graph=True
        )[0]

        # Accumulate gradients
        attributions += gradients / n_steps
    # Scale attributions by (input - baseline)
    attributions = attributions * (input_samples - baseline)
    return attributions


# Integrated Gradients using Captum
def compute_integrated_gradients(model, X_test, feature_names, n_steps=50,device = torch.device("cpu")):
    model.eval()
    model.to(device)
    ig = IntegratedGradients(model)

    input_samples = X_test.to(device).requires_grad_(True)
    baseline = torch.zeros_like(input_samples).to(device)

    # Get model predictions to use as targets
    with torch.no_grad():
        outputs = model(input_samples)
        targets = torch.argmax(outputs, dim=1)

    # Compute attributions with targets
    attributions = ig.attribute(
        input_samples, baseline, target=targets, n_steps=n_steps
    )
    return attributions


# Get significant features
def get_threshold_attributions(values, percentile) -> np.ndarray:
    assert 0 <= percentile <= 100, "Percentile must be between 0 and 100 inclusive."
    values = np.abs(values)
    flat = values.flatten()
    sorted_vals = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_vals)
    # print("Cumulative sum: ", cumsum)
    # print("Percntile: ", cumsum[-1] * percentile / 100.0)
    cutoff_index = np.searchsorted(cumsum, cumsum[-1] * percentile / 100.0)
    # print("Cutoff index: ", cutoff_index)
    cutoff_index = min(cutoff_index, len(sorted_vals) - 1)
    threshold = sorted_vals[cutoff_index]
    # print("Threshold: ", threshold)
    significant_features = values >= threshold
    return np.where(significant_features)[0]


if __name__ == "__main__":
    model, test_data = model_train.gendata_trainmodel().values()
    X_test, y_test = test_data
    print(X_test.shape)
    # X_test = X_test.to(device)  # Ensure X_test is on device
    with torch.no_grad():
        outputs = model(X_test)
        targets = torch.argmax(outputs, dim=1)
    print("Xtest: ", X_test[:1])
    custom_attributions = custom_integrated_gradients(
        model, X_test, torch.zeros_like(X_test), target=targets, n_steps=50
    )

    avg_custom_attributions = (
        torch.mean(custom_attributions, dim=0).cpu().detach().numpy()
    )

    print(f"Custom attribution: {avg_custom_attributions}")

    M = get_threshold_attributions(avg_custom_attributions, percentile=80)
    print(f"Significant features (above 80th percentile): {M}")
