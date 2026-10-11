
import matplotlib.pyplot as plt
import torch

plt.figure(figsize=(10, 6))

for i, layer in enumerate(layers):
    if isinstance(layer, Tanh):
        out = layer.out.detach().flatten()

        # Mean and standard deviation
        mean = out.mean().item()
        std = out.std().item()

        # Percentage of saturated activations
        saturated = ((out > 0.97) | (out < -0.97)).float().mean().item() * 100

        print(
            f"Layer {i}: "
            f"mean = {mean:.4f}, "
            f"std = {std:.4f}, "
            f"saturated = {saturated:.2f}%"
        )

        # Histogram counts for line plot
        counts = torch.histc(out, bins=50, min=-1, max=1)
        x = torch.linspace(-1, 1, 50)

        plt.plot(x.numpy(), counts.numpy(), label=f"Layer {i}")

plt.xlabel("Tanh output value")
plt.ylabel("Number of activations")
plt.title("Activation Distribution Across Tanh Layers")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
