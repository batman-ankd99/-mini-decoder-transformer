import matplotlib.pyplot as plt
for number, layer in enumerate(layers):
  if isinstance(layer, Tanh):
    t = layer.out.detach()
    t_mean = t.flatten().mean()
    t_std = t.flatten().std()
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach(), label=f'layer {number}')
    plt.legend()
    plt.title('activation distribution')
    print(f'Layer {number} : STD {t_std}, Mean {t_mean}')
