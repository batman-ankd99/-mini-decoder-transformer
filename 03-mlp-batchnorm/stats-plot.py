for number, layer in enumerate(layers):
  if isinstance(layer, Tanh):
    t = layer.out.detach()
    t_mean = t.flatten().mean()
    t_std = t.flatten().std()
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach(), label=f'layer {number}')

    print(f'Layer {number} : STD {t_std.item()}, Mean {t_mean.item()}')
plt.legend()
plt.title('activation distribution')
plt.show()

#case -2 activation gradients plot
for number, layer in enumerate(layers):
  if isinstance(layer, Tanh):
    t = layer.out.grad.detach()
    t_flat = t.flatten()
    t_mean = t.flatten().mean()
    t_std = t.flatten().std()
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach(), label=f'layer {number}')

    print(f'Layer {number} : STD {t_std.item()}, Mean {t_mean.item()}')
plt.legend()
plt.title('activation gradient distribution')
plt.show()


legends=[]
for i,p in enumerate(parameters):
  if p.ndim == 2:
      plt.plot([ud[j][i] for j in range(len(ud))])
      legends.append('param %d' % i)
plt.plot([0, len(ud)],[-3,-3], 'k')
plt.legend(legends)
