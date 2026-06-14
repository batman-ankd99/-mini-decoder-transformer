import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        plt.text(j,i,chstr, ha ="center", va="bottom", color='gray')
        plt.text(j,i, N[i, j].item(), ha="center", va="top", color="gray")

plt.axis('off')
