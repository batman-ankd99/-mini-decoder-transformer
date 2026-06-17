import torch
log_likelyhood = 0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelyhood += logprob
        n += 1
print(f'{log_likelyhood=}')
nll = -log_likelyhood
print(f'{nll=}')
print(f'{nll/n=}')       
