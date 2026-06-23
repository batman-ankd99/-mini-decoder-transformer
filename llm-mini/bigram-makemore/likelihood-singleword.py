logprob = 0
n =0
for w in ["andrej"]:
  chs = ['.'] + list(w) + ['.']
  for ch1,ch2 in zip(chs,chs[1:]):
    ix1 = stoi[ch1]
    iy1 = stoi[ch2]
    logprob += torch.log(P[ix1,iy1])
    n += 1

print("logprob in andrej : ", logprob.item())
print("NLL in andrej : ", -logprob.item())
print("count of bigrams :  ", n)
print("avg NLL in andrej : ", -logprob.item()/n) 
