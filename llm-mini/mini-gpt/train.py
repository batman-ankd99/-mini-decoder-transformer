
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

for iter in range(max_iters):
  if iter % 300 == 0:
    losses = estimate_loss()
    print(f"step {iter}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")

  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
