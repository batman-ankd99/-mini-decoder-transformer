for k in range(20):
  #forward pass
  ypred = []
  for x in xs:
    ypred.append(n(x))

  loss = 0
  for y_actual, y_predic in zip(ys, ypred):
    loss += (y_actual - y_predic)**2
  loss

  #backward pass
  for p in n.paramters():
    p.grad = 0.0 #as in constructor grad is getting reset but when we call backward( it just accumulates all grad from previous and new run
  loss.backward()

  #update - Gradient Descent
  for p in n.parameters():
    p.data += -0.05 * p.grad

  print(k, loss.data )

    
