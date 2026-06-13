loss = 0
for y_actual, y_predic in zip(ys, predicted_out):
 # print((y_actual - y_predic)**2)
  loss += (y_actual - y_predic)**2
print(loss)
 # print(y_actual, y_predic)
