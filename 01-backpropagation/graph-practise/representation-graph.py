n = 5
import numpy as np
matrix = np.zeros((n+1, n+1))
matrix

edges = [[1,2],[2,4],[3,4],[1,3],[3,5],[5,4]]

## Adjancy Matrix
for u,v in edges:
  matrix[u][v] = 1
  matrix[v][u] = 1
print(matrix)

## Adjancy List
edges = [[1,2],[2,4],[3,4],[1,3],[3,5],[5,4]]
n = 5
mat = []
for n in range(n+1):
  mat.append([])
print(mat)


for u,v in edges:
  mat[u].append(v)
  mat[v].append(u)

print(mat)
