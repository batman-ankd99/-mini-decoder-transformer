n = 9
adj = [
    [],
    [2,8],
    [1,3,4],
    [2],
    [2,5],
    [4,6],
    [5,7],
    [6,8],
    [1,7,9],
    [8]
]

from collections import deque

def bfs(n, adj, start):
  queue = deque()
  result = []
  visited = [0] * (n+1)
  queue.append(start)
  visited[start] = 1
  while len(queue) != 0:
    e = queue.popleft()
    result.append(e)
    for node in adj[e]:
      if visited[node] == 0:
        visited[node] = 1
        queue.append(node)
  return result


bfs(9, adj, 3)
