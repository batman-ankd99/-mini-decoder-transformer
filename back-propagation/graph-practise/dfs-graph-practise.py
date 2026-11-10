adj_list = [
    [],
    [2,4],
    [1,3,6],
    [2],
    [1,5,7],
    [4,8],
    [2],
    [4,8],
    [2],
    [4,8],
    [5,7]
]

start = 1
n = 8
result = []
visited = [0] * (n+1)
def dfs(node, adj, n):
  result.append(node)
  visited[node] = 1
  for n in adj[node]:
    if visited[n] == 0:
      dfs(n,adj,n)

  return result

dfs(start, adj_list, n)
