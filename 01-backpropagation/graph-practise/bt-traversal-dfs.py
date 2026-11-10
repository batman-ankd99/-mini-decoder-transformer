def dfs(node):
    if node is None:
        return
    print(node.val)
    dfs(node.left)
    dfs(node.right)

dfs(drinks)
