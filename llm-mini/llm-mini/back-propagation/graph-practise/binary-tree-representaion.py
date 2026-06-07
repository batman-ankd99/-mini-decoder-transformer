class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

## Creating all nodes first

drinks = Node("drinks")
hot = Node("hot")
coffee = Node("coffee")
cold = Node("cold")
tea = Node("tea")
cola = Node("cola")
fanta = Node("fanta")

## Lets connect nodes

drinks.left = hot ## putting complete node and not just string, so not in quotes
drinks.right = cold
hot.left = coffee
hot.right = tea
cold.left = cola
cold.right = fanta

print(drinks.left.left.val)
