from typing import Callable, List

class Node:
    def __init__(self, *children):
        self.children = list(children)

def get_children(node: Node) -> List[Node]:
    return node.children

def topological_sort(node: Node, get_children: Callable) -> List[Node]:
    '''
    Return a list of node's descendants in reverse topological order from future to past (i.e. `node` should be last).

    Should raise an error if the graph with `node` as root is not in fact acyclic.
    '''
    result: List[Node] = [] # stores the list of nodes to be returned (in reverse topological order)
    perm: set[Node] = set() # same as `result`, but as a set (faster to check for membership)
    temp: set[Node] = set() # keeps track of previously visited nodes (to detect cyclicity)

    def visit(cur: Node):
        '''
        Recursive function which visits all the children of the current node, and appends them all
        to `result` in the order they were found.
        '''
        if cur in perm:
            return
        if cur in temp:
            raise ValueError("Not a DAG!")
        temp.add(cur)

        for next in get_children(cur):
            visit(next)

        result.append(cur)
        perm.add(cur)
        temp.remove(cur)

    visit(node)
    return result

def sorted_computational_graph(tensor: 'Tensor') -> List['Tensor']:
    '''
    For a given tensor, return a list of Tensors that make up the nodes of the given Tensor's computational graph,
    in reverse topological order (i.e. `tensor` should be first).

    Note: confusing, but switch to use "parents" instead of "children" to match recipe field.
    '''
    def get_parents(tensor: 'Tensor') -> List['Tensor']:
        if tensor.recipe is None:
            return []
        return list(tensor.recipe.parents.values())

    return topological_sort(tensor, get_parents)[::-1]