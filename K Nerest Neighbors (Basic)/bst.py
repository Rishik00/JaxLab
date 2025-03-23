## can do searching, insertions and deletions in O(h) time

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def insertNode(root, key):
    if not root:
        return Node(key)
    
    if root.val == key:
        return root
    
    if root.val < key:
        root.right = insertNode(root.right, key)
    
    else:
        root.left = insertNode(root.left, key)

    return root

def searchFn(root, key):
    if not root:
        return None
    
    if root.val == key:
        return root
    
    if root.val < key:
        return searchFn(root.right, key)
    else:
        return searchFn(root.left, key)


if __name__ == "__main__":
    root = Node(50)
    root.left = Node(30)
    root.right = Node(70)
    root.left.left = Node(20)
    root.left.right = Node(40)
    root.right.left = Node(60)
    root.right.right = Node(80)

    print("Found" if searchFn(root, 19) else "Not Found")
    print("Found" if searchFn(root, 80) else "Not Found")