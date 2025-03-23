k = 2  # Dimension of the KD-Tree

class KDNode:
    def __init__(self, point):
        self.point = point
        self.left = None
        self.right = None


def insertRecord(root, point, depth):
    # Base case: if tree is empty, return a new node
    if not root:
        return KDNode(point)
    
    # Calculate current axis
    axis = depth % k

    # Recursively insert to the left or right subtree
    if point[axis] < root.point[axis]:
        root.left = insertRecord(root.left, point, depth + 1)
    else:
        root.right = insertRecord(root.right, point, depth + 1)

    return root  # Return the (possibly updated) root


def arePointsSame(point1, point2):
    # Compare all coordinates of the two points
    return all(point1[i] == point2[i] for i in range(k))


def searchNode(root, key, depth):
    # Base case: root is None or the point is found
    if not root:
        return False
    if arePointsSame(root.point, key):
        return True

    # Calculate current axis
    axis = depth % k

    # Recursively search in the left or right subtree
    if key[axis] < root.point[axis]:
        return searchNode(root.left, key, depth + 1)
    else:
        return searchNode(root.right, key, depth + 1)


if __name__ == "__main__":
    root = None
    points = [[3, 6], [17, 15], [13, 15], [6, 12], [9, 1], [2, 7], [10, 19]]

    # Insert points into the KD-Tree
    for point in points:
        root = insertRecord(root, point, 0)  # Update root after each insertion

    print("All nodes inserted")

    # Search for a point
    print("Found" if searchNode(root, [13, 15], 0) else "Not Found")
    print("Found" if searchNode(root, [10, 5], 0) else "Not Found")
