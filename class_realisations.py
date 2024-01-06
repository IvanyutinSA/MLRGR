import numpy as np


class KMeans:
    def __init__(self, k=2, max_iter=300):
        self.k = k
        self.max_iter = max_iter

    def fit(self, data):
        min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
        self.centroids = [np.random.uniform(min_, max_) for _ in range(self.k)]

        iter = 0
        self.clusters = [[] for _ in range(len(data))]

        while iter < self.max_iter:
            iter += 1
            for i in range(len(data)):
                x = data[i]
                distance = [(self.distance(x, self.centroids[i]), i) for i in range(self.k)]
                self.clusters[i] = sorted(distance, key=lambda x: x[0])[0][1]

            for i in range(self.k):
                x_j = [data[j] for j in range(len(self.clusters)) if self.clusters[j] == i]
                self.centroids[i] = list(np.mean(x_j, axis=0))
        self.labels_ = self.clusters
        self.cluster_centers_ = self.centroids
        return self

    def distance(self, x, y):
        return sum((x[i]-y[i])**2 for i in range(len(x)))


class TreeNode:
    def __init__(self):
        self.feature = 0
        self.right = None
        self.left = None
        self.t = 0
        self.y = -1


class CART:
    def __init__(self, classification=True, max_depth=4, tolerance=1):
        self.classification = classification
        self.max_depth = max_depth
        self.tolerance = tolerance 

    def fit(self, X, y):
        self.root = self.create_tree(X, y)

    def predict(self, X):
        pred = []
        for x in X:
            node = self.root
            while node.right:
                if x[node.feature] <= node.t:
                    node = node.left
                else:
                    node = node.right
            pred.append(node.y)
        return pred


    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)

        probabilities = counts/counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))

        return entropy
    
    def get_child_entropy(self, left, right):
        total_len = len(left) + len(right)
        left_entropy = self.entropy(left)
        right_entropy = self.entropy(right)

        return len(left)/total_len * left_entropy + len(right)/total_len * right_entropy

    def all_splits(self, X):
        splits = {}
        _, n_col = X.shape

        for col_index in range(n_col):
            values = X[:, col_index] 
            unique_values = np.unique(values)
            splits[col_index] = unique_values

        return splits

    def split(self, X, y, column, t):
        column_values = X[:, column]

        left_X = X[column_values <= t]
        left_y = y[column_values <= t]

        right_X = X[column_values > t]
        right_y = y[column_values > t]

        return (left_X, left_y), (right_X, right_y)

    def best_split(self, X, y, potential_splits):
        best_entropy = 10**8
        best_column = -1
        best_value = -1

        for index in potential_splits:
            for value in potential_splits[index]:
                left, right = self.split(X, y, index, value)
                child_entropy = self.get_child_entropy(left[1], right[1]) 
                if child_entropy < best_entropy:
                    best_column = index
                    best_value = value
                    best_entropy = child_entropy
        
        return best_column, best_value

    def stop(self, y, depth):
        if self.max_depth < depth:
            return True
        if len(np.unique(y)) == 1:
            return True
        else:
            return False
            
    def get_prediction(self, y):
        if self.classification:
            return sorted((counts, uniq) for uniq, counts in  zip(*np.unique(y, return_counts=True)))[-1][1]
        else:
            return np.mean(y)
            

    def create_tree(self, X, y, depth=0):
        prediction = self.get_prediction(y)
        node = TreeNode()
        node.y = prediction

        if not self.stop(y, depth):
            potential_splits = self.all_splits(X)
            best_column, best_value = self.best_split(X, y, potential_splits)
            node.t = best_value
            node.feature = best_column
            left, right = self.split(X, y, best_column, best_value)
            node.left = self.create_tree(left[0], left[1], depth+1)
            node.right = self.create_tree(right[0], right[1], depth+1)

        return node

    def traverse(self, node=None, depth=0):
        x = '|\t'*depth
        if depth == 0:
            node = self.root
        if node.left is None:
            print(f'{x}value:[{node.y if self.classification else round(node.y, 3)}]')
            return
        print(f'{x}feature {node.feature} <= {round(node.t, 3)}')
        self.traverse(node.left, depth+1)
        self.traverse(node.right, depth+1)



