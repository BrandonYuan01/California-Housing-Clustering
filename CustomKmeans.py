import numpy as np


class CustomKMeans:
    def __init__(self, k, distance_metric='euclidean', max_iters=100, tol=1e-4):
        self.k = k
        self.distance_metric = distance_metric
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.clusters = None

    def minkowski_distance(self, x, y, p):
        if p == np.inf:
            return np.max(np.abs(x - y))
        return np.sum(np.abs(x - y) ** p) ** (1 / p)

    def initialize_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    def assign_clusters(self, X, centroids, p):
        clusters = []
        for x in X:
            distances = [self.minkowski_distance(x, centroid, p) for centroid in centroids]
            clusters.append(np.argmin(distances))
        return np.array(clusters)

    def update_centroids(self, X, clusters):
        new_centroids = []
        for i in range(self.k):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(X[np.random.randint(0, X.shape[0])])
        return np.array(new_centroids)

    def calculate_sse(self, X, centroids, clusters):
        sse = 0
        for i, centroid in enumerate(centroids):
            cluster_points = X[clusters == i]
            sse += np.sum((cluster_points - centroid) ** 2)
        return sse

    def fit(self, X):
        if self.distance_metric == 'manhattan':
            p = 1
        elif self.distance_metric == 'euclidean':
            p = 2
        elif self.distance_metric == 'sup':
            p = np.inf

        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iters):
            self.clusters = self.assign_clusters(X, self.centroids, p)
            new_centroids = self.update_centroids(X, self.clusters)
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

        sse = self.calculate_sse(X, self.centroids, self.clusters)
        return self.centroids, self.clusters, sse
