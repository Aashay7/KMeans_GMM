import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, num_centroids, data):
        self.X = data
        self.k = num_centroids
        self.centroids = {}
        self.centroid_members = {}
        self.history = {}
        self.initialize_centroids()


    def initialize_centroids(self):
        for i in range(self.k):
            self.centroids[f"centroid_{i}"] = np.random.choice(self.X)
            self.history[f"centroid_{i}"] = [self.centroids[f"centroid_{i}"]]
            self.centroid_members[f"centroid_{i}"] = [0]*len(self.X)
        print(f"Initialized centroids : {self.centroids}")
    

    def initialize_centroid_members(self):
        for i in range(self.k):
            self.centroid_members[f"centroid_{i}"] = [0]*len(self.X)


    def form_clusters(self,  cycles=10, threshold=0.1):
        for i in range(cycles):
            print(f"Iteration : {i+1}")
            # Assignment
            self.initialize_centroid_members()
            for j, datum_x in enumerate(self.X):
                min_dist, min_index = 1e40, None
                
                for centroid, centroid_x in self.centroids.items():
                    euclidean_dist = (datum_x - centroid_x)**2
                    if euclidean_dist < min_dist:
                        min_dist = euclidean_dist
                        min_index= centroid
                self.centroid_members[min_index][j] = 1
            print(f"Assigned members to centroids : {self.centroid_members}")
            
            # Update the means
            for centroid, centroid_member in self.centroid_members.items():
                print(f"{centroid} : {centroid_member}")
                self.centroids[centroid] =  np.sum(np.multiply(np.array(self.centroid_members[centroid]), np.array(self.X))) / np.sum(np.array(self.centroid_members[centroid])>0)
                self.history[centroid].append(self.centroids[centroid])
            print(f"Updated centroids : {self.centroids}\n")
            

    def get_centroid_value(self):
        for centroid, centroid_x in self.centroids.items():
            print(f"{centroid} : {centroid_x}")


    def plot_clusters(self):
        plt.figure(figsize=(20,10))
        plt.hist(self.X, bins=50)
        for centroid, centroid_x in self.centroids.items():
            plt.plot(centroid_x, 0, 'o', alpha = 0.5, markersize=20*(15-len(self.centroids)))
            plt.plot(centroid_x, 10*(15-len(self.centroids))//8, 'o:k', markersize=20)
        return plt