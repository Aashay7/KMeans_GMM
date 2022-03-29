import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, num_clusters, data):
        self.X = data
        self.k = num_clusters
        self.clusters = {}
        self.cluster_members = {}
        self.history = {}
        self.initialize_clusters()


    def initialize_clusters(self):
        for i in range(self.k):
            self.clusters[f"cluster_{i}"] = np.random.choice(self.X)
            self.history[f"cluster_{i}"] = [self.clusters[f"cluster_{i}"]]
            self.cluster_members[f"cluster_{i}"] = [0]*len(self.X)
        print(f"Initialized clusters : {self.clusters}")
    

    def initialize_cluster_members(self):
        for i in range(self.k):
            self.cluster_members[f"cluster_{i}"] = [0]*len(self.X)


    def form_clusters(self,  cycles=10):
        """
        The form_clusters method performs the cluster assignment step and 
        subsequently the mean update step. 

        Args:
            cycles (int, optional): Number of cycles/epochs. Defaults to 10.
        """
        for i in range(cycles):
            print(f"Iteration : {i+1}")
            # Cluster Assignment step
            self.initialize_cluster_members()
            for j, datum_x in enumerate(self.X):
                min_dist, min_index = 1e40, None
                
                for cluster, cluster_x in self.clusters.items():
                    euclidean_dist = (datum_x - cluster_x)**2
                    if euclidean_dist < min_dist:
                        min_dist = euclidean_dist
                        min_index= cluster
                self.cluster_members[min_index][j] = 1
            print(f"Assigned members to clusters : {self.cluster_members}")
            
            # Mean Update step
            for cluster, cluster_member in self.cluster_members.items():
                print(f"{cluster} : {cluster_member}")
                self.clusters[cluster] =  np.sum(np.multiply(np.array(self.cluster_members[cluster]), np.array(self.X))) / np.sum(np.array(self.cluster_members[cluster])>0)
                self.history[cluster].append(self.clusters[cluster])
            print(f"Updated clusters : {self.clusters}\n")
            

    def get_cluster_centroids(self):
        for cluster, cluster_x in self.clusters.items():
            print(f"{cluster} : {cluster_x}")


    def plot_clusters(self):
        plt.figure(figsize=(20,10))
        plt.hist(self.X, bins=50)
        for cluster, cluster_x in self.clusters.items():
            plt.plot(cluster_x, 0, 'o', alpha = 0.5, markersize=20*(15-len(self.clusters)))
            plt.plot(cluster_x, 10*(15-len(self.clusters))//8, 'o:k', markersize=20)
        return plt