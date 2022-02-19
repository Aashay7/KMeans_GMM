import numpy as np
import matplotlib.pyplot as plt

class GaussianMixture:
    def __init__(self, num_centroids, data):
        self.X                  = np.array(data)
        self.k                  = num_centroids
        self.centroid_mean      = {}
        self.centroid_variance  = {}
        self.centroid_scaling   = {}
        self.centroid_members   = {}
        self.likelihood         = []
        self.posterior          = {}
        self.delta              = 1e-10
        self.initialize_centroids()


    def initialize_centroids(self):
        weights     = np.ones(shape=(self.k)) / self.k
        means       = np.linspace(np.min(self.X), np.max(self.X), self.k)
        variance    = weights * np.var(self.X)
        
        for i in range(self.k):
            self.centroid_mean[f"centroid_{i}"]     = means[i]
            self.centroid_variance[f"centroid_{i}"] = variance[i]
            self.centroid_scaling[f"centroid_{i}"]  = weights[i]
            self.history_mean[f"centroid_{i}"]      = [means[i]]

        print(f"Initialized centroids mean:     {self.centroid_mean}")
        print(f"Initialized centroids variance: {self.centroid_variance}")
        print(f"Initialized centroids scaling:  {self.centroid_scaling}")


    def get_gaussian_likelihood(self, centroid):
        normalize_term  = 1 / (np.sqrt(2 * np.pi * (self.centroid_variance[centroid])))
        exp_term        = np.exp(-(np.power((self.X - self.centroid_mean[centroid]),2)) / (2*self.centroid_variance[centroid])) 
        return normalize_term * exp_term


    def E_step(self):
        """
        
        """
        # Calculate the likelihood of each observation x_i using the 
        # estimated parameters.
        self.likelihood = []
        for centroid in self.centroid_mean.keys():
            likely = self.get_gaussian_likelihood(centroid)
            self.likelihood.append(likely)
        self.likelihood= np.array(self.likelihood)
        

    def M_step(self):
        # Recompute mean, variance and scaling factor.
        for j, centroid in enumerate(self.centroid_mean.keys()):
            self.posterior[centroid] = []
            posterior_val = (self.likelihood[j] * self.centroid_scaling[centroid]) / (np.sum([self.likelihood[i] * self.centroid_scaling[ctr] for i,ctr in enumerate(self.centroid_mean.keys())], axis=0) + self.delta)
            self.posterior[centroid].append(posterior_val)

            self.centroid_mean[centroid]        = np.sum(self.posterior[centroid] * self.X) / (np.sum(self.posterior[centroid]) + self.delta)
            self.centroid_variance[centroid]    = np.sum(self.posterior[centroid] * np.square(self.X- self.centroid_mean[centroid])) / (np.sum(self.posterior[centroid]) + self.delta)
            self.centroid_scaling[centroid]     = np.mean(self.posterior[centroid])
            
        print(f"The Centroid Mean is        {self.centroid_mean}")
        print(f"The Centroid Variance is    {self.centroid_variance}")
        print(f"The Centroid Scaling is     {self.centroid_scaling}\n\n")


    def learn(self, cycles=10):
        for i in range(cycles):
            print(f"Iteration : {i+1}")
            self.E_step()
            self.M_step()


    def plot_clusters(self):
        plt.figure(figsize=(20,10))
        plt.hist(self.X, bins=50)
        for centroid, centroid_x in self.centroids.items():
            plt.plot(centroid_x, 0, 'o', alpha = 0.5, markersize=20*(15-len(self.centroids)))
            plt.plot(centroid_x, 10*(15-len(self.centroids))//8, 'o:k', markersize=20)
        return plt