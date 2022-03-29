import numpy as np
import matplotlib.pyplot as plt

class GaussianMixture:
    def __init__(self, num_gaussians, data):
        self.X                  = np.array(data)
        self.k                  = num_gaussians
        self.gaussian_mean      = {}
        self.gaussian_variance  = {}
        self.gaussian_scaling   = {}
        self.gaussian_members   = {}
        self.likelihood         = []
        self.posterior          = {}
        self.history            = {}
        self.delta              = 1e-10
        self.initialize_gaussians()


    def initialize_gaussians(self):
        scaling     = np.ones(shape=(self.k)) / self.k
        means       = np.linspace(np.min(self.X), np.max(self.X), self.k) 
        variance    = scaling * np.var(self.X)
        
        self.history['mean']        = {}
        self.history['variance']    = {}
        self.history['scaling']     = {}

        for i in range(self.k):
            self.gaussian_mean[f"gaussian_{i}"]         = means[i]
            self.gaussian_variance[f"gaussian_{i}"]     = variance[i]
            self.gaussian_scaling[f"gaussian_{i}"]      = scaling[i]
            self.history['mean'][f"gaussian_{i}"]       = [means[i]]
            self.history['variance'][f"gaussian_{i}"]   = [variance[i]]
            self.history['scaling'][f"gaussian_{i}"]    = [scaling[i]]

        print(f"Initialized gaussians mean:     {self.gaussian_mean}")
        print(f"Initialized gaussians variance: {self.gaussian_variance}")
        print(f"Initialized gaussians scaling:  {self.gaussian_scaling}")


    def get_gaussian_likelihood(self, data, mean, variance):
        """
        Calculates the likelihood of the data exemplars to fall under a given 
        gaussian gaussian with unique mean, variance and scaling.

        Args:
            gaussian (string): Key/id of the gaussian.

        Returns:
            likelihood (float) : The likelihood value of each exemplar to be associated with the given gaussian.
        """
        normalize_term  = 1 / (np.sqrt(2 * np.pi * (variance)))
        exp_term        = np.exp(-(np.power((data - mean),2)) / (2*variance)) 
        return normalize_term * exp_term


    def E_step(self):
        """
        The E-step or Expectation step consists of calculating the likelihood
        of each x_i by using the estimated parameters.
        The function calculates the likelihood of a given example x_i to belong
        to a given cluster.
        """
        self.likelihood = []
        for gaussian in self.gaussian_mean.keys():
            likely = self.get_gaussian_likelihood(self.X, self.gaussian_mean[gaussian], self.gaussian_variance[gaussian])
            self.likelihood.append(likely)
        self.likelihood= np.array(self.likelihood)
        

    def M_step(self):
        # Recompute mean, variance and scaling factor.
        """
        The M-Step or Maximization step consists of updating the parameters of 
        the model such as the cluster/gaussian mean, variance and the scaling factor. 
        """
        for j, gaussian in enumerate(self.gaussian_mean.keys()):
            self.posterior[gaussian] = []
            posterior_val = (self.likelihood[j] * self.gaussian_scaling[gaussian]) / (np.sum([self.likelihood[i] * self.gaussian_scaling[ctr] for i,ctr in enumerate(self.gaussian_mean.keys())], axis=0) + self.delta)
            self.posterior[gaussian].append(posterior_val)

            self.gaussian_mean[gaussian]        = np.sum(self.posterior[gaussian] * self.X) / (np.sum(self.posterior[gaussian]) + self.delta)
            self.gaussian_variance[gaussian]    = np.sum(self.posterior[gaussian] * np.square(self.X- self.gaussian_mean[gaussian])) / (np.sum(self.posterior[gaussian]) + self.delta)
            self.gaussian_scaling[gaussian]     = np.mean(self.posterior[gaussian])

            self.history['mean'][gaussian].append(self.gaussian_mean[gaussian])
            self.history['variance'][gaussian].append(self.gaussian_variance[gaussian])
            self.history['scaling'][gaussian].append(self.gaussian_scaling[gaussian])
            
            
        print(f"The Gaussian Mean is        {self.gaussian_mean}")
        print(f"The Gaussian Variance is    {self.gaussian_variance}")
        print(f"The Gaussian Scaling is     {self.gaussian_scaling}\n\n")


    def learn(self, cycles=10):
        for i in range(cycles):
            print(f"Iteration : {i+1}")
            self.E_step()
            self.M_step()


    def plot_evolution(self):
        bins = np.linspace(np.min(self.X), np.max(self.X), 100)
        plt.figure(figsize=(30,15))
        plt.xlabel("x")
        plt.ylabel("pdf")
        plt.scatter(self.X, [0.]*len(self.X), color='red', marker='+', label='train data')
        #plt.hist(self.X, bins=50)
        
        for gaussian, gaussian_mean_list in self.history["mean"].items():
            color = (np.random.random(), np.random.random(), np.random.random())
            for i,gaussian_mean in enumerate(gaussian_mean_list):
                alpha_val = 0.01 + i/(len(gaussian_mean_list) + 10)
                plt.plot(bins, self.get_gaussian_likelihood(bins, self.history["mean"][gaussian][i],  self.history["variance"][gaussian][i]), color=color, label="Cluster 1", alpha = alpha_val)
        return plt

"""
We see that there are far fewer blue stars (Cluster 0) than compared to the red stars (Cluster 1).
The stars we see in our galaxy are far lesser than the stars from the other galaxy. 
This is logical as a image is a 2D representation of the 3D space, as the distance increases, the scope window also increases and as a result more stars from far away are captured (however they might be a bit dimmer!)

"""