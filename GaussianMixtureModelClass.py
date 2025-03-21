import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate
import scipy.stats
import scipy.interpolate
import scipy.optimize
import AuxFunctions

class GaussianMixture:
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.Npoints = self.x_data.shape[0]
        self.model = []
        self.p0 = [np.max(y_data), x_data[np.argmax(y_data)], 0.1]

    def add_gaussian(self, scale, shift, disp):
        self.model.append((scale, shift, disp))
    
    def evaluate(self, x):
        return AuxFunctions.gaussian_mixture(x, self.model)

    def get_support(self):
        support = set()
        for scale, shift, disp in self.model:
            i1 = max(0, int((shift - 2 * disp) / self.x_data[-1] * self.Npoints))
            i2 = min(self.Npoints - 1, int((shift + 2 * disp) / self.x_data[-1] * self.Npoints))
            support = support.union(set(range(i1, i2 + 1)))
        return support
    
    def get_residuals(self, indicies):
        xs = np.zeros(len(indicies))
        residuals = np.zeros(len(indicies))
        for i, ind in enumerate(indicies):
            xs[i] = self.x_data[ind]
            residuals[i] = self.evaluate(self.x_data[ind]) - self.y_data[ind]
        return (residuals, xs)
            

    def check_residuals(self):
        """
        return True if all residuals are statistically same
        """
        res_support, x_support = self.get_residuals(self.get_support())
        res_other, x_other = self.get_residuals(set(range(self.Npoints)) - self.get_support()) 
        if scipy.stats.anderson_ksamp([res_support, res_other]).pvalue <= 0.05:
            return False
        else:
            return True

    def loss_function(self, params):
        gaussian_params = np.array(params).reshape(-1, 3)
        return np.sum((self.y_data - AuxFunctions.gaussian_mixture(self.x_data, gaussian_params))**2)
    
    def add_gaussian_to_mixture_opt(self):
        Ngaussians = len(self.model) + 1
        print(Ngaussians) 
        lower_bounds = [0.05, 0, 1/self.Npoints * self.x_data[-1]] * Ngaussians
        upper_bounds = [2, 1, 1] * Ngaussians
        bounds = list(zip(lower_bounds, upper_bounds))
        res = scipy.optimize.dual_annealing(self.loss_function, bounds=bounds)
        self.model = res.x.reshape(-1, 3).tolist()
        resid, xs = self.get_residuals(set(range(self.Npoints)))
        # plt.plot(xs, resid)s
        

    def add_gaussian_to_mixture(self):
        Ngaussians = len(self.model) + 1
        print(Ngaussians)
        def fit_func(x, *args):
            gaussian_params = np.array(args).reshape(-1, 3)
            return AuxFunctions.gaussian_mixture(x, gaussian_params)
        bounds = (np.tile(np.array((0.1, 0, 1e-5)), Ngaussians), np.tile(np.array((2, 1, 2)), Ngaussians))
        print(self.p0)
        output = scipy.optimize.curve_fit(fit_func, self.x_data, self.y_data, p0=self.p0, bounds=bounds, full_output=True, maxfev=5000)
        print(output[3], output[4])
        self.model = output[0].reshape(-1, 3).tolist()

        res, xs = self.get_residuals(set(range(self.Npoints)))
        plt.plot(xs, res)
        self.p0.append(np.max(np.abs(res)))
        self.p0.append(xs[np.argmax(np.abs(res))])
        self.p0.append(0.1)

    
    def gaussian_mixture_fit(self):
        iter_count = 1
        iter_max = 8
        self.add_gaussian_to_mixture_opt()
        print(self.model)
        while self.check_residuals() != True:
            if iter_count >= iter_max:
                print(f"Routine did not converged with {iter_max} components and was terminated.")
                break
            self.add_gaussian_to_mixture_opt()
            iter_count += 1
            print(self.model)

    
    


        


    