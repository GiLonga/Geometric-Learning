import scipy
import numpy as np
import matplotlib.pyplot as plt

class Normalizer():
    def __init__(self, data):
        self.data = np.array(data)
        self.n_leaves = self.data.shape[0]

    def curve_unite_length(self, curve_number : int = 0) -> np.array:
        """
        Calculate the length of a curve represented as a sequence of points.
        Parameters
        ----------
        curve : np.array
            An array of shape (N, D) where N is the number of points and D is the dimension of each point.
        Returns
        -------
        normalized_curve : np.array
            The curve normalized to unit length.
        """
        length = 0
        curve = self.data[curve_number,:,:]
        for i in range(curve.shape[0]-1):
            length += np.linalg.norm(curve[i+1,:]-curve[i,:])
        
        normalized_curve = curve/length 

        return normalized_curve

if __name__ == "__main__":
    # THIS IS A COMMENT
    path = #INSERT THE PATH TO YOUR DATA
    A = scipy.io.loadmat(path)
    my_first_class = Normalizer(A['leaves_parameterized'])
    print(my_first_class.curve_unite_length(1))
    print('my_first_class')