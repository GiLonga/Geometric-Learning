import scipy
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_area, compute_center_of_mass

class Preprocess():
    """
    A class to preprocess curves represented as sequences of points.
    Attributes
    ----------
    data : np.array
        An array of shape (M, N, D) where M is the number of curves, N is the number of points per curve, and D is the dimension of each point.
    n_leaves : int
        The number of curves (leaves) in the dataset.
    Methods
    -------
    curve_unite_length(curve_number : int = 0) -> np.array
        Normalize a curve to have unit length.
    scale_unit_area(curve_number : int = 0) -> np.array
        Scale a 2D curve to have unit area.
    """
    def __init__(self, data):
        self.data = np.array(data)
        self.n_leaves = self.data.shape[0]

    def curve_unite_length(self, curve_number : int = 0) -> np.array:
        """
        Calculate the length of a curve represented as a sequence of points.
        Parameters
        ----------
        curve : np.array
            An array of shape (N, 2) where N is the number of points and D is the dimension of each point.
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
    
    def scale_unit_area(self, curve_number : int = 0) -> np.array:
        """
        Scale a 2D curve so that the area it encloses is equal to one.
        Parameters
        ----------
        curve : np.array
            An array of shape (N, 2) where N is the number of points.
        Returns
        -------
        scaled_curve : np.array
            The curve scaled to unit area.
        """
        curve = self.data[curve_number,:,:]
        area = compute_area(curve)
        scaled_curve = curve/np.sqrt(np.abs(area))

        return scaled_curve
    
    def rotate_shape(self, curve_number: int, theta: float) -> np.array:
        """
        Rotate a 2D curve by a given angle.
        Parameters
        ----------
        curve_number : int
            The index of the curve to be rotated.
        theta : float
            The angle in radians by which to rotate the curve.
        Returns
        -------
        rotated_curve : np.array
            The rotated curve.
        """
        curve = self.data[curve_number,:,:]
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta),  np.cos(theta)]])
        rotated_curve = curve @ rotation_matrix.T
        return rotated_curve

if __name__ == "__main__":
    # THIS IS A COMMENT
    path = #INSERT THE PATH TO YOUR DATA
    A = scipy.io.loadmat(PATH)
    my_first_class = Preprocess(A['leaves_parameterized'])
    print("Normalizing the second leaf by length")
    print(my_first_class.curve_unite_length(1))
    print("Normalizing the second leaf by area")
    print(my_first_class.scale_unit_area(1))

    plt.plot(my_first_class.data[1,:,0], my_first_class.data[1,:,1], label='Original Curve')
    rotated_leaf = my_first_class.rotate_shape(1, np.pi/4)
    plt.plot(rotated_leaf[:,0], rotated_leaf[:,1], label='Rotated pi/4')
    return_leaf = my_first_class.rotate_shape(1, -np.pi/4)
    plt.plot(return_leaf[:,0], return_leaf[:,1], label='Rotated -pi/4')

    plt.legend()
    plt.show()

    print("Center of mass:" , compute_center_of_mass(my_first_class.data[1,:,:]))

    