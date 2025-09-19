import scipy
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_area, compute_center_of_mass, compute_length, get_max_y, rotate_ellipse, scikit_PCA_rotation, rotate_curve_major_vertical, rotate_ellipse_surface, rotate_axis
from visualize import colorbar_rainbow

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
        length = compute_length(curve)
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
    
    def plot_curves(self, n1 = 75, n2 = 81):
        """
        Plot curves from the dataset with a rainbow color gradient.
        Parameters
        ----------
        n1 : int
            The starting index of the curves to plot.
        n2 : int
            The ending index of the curves to plot.
        """
        plt.figure(1)  
        for i in range(n1, n2 + 1):
            col = colorbar_rainbow((i - (n1 - 1)) * 1 / 7).flatten()  # flatten in case it's 2D
            x = self.data[i, :, 0] / 500 + 5 * (i - n1)
            y = self.data[i, :, 1] / 500
            plt.plot(x, y, linewidth=4, color=col)
            plt.axis('equal')

        plt.show()

        plt.figure(2)
        curve_unit_length_list = []
        for idx,i in enumerate(range(n1, n2 + 1)):
            col = colorbar_rainbow( (idx+1) / 7 )
            curve_unit_length = self.curve_unite_length(i)
            x = curve_unit_length[:,0] + idx * 0.15 #better with 0.20
            y = curve_unit_length[:, 1]
            curve_unit_length_list.append(curve_unit_length)
            plt.plot(x, y, linewidth=4, color = col)
            plt.axis('equal')
        plt.show()

        plt.figure(3)
        curve_unit_area_list = []
        for idx,i in enumerate(range(n1, n2 + 1)):
            col = colorbar_rainbow( (idx+1) / 7 )
            curve_unit_area = self.scale_unit_area(i)
            x = curve_unit_area[:,0] + idx * 1.5
            y = curve_unit_area[:, 1]
            curve_unit_area_list.append(curve_unit_area)
            plt.plot(x, y, linewidth=4, color = col)
            plt.axis('equal')
        plt.show()

        plt.figure(4)

        for i in range(7):
            curve = curve_unit_length_list[i]
            col = colorbar_rainbow( (i+1) / 7 )
            c, curve = get_max_y(curve)
            x = curve[:,0] + i * 0.18
            y = curve[:, 1]
            plt.plot(x, y, linewidth=4, color = col, label= f"leaf_{i}")
            plt.plot(curve[c,0] + i *0.18, curve[c,1], '*k', linewidth= 5,)
            center_of_contour = np.mean(curve, axis=0)
            col = [1,127/255,80/255]
            plt.plot(center_of_contour[0] + i * 0.18, center_of_contour[1], '*',color = col, linewidth= 5,)
            col = colorbar_rainbow(0.8)
            center_of_mass = compute_center_of_mass(curve)
            plt.plot(center_of_mass[0] + i * 0.18, center_of_mass[1], '*', color = col, linewidth= 5,)
            
            plt.plot([center_of_mass[0]+ i * 0.18,curve[c,0]+ i * 0.18], [center_of_mass[1],curve[c,1]], '-', color = "brown", label = 'center of mass', linewidth= 2,)
            plt.axis('equal')
        plt.legend()
        plt.show()

        plt.figure(5)
        for i in range(7):
            curve = curve_unit_length_list[i]
            col = colorbar_rainbow( (i+1) / 7 )
            c, curve = get_max_y(curve)
            x = curve[:,0]
            y = curve[:, 1]
            plt.plot(x, y, linewidth = 5, color = col, label= f"leaf_{i}")
            plt.plot(curve[c,0] , curve[c,1], '*k', linewidth= 20)
        plt.legend()
        plt.show()
        


if __name__ == "__main__":
    # THIS IS A COMMENT
    PATH = r"C:\Users\LONGA\Downloads\leaves_parameterized.mat"
    #path = #INSERT THE PATH TO YOUR DATA

    A = scipy.io.loadmat(PATH)
    leaf_curves = A['leaves_parameterized']
    my_first_class = Preprocess(leaf_curves)
    leaf_idx = 76
    plt.plot(leaf_curves[leaf_idx][:,0], leaf_curves[leaf_idx][:,1], 'r-')
    rotate_leaf = rotate_ellipse(leaf_curves[leaf_idx])
    plt.plot(rotate_leaf[:,0], rotate_leaf[:,1], 'b-')

    rotated_curve = scikit_PCA_rotation(leaf_curves[leaf_idx])
    plt.plot(rotated_curve[:,0], rotated_curve[:,1], 'g-')

    rotated_curve_major_vertical = rotate_curve_major_vertical(leaf_curves[leaf_idx])
    plt.plot(rotated_curve_major_vertical[:,0], rotated_curve_major_vertical[:,1], 'y-')

    rotated_surf = rotate_ellipse_surface(leaf_curves[leaf_idx])
    plt.plot(rotated_surf[:,0], rotated_surf[:,1], 'm-')

    plt.axis('equal')
    plt.legend(['Original','Ellipse rotation', 'PCA rotation', 'Major vertical PCA rotation', 'Surface moment rotation'])
    plt.show()



    #my_first_class.plot_curves()
 
    