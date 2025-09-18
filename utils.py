import geomstats.backend as gs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import scipy.io
import matplotlib.path as mpltPath
from visualize import plot_curve
#from projection import projection_clock_a_lmbda



def compute_area(curve, absolute = True):
    """ 
    Compute the area enclosed by a 2D curve using the shoelace formula.
    Parameters
    ----------
    curve : np.array
        An array of shape (N, 2) where N is the number of points.
    Returns
    -------
    area : float
        The area enclosed by the curve.
    """
    area = 0
    for i in range(curve.shape[0]-1):
        u = gs.array([curve[i,0],   curve[i,1],   0.0])
        v = gs.array([curve[i+1,0], curve[i+1,1], 0.0])

        w = gs.cross(u, v)
        area += 0.5 * w[2]
    if absolute:
        return abs(area)
    return area

def compute_area_stokes(curve, absolute = True):
    """
    Compute the area enclosed by a 2D curve using Stokes' theorem.
    Parameters
    ----------
    curve : np.array
        An array of shape (N, 2) where N is the number of points.
    absolute : bool
        If True, return the absolute value of the area.
    Returns
    -------
    area : float
        The area enclosed by the curve.
    """
    N = curve.shape[0]
    area = 0
    for i in range (N-1):
        area += curve[i,0]*(curve[i+1,1] - curve[i-1,1])/2
    area += curve[N-1,0]*(curve[0,1] - curve[N-2,1])/2
    if absolute:
        return abs(area)
    return area

def compute_length(curve):
    """ Compute the length of a 2D curve.
    Parameters
    ----------
    curve : np.array
        An array of shape (N, 2) where N is the number of points.
    Returns
    -------
    length : float
        The length of the curve.
    """
    length = 0
    for i in range(curve.shape[0]-1):
        length += gs.linalg.norm(curve[i+1,:]-curve[i,:])
    return length

def get_max_y_and_roll(curve2, shift = True):
    """
    Find the index with the max y-coord in a 2D curve and roll the curve so that this point is first.
    Parameters
    ----------
    curve2 : np.array
        An array of shape (N, 2) where N is the number of points.
    Returns
    -------
    curve2 : np.array
        The curve rolled so that the point with the maximum y-coordinate is first.
    """
    max_y_coord = gs.argmax(curve2[:,1])
    rolled_curve = np.roll(curve2, -max_y_coord, axis=0)
    if shift:     
        rolled_shifted_curve = rolled_curve - curve2[max_y_coord]
        return rolled_shifted_curve
    else:
        return rolled_curve

def compute_center_of_mass(curve):
    """ Compute the center of mass of a 2D curve.
    Parameters 
    ----------
    curve : np.array
        An array of shape (N, 2) where N is the number of points.
    Returns
    -------
    center_of_mass : np.array
        The center of mass of the curve.
    """
    cx = 0
    cy = 0
    area = compute_area(curve, absolute = False)
    for i in range(len(curve) - 1):
        integral_contribution_x = 0.5*gs.power(curve[i,0],2)*(curve[i+1,1]-curve[i,1])#Stokes' theorem
        integral_contribution_y = 0.5*gs.power(curve[i,1],2)*(curve[i+1,0]-curve[i,0])#Stokes' theorem
        cx += (  integral_contribution_x ) / area
        cy += (- integral_contribution_y ) / area
    return gs.array([cx, cy])

def translate_center_of_mass(curve):
    """ Translate a 2D curve so that its center of mass is at the origin.
    Parameters
    ----------
    curve : np.array
        An array of shape (N, 2) where N is the number of points.
    Returns
    -------
    translated_curve : np.array
        The curve translated so that its center of mass is at the origin.
    """
    center_of_mass = compute_center_of_mass(curve)
    translated_curve = curve - center_of_mass
    return translated_curve

def get_max_y(curve):
    """
    Find the index of the point with the maximum y-coordinate in a 2D curve and translate the curve so that this point is at the origin.
    Parameters
    ----------
    curve : np.array
        An array of shape (N, 2) where N is the number of points.
    Returns
    -------
    c : int
        The index of the point with the maximum y-coordinate.
    curve : np.array
        The curve translated so that the point with the maximum y-coordinate is at the origin.
    """
    c = gs.argmax(curve[:,1])
        #curve  
    curve_t = curve.copy()
    curve_t[:,0] = curve[:,0] - curve[c,0]
    curve_t[:,1] = curve[:,1] - curve[c,1]
    return c, curve_t

def rotate_axis(curve) -> gs.array:
    """
    Rotate a 2D curve so that the point with the maximum y-coordinate lies on the positive y-axis.
    Parameters
    ----------
    curve : np.array
        An array of shape (N, 2) where N is the number of points.
    Returns
    -------
    rotated_curve : np.array
        The curve rotated so that the point with the maximum y-coordinate lies on the positive y-axis.
    """
    curve = translate_center_of_mass(curve)
    if gs.argmax((curve[:,1])) != 0:
        raise ValueError("Something wrong happend during rolling.")
    
    vector = curve[gs.argmax(curve[:,1]), :]
    n_vector = vector/gs.sqrt(gs.power(vector[0],2) + gs.power(vector[1], 2))
    x = n_vector[0]
    y = n_vector[1]
    rotation_matrix = gs.array([[y, -x], [x, y]])
    curve_rotated_axis = (rotation_matrix @ curve.T).T

    return curve_rotated_axis

def rotate_ellipse(curve):
    """
    Rotate a 2D curve so that its principal axes align with the coordinate axes using PCA.
    Parameters
    ----------
    curve : np.array
        An array of shape (N, 2) where N is the number of points.
    Returns
    -------
    rotated_curve : np.array
        The curve rotated so that its principal axes align with the coordinate axes.
    """
    N = curve.shape[0]
    area = 0
    E = gs.zeros((2,2))
    for i in range (N-1):
        u = gs.array([curve[i,0], curve[i,1], 0.0])
        v = gs.array([curve[i+1,0], curve[i+1,1], 0.0])
        w = gs.cross(u, v)
        area += 0.5 * w[2]
        E[0,0] += (w[2]*0.25*(curve[i,0]**2+curve[i+1,0]**2))
        E[1,1] += (w[2]*0.25*(curve[i,1]**2+curve[i+1,1]**2))
        E[0,1] += (w[2]*0.25*(curve[i,0]*curve[i,1]+curve[i+1,0]*curve[i+1,1]))
    E[1,0] = E[0,1]
    E = E/area
    U,S,V = gs.linalg.svd(E)
    flip = gs.array([[1,0],[0,1]])
    rotated_curve = flip @ U @curve.T
    return rotated_curve.T

def scikit_PCA_rotation(curve):
    """
    Rotate a 2D curve using PCA from sklearn.
    Parameters
    ----------
    curve : np.ndarray, shape (N, 2)
        Input 2D curve (array of points).
    Returns
    -------
    rotated_curve : np.ndarray, shape (N, 2)
        Curve rotated.
    """

    pca = PCA(n_components=2)
    pca.fit(curve)
    rotated_curve = pca.transform(curve)
    return rotated_curve

def rotate_curve_major_vertical(curve):
    """
    Rotate a 2D curve so that:
      - Minor axis aligns with the horizontal axis (x)
      - Major axis aligns with the vertical axis (y)

    Parameters
    ----------
    curve : np.ndarray, shape (N, 2)
        Input 2D curve (array of points).

    Returns
    -------
    rotated_curve : np.ndarray, shape (N, 2)
        Curve rotated (no translation).
    """
    # Perform PCA on the original (unshifted) curve
    pca = PCA(n_components=2)
    pca.fit(curve)
    components = pca.components_
    variances = pca.explained_variance_

    # Find major/minor axis
    major_idx = gs.argmax(variances)
    minor_idx = 1 - major_idx

    # Build rotation matrix
    R = gs.vstack([components[minor_idx], components[major_idx]])

    # Ensure right-handed orientation
    if gs.linalg.det(R) < 0:
        R[1, :] *= -1

    # Apply rotation (no mean-centering)
    rotated_curve = curve @ R.T

    return rotated_curve

def rotate_ellipse_surface(curve):
    """ 
    Rotate a 2D curve so that its principal axes align with the coordinate axes using the surface moments.
    Parameters
    ----------
    curve : np.array
        An array of shape (N, 2) where N is the number of points.
    Returns
    -------
    rotated_curve : np.array
        The curve rotated so that its principal axes align with the coordinate axes.
    """
    N,d = curve.shape
    rotated_curve = gs.zeros((N,d))
    area = compute_area(curve)
    E = gs.zeros((2,2))
    f = curve/gs.sqrt(area)
    for i in range (N-1):
        u = gs.array([f[i,0], f[i,1], 0.0])
        v = gs.array([f[i+1,0], f[i+1,1], 0.0])
        w = gs.cross(u, v)
        E[0,0] += (w[2]*0.25*(f[i,0]**2+f[i+1,0]**2))
        E[1,1] += (w[2]*0.25*(f[i,1]**2+f[i+1,1]**2))
        E[0,1] += (w[2]*0.25*(f[i,0]*f[i,1]+f[i+1,0]*f[i+1,1]))
    E[1,0] = E[0,1]
    U,S,V = gs.linalg.svd(E)
    rot90 = gs.array([[0,-1],[1,0]])
    if gs.linalg.det(U) < 0:
        V = gs.array([[1,0],[0,-1]]) @ V
    for i in range(N):
        rotated_curve[i,:] = rot90 @ V @ curve[i,:].T
    return rotated_curve

def check_and_shift_center_of_mass(curve_to_check, tolerance=5e-4, verbose = False):
    """
    Shift a 2D curve (polygon) if necessary so that the origin (0,0) lies inside it,
    with a safety tolerance from the border.

    Parameters
    ----------
    curve : np.ndarray
        Array of shape (N, 2), representing N 2D points of the polygon.
    tolerance : float, optional
        Minimum distance by which the origin should be pushed inside.
        Default is 1e-3.
    verbose : bool, optional
        If True, plot debugging info. Default is False.

    Returns
    -------
    shifted_curve : np.ndarray
        The curve, shifted if needed so that the origin lies safely inside.
    """
    path = mpltPath.Path(curve_to_check)

    # If origin already inside, no shift needed
    if path.contains_point([0,0]):
        return curve_to_check.copy()

    # Find closest point to origin
    distances = np.linalg.norm(curve_to_check, axis=1)
    closest_point = curve_to_check[np.argmin(distances)]

    # Shift vector toward origin, overshoot slightly
    shift_vec = -closest_point
    norm = np.linalg.norm(shift_vec)
    if norm > 0:
        shift_vec = shift_vec * (1 + tolerance / norm)

    shifted_curve = curve_to_check + shift_vec

    if verbose:
        print(f"The curve was shifted so the origin lies inside (tolerance {tolerance}).")
        plt.plot(curve_to_check[:,0], curve_to_check[:,1], 'g-', label='Original curve')
        plt.plot([0],[0], 'rx', label='Origin')
        plt.plot(shifted_curve[:,0], shifted_curve[:,1], 'y-', label='Shifted curve')
        plt.axis('equal')
        plt.legend()
        plt.show()

    return shifted_curve

def scale_unit_area(curve_local : int = 0) -> np.array:
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
        area = compute_area(curve_local)
        scaled_curve = curve_local/np.sqrt(np.abs(area))

        return scaled_curve

def curve_unite_length(curve) -> np.array:
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
    length = compute_length(curve)
    normalized_curve = curve/length
    return normalized_curve

if __name__ == "__main__":


    #ETAPE 4 LEAVES
    workspace_path = os.getcwd()
    path = os.path.join(workspace_path, 'training_set.mat')
    A = scipy.io.loadmat(path)
    training_leaves = A['etape4']

    # ORIGINAL LEAVES 
    PATH = r"C:\Users\LONGA\Downloads\leaves_parameterized.mat"
    #path = #INSERT THE PATH TO YOUR DATA
    A = scipy.io.loadmat(PATH)
    training_leaves = A['leaves_parameterized']
    #plot_curve(training_leaves[95])


    #curve = leaves[677,:,:]
    for i in range(675,680,1): #453,453, 501):
        naive_curve = training_leaves[i]
        #naive_area = compute_area(naive_curve)
        
        rotate_curve = rotate_curve_major_vertical(naive_curve)
        rotate_curve = get_max_y_and_roll(rotate_curve)
        rotate_curve = curve_unite_length(rotate_curve)
        rotate_curve = translate_center_of_mass(rotate_curve)
        rotate_curve = rotate_axis(rotate_curve)
        #rotate_curve1, rotate_curve2 = projection_clock_a_lmbda(rotate_curve)
        #rotate_curve = rotate_curve1
        cen_of_mass = compute_center_of_mass(rotate_curve)

        #plt.plot(naive_curve[:,0], naive_curve[:,1], 'k-')
        plt.plot(rotate_curve[:,0], rotate_curve[:,1], 'g-')
        plt.plot(0.0,0.0, 'gx')
        plt.plot(cen_of_mass[0], cen_of_mass[1], 'bx')  # Mark center of mass
        #print("Area:", naive_area)
        #print("Area Stokes:", compute_area_stokes(naive_curve))
        #check_and_shift_center_of_mass(naive_curve, verbose=True)          

        #print("Length:", compute_length(naive_curve))
        #print(naive_curve)
        #print("Center of Mass:", cen_of_mass)
        #print("Translated:\n", translate_center_of_mass(naive_curve))
        #print("Max y point index and translated:\n", get_max_y(naive_curve))
        #r_curve = rotate_ellipse(naive_curve)
        #print("Rotated ellipse:\n",r_curve )
        #plt.plot(naive_curve[:,0], naive_curve[:,1], 'y-')
        #plt.plot(0.0,0.0, 'gx')
        #plt.plot(cen_of_mass[0], cen_of_mass[1], 'bx')  # Mark center of mass
        #plt.plot(r_curve[:,0], r_curve[:,1], 'b-')
        plt.title(f'Leaf {i}')
        plt.axis('equal')
        plt.show()