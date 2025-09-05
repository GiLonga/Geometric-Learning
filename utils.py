import numpy as np

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
        u = np.array([curve[i,0],   curve[i,1],   0.0])
        v = np.array([curve[i+1,0], curve[i+1,1], 0.0])

        w = np.cross(u, v)
        area += 0.5 * w[2]
    if absolute:
        return abs(area)
    return area

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
    area = compute_area(curve)
    for i in range(len(curve) - 1):
        integral_contribution = 0.5*np.power(curve[i,0],2)*(curve[i+1,1]-curve[i,1])#Stokes' theorem
        cx += (-integral_contribution ) / area
        cy += ( integral_contribution ) / area
    return np.array([cx, cy])

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

def rotate_axis(shape):
    """ TO DO """
    pass

if __name__ == "__main__":
    naive_curve = np.array([[0,0], [1,0], [1,1], [0,1], [0,0]])
    naive_area = compute_area(naive_curve)
    cen_of_mass = compute_center_of_mass(naive_curve)
    print("Area:", naive_area)
    print("Center of Mass:", cen_of_mass)
    print("Translated:\n", translate_center_of_mass(naive_curve))