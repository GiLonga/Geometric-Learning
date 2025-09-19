import warnings
from colorama import Fore, Style
from  scipy.interpolate import interp1d, CubicSpline
from matplotlib import pyplot as plt
import numpy as np
from visualize import colorbar_rainbow
import geomstats.backend as gs
from utils import compute_length, translate_center_of_mass, compute_area, rotate_axis, get_max_y_and_roll, check_and_shift_center_of_mass, check_shift_voronoi

def projection_clock_a_lmbda(curve, n_subdivision = 12, lmbda = None, visualize = False):
    """
    Compute the clock parametrization (with optional λ-weight) of a closed curve.

    This function takes a discrete curve in R^2 (or higher dimension) and reparametrizes it 
    using a "clock parametrization" approach. The curve is subdivided into approximately 
    equal angular segments, and each segment is linearly interpolated over a uniform 
    parameter grid. Optionally, a λ-dependent projection is applied to each segment to 
    produce an additional parametrization taking in account the curvature.

    Parameters
    ----------
    curve : ndarray of shape (M, dim)
        The input discrete curve, given as an ordered sequence of points.
        Typically dim = 2 for planar curves.
    n_subdivision : int, default=12
        Number of angular subdivisions used to partition the curve.
    lmbda : float or None, default=None
        If provided, computes an additional λ-weighted curvature projected parametrization. 
        If None, only the standard clock parametrization is returned.
    visualize : To set True if you wanna visualize the problematic cases.

    Returns
    -------
    clock_param_fun_a : ndarray of shape (N, dim)
        The curve reparametrized using the clock parametrization.
    clock_param_fun_a_lambda : ndarray of shape (N, dim), optional
        The λ-projected version of the clock parametrization. 
        Returned only if `lmbda` is not None.

    Notes
    -----
    - The curve is first recentered to remove translation effects via 
      `check_and_shift_center_of_mass`.
    - If the curve orientation is clockwise (negative area), it is flipped 
      to ensure counter-clockwise orientation.
    - Subdivision indices are chosen based on uniform angular sampling 
      of the curve.
    """
    n_frame, dim = curve.shape
    newdelta = np.linspace(0,1,n_frame)
    fixed_curve = check_shift_voronoi(curve, verbose = False)
    if compute_area(fixed_curve, absolute = False) < 0:
        fixed_curve = np.flipud(fixed_curve)
    if n_subdivision > 0:
        angles = extract_angle_sequence(fixed_curve)
        indices = extract_uniform_angles(n_subdivision, angles) #Indixes of the curve points closest to the unif ang
        unif_subset = int(np.floor(n_frame/n_subdivision))
    elif n_subdivision == 0:
        indices = np.array([0, n_frame-1])
        unif_subset = n_frame
    else: 
        print("Give as number of angles a positive integer")
        return False
        
    clock_param_fun_a = gs.zeros((n_frame, dim))
    clock_param_fun_a_lambda = gs.zeros((n_frame, dim))

    for k in range(len(indices) - 1):
        argument = newdelta[indices[k]:indices[k + 1] + 1] #Taking a portion of the uniform param.
        newdel2 = np.linspace(argument[0], argument[-1], num = unif_subset)
        #Calculating the parametrizations
        for d in range(dim):
            portion_of_x_or_y = fixed_curve[indices[k]:indices[k + 1] + 1, d]
            f_interp = interp1d(argument, portion_of_x_or_y, kind='linear')
            clock_param_fun_a[(k)*unif_subset:(k+1)*unif_subset, d] = f_interp(newdel2)
        
        if lmbda is not None:
                #portion_of_x_and_y = clock_param_fun_a[indices[k]:indices[k + 1] + 1, :]
                clock_param_fun_a_lambda[(k)*unif_subset:(k+1)*unif_subset, :] = reparametrize_by_curvature(clock_param_fun_a[(k)*unif_subset:(k+1)*unif_subset, :] , lmbda, unif_subset)
                if visualize:
                    plt.plot(
                    clock_param_fun_a_lambda[(k) * unif_subset: ( k + 1 ) * unif_subset, 0], 
                    clock_param_fun_a_lambda[(k) * unif_subset: ( k + 1 ) * unif_subset, 1],
                    '*',
                    color=colorbar_rainbow(k / (len(indices) - 1)))
                #Adding lines for the subsections
                    plt.plot(
                        [0,clock_param_fun_a_lambda[( k ) * unif_subset, 0]],
                        [0,clock_param_fun_a_lambda[( k ) * unif_subset, 1]],
                        'k--')
    if visualize:
        plt.title(f'Clock parameterization with number of angles = {n_subdivision} and lambda = {lmbda}')
        plt.axis('equal')
        plt.show()
    if lmbda is None:
        return clock_param_fun_a
    else:
        return clock_param_fun_a, clock_param_fun_a_lambda

def reparametrize_by_curvature(curve, lmbda = 1, unif_subset = 12):
    """
    Reparametrize a curve segment proportionally to curvature and arc length.

    This function redistributes points along a curve segment such that regions of 
    high curvature are sampled more densely. The parameter λ controls the trade-off 
    between arc-length uniformity and curvature sensitivity.

    Parameters
    ----------
    curve : ndarray of shape (M, dim)
        Discrete curve segment as an ordered sequence of M points in `dim` dimensions.
    lmbda : float or None, default=1
        Weighting parameter:
        - If None, parametrization is based purely on curvature.
        - If float, parametrization weights arc length (scaled by λ) in 
          addition to curvature.
    uniform_subset = integer,
        Nuber of points to add in the new parametrization.

    Returns
    -------
    function_parametrized_prop_to_curv_arc_length : ndarray of shape (M, dim)
        The reparametrized curve segment, redistributed according to curvature 
        and arc length.

    Notes
    -----
    - Curvature is approximated via finite differences of the second derivative.
    - The weight function is defined as:
        * curvature (if `lmbda` is None)
        * λ * length + curvature (otherwise)
    - Cubic interpolation is applied to resample the segment.
    """
    N, dim = curve.shape
    length = compute_length(curve)
    newdelta = np.linspace(0, 1, unif_subset)
    #newdelta, curve = reparametrize_by_arc_length(curve, N, normalized = True)
    f_seconde = gs.zeros((unif_subset, dim))
    for i in range(2):
        for j in range (1,unif_subset-1, 1):
            f_seconde[j,i] = (
                curve[j+1,i] - 2*curve[j,i] + curve[j-1,i])*(gs.power((unif_subset-1),2)/(length**2))
        f_seconde[0,i] = 0
        f_seconde[-1,i] = 0

    curvature = gs.linalg.norm(f_seconde, axis = 1 ) #Curvature = increment of the velocity vect
    curvedelta = gs.zeros(unif_subset) #unif_subset

    if lmbda is None:
        weights = curvature
    else:
        weights = lmbda*length + curvature

    total_curve_arc_length = np.trapz(weights, newdelta)
    for s in range(1,unif_subset,1):
        curvedelta[s-1] = np.trapz( weights[:s], newdelta[:s]) / total_curve_arc_length
    curvedelta[-1] = 1.0
    newcurvedelta = gs.linspace(0,1,unif_subset)
    function_parametrized_prop_to_curv_arc_length = gs.zeros((unif_subset, dim))
    for i in range(dim):
        #inter_func2 = interp1d(curvedelta, curve[:,i], kind='cubic', fill_value="extrapolate")
        inter_func2 = interp1d(curvedelta, curve[:,i], kind='linear')
        #inter_func2 = CubicSpline(curvedelta, curve[:,i])
        function_parametrized_prop_to_curv_arc_length[:,i] = inter_func2(newcurvedelta)

    return function_parametrized_prop_to_curv_arc_length

def projection_prop_curv_lambda_length_in_R2(curve, N, lmbda = None, normalized = False ):
    """ Project a 2D curve onto a new parameterization proportional to its curvature and length.
    Parameters
    ----------
    curve : np.array
        An array of shape (nb_frame, 2) representing the 2D curve.
    N : int
        The number of points in the output curve.
    lmbda : float
        A parameter that balances the influence of length and curvature in the new parameterization.
    Returns
    -------
    signed_curvature_arclength : np.array
        An array of shape (N,) representing the signed curvature of the curve in the new parameterization.
    """

    _, dim = curve.shape
    assert dim == 2, "The input curve must be 2D."

    # --- First Step: Arc Length Parameterization ---
    length = compute_length(curve)
    newdelta, arc_length_parametrized_curve = reparametrize_by_arc_length(curve, N, normalized = normalized)
    f_seconde = gs.zeros(arc_length_parametrized_curve.shape)
    for i in range(2):
        for j in range (1,N-1, 1):
            f_seconde[j,i] = (
                arc_length_parametrized_curve[j+1,i] - 2*arc_length_parametrized_curve[j,i] + arc_length_parametrized_curve[j-1,i])*(gs.power((N-1),2)/(length**2))
        f_seconde[0,i] = 0
        f_seconde[-1,i] = 0

    curvature = gs.linalg.norm(f_seconde, axis = 1 ) #Curvature = increment of the velocity vect
    curvedelta = gs.zeros(N)

    # --- Step 3: Curvature-weighted arc length ---
    if lmbda is None:
        weights = curvature
    else:
        weights = lmbda*length + curvature
        #weights = lmbda + curvature
    total_curve_arc_length = np.trapz(weights, newdelta)
    for s in range(1,N,1):
        #With the 1 + curvature
        curvedelta[s-1] = np.trapz( weights[:s], newdelta[:s]) / total_curve_arc_length
        # the new arc length parameterization
        #curvedelta[s] = np.trapz( newdelta[1:s], lmbda*length + curvature[1:s]) / total_curve_arc_length
    curvedelta[-1] = 1.0
    newcurvedelta = gs.linspace(0,1,N) #uniform parameterization
    function_parametrized_prop_to_curv_arc_length = gs.zeros((N, dim))
    for i in range(dim):
        inter_func2 = CubicSpline(curvedelta, arc_length_parametrized_curve[:,i])
        #inter_func2 = interp1d(curvedelta, arc_length_parametrized_curve[:,i], kind='linear')
        #inter_func2 = interp1d(curvedelta, arc_length_parametrized_curve[:,i], kind='cubic', fill_value="extrapolate")
        function_parametrized_prop_to_curv_arc_length[:,i]  = inter_func2(newcurvedelta)

    f_prime = gs.zeros(function_parametrized_prop_to_curv_arc_length.shape)
    for i in range(2):
        for j in range(1,N-1, 1):
            f_prime[j,i] = (function_parametrized_prop_to_curv_arc_length[j+1,i] - function_parametrized_prop_to_curv_arc_length[j-1,i])*((N-1)/(2*length))
        f_prime[0,i] = (arc_length_parametrized_curve[1,i] - arc_length_parametrized_curve[0,i])*(N-1)/length
        f_prime[-1,i] = (arc_length_parametrized_curve[-1,i] - arc_length_parametrized_curve[-2,i])*(N-1)/length
    
    norm_f_prime = gs.power(f_prime[:,0],2) + gs.power(f_prime[:,1],2)
    #max_norm_f_prime = gs.max(norm_f_prime)
    #min_norm_f_prime = gs.min(norm_f_prime)

    #HERE WE MANAGE THE SIGN
    h = gs.zeros(N)
    signed_curvature_arclength = gs.zeros(N)
    for s in range(1,N,1):
        v = f_prime[s,:]
        n_tilde = gs.array([-f_prime[s,1], f_prime[s,0]])
        h[s] = gs.dot(f_seconde[s,:], n_tilde)
        sign =-1 if h[s] < 0 else 1
        signed_curvature_arclength[s] = sign*curvature[s]
    signed_curvature_arclength = interp1d(curvedelta, signed_curvature_arclength,  kind='cubic', fill_value="extrapolate")(newcurvedelta)

    return function_parametrized_prop_to_curv_arc_length, curvedelta, signed_curvature_arclength

def reparametrize_by_arc_length(curve, N, normalized = True):
    """ Reparameterize a 2D curve by its arc length.
    Parameters
    ----------
    curve : np.array
        An array of shape (nb_frame, 2) representing the 2D curve.
    N : int
        The number of points in the output curve.
    Returns
    -------
    newdelta : np.array
        An array of shape (N,) representing the new arc length parameterization.
    arc_length_parametrized_curve : np.array
        An array of shape (N, 2) representing the curve reparameterized by arc length.
    """
    dim = curve.shape[1]
    length = compute_length(curve)
    deltas = gs.linalg.norm(np.diff(curve, axis=0), axis=1) + 1e-6
    deltas = np.insert(deltas, 0, 0.0)
    cumdelta = gs.cumsum(deltas)
    if normalized:
        cumdelta = cumdelta/length
        newdelta = gs.linspace(0, 1, N)
    else:
        newdelta = gs.linspace(0, length, N) 
    #newdelta[-1] = cumdelta[-1] # ensure the last point matches exactly

    # Interpolate to uniform arc-length
    arc_length_parametrized_curve = gs.zeros((N, dim))
    for i in range(dim):
        arc_length_parametrized_curve[:, i] = interp1d(cumdelta, curve[:, i], kind="linear", fill_value="extrapolate")(newdelta)
    return newdelta,arc_length_parametrized_curve

def projection_clock(curve, number_of_angles = 20, lmbda = None, visualize = False):
    """
    Project a 2D curve onto a clock parameterization with a = 20.
    Parameters
    ----------
    curve : np.array
        An array of shape (nb_frame, 2) representing the 2D curve.
    number_of_angles : int
        The number of angles in the clock parameterization (default is 20).
    lmbda : float
        A parameter that balances the influence of length and curvature in the new parameterization (default is None).
    Returns
    -------
    angles : np.array
        An array of shape (nb_frame,) representing the angles in the clock parameterization.
    """

    nb_frames, dim = curve.shape
    N = nb_frames
    curve2 = gs.zeros((N, dim))
    if compute_area(curve, absolute = False) < 0: 
        curve = np.flipud(curve)
    newdelta, curve2 = reparametrize_by_arc_length(curve, N)    
    curve2 = get_max_y_and_roll(curve2)
    area = compute_area(curve2)
    curve2 = translate_center_of_mass(curve2)
    curve3 = curve2/np.sqrt(np.abs(area))
    plt.plot(curve3[:,0], curve3[:,1], 'b-')
    curve3 = rotate_axis(curve3)
    plt.plot(curve3[:,0], curve3[:,1], 'r-')
    plt.axis('equal')
    plt.show()
    curve3 = check_and_shift_center_of_mass(curve3, verbose = visualize)
    angles = extract_angle_sequence(curve3)
    indices = extract_uniform_angles(number_of_angles, angles) #Indixes of the curve points closest to the unif ang
    clock_param_fun_a = gs.zeros((N, dim))
    for k in range(len(indices) - 1):
        argument = newdelta[indices[k]:indices[k + 1] + 1] #Taking a portion of the uniform param.
        unif_subset = int(np.floor(N/number_of_angles))
        newdel2 = np.linspace(argument[0], argument[-1], num = unif_subset)
        
        # Interpolate for each dimension

        for d in range(dim):
            portion_of_x_and_y = curve3[indices[k]:indices[k + 1] + 1, d]
            f_interp = interp1d(argument, portion_of_x_and_y, kind='linear')
            clock_param_fun_a[(k)*unif_subset:(k+1)*unif_subset, d] = f_interp(newdel2)


        # Plot the second coordinate
        if visualize:
            plt.plot(
                clock_param_fun_a[(k) * unif_subset: ( k + 1 ) * unif_subset, 0], 
                clock_param_fun_a[(k) * unif_subset: ( k + 1 ) * unif_subset, 1],
                '*',
                color=colorbar_rainbow(k / (len(indices) - 1)))
            #Adding lines for the subsections
            plt.plot([0,clock_param_fun_a[( k ) * unif_subset, 0]],
                    [0,clock_param_fun_a[( k ) * unif_subset, 1]],
                    'k--')
    if visualize:
        plt.title(f'Clock parameterization with number of angles = {number_of_angles}')
        plt.axis('equal')
        plt.show()

    clock_param_fun_a_lambda = gs.zeros((clock_param_fun_a.shape[0],dim))
    if lmbda is not None:
        for k in range(indices.shape[0] - 1):
            #seg = curve3[indices[k]:indices[k + 1] + 1]
            argument = newdelta[indices[k]:indices[k + 1] + 1]
            newdel2 = np.linspace(argument[0], argument[-1], num = unif_subset)
            clock_param_fun_a_lambda[(k)*unif_subset:(k+1)*unif_subset], _, _ = projection_prop_curv_lambda_length_in_R2(clock_param_fun_a[k*unif_subset:(k + 1)*unif_subset], unif_subset, lmbda, normalized = True)
            if visualize:
                plt.plot(
                clock_param_fun_a_lambda[(k) * unif_subset: ( k + 1 ) * unif_subset, 0], 
                clock_param_fun_a_lambda[(k) * unif_subset: ( k + 1 ) * unif_subset, 1],
                '*',
                color=colorbar_rainbow(k / (len(indices) - 1)))
            #Adding lines for the subsections
                plt.plot(
                    [0,clock_param_fun_a_lambda[( k ) * unif_subset, 0]],
                    [0,clock_param_fun_a_lambda[( k ) * unif_subset, 1]],
                    'k--')
        if visualize:
            plt.title(f'Clock parameterization with number of angles = {number_of_angles} and lambda = {lmbda}')
            plt.axis('equal')
            plt.show()

    return clock_param_fun_a,clock_param_fun_a_lambda

def extract_uniform_angles(a, angles):
    """ Extract indices corresponding to uniform angles in a 2D curve.
    Parameters
    ----------
    a : int
        The number of uniform angles to extract.
    angles : np.array
        An array of shape (nb_frames,) representing the angles of the curve.
    Returns
    -------
    indices : np.array
        An array of shape (a+1,) representing the indices of the curve corresponding to uniform angles.
    """
    a = a+1
    flag = False
    nb_frames = angles.shape[0]
    angles_uniform = gs.zeros(a-1)
    indices = gs.zeros(a, dtype=int)
    indices[0] = 0 # TOCHECK
    for s in range(a-1):
        angles_uniform[s] = angles[0] + (s+1)* 2*np.pi/(a-1)
    
    j = 0
    for i in range(1, nb_frames-1, 1):
        if np.max(angles[:i+1]) > angles_uniform[j]:
            indices[j+1] = i
            if j == angles_uniform.shape[0]-1:
                break
            else:
                j += 1
        

    indices[-1] = nb_frames-1
    for ii in range(indices.shape[0]-1):
        if indices[ii] == indices[ii+1] - 1 or indices[ii] == indices[ii+1] or indices[ii]> indices[ii+1] :
            #indices[ii+1] = indices[ii] + 3
            flag = True
    if flag:
            message = Fore.YELLOW + "WARNING: It seems the center of mass is not well positioned within the shape." + Style.RESET_ALL
            warnings.warn(message, UserWarning)
    if indices[-1] != nb_frames - 1:
        raise ValueError(f"Something wrong during the calculation of uniform angles, the last indices does not correspond to {nb_frames-1}. \nCheck the extract_uniform_angles function.")
    

    return indices

def extract_angle_sequence(curve):
    """ Extract a sequence of angles from a 2D curve.
    Parameters
    ----------
    unite_curve : np.array
        An array of shape (nb_frames, 2) representing the 2D curve with unit length.
    Returns
    -------
    angles : np.array
        An array of shape (nb_frames,) representing the angles of the curve.
    """
    nb_frames, dim = curve.shape
    unite_curve = gs.zeros((nb_frames, dim))
    for i in range(nb_frames):
        unite_curve[i,:] = curve[i,:] /(gs.linalg.norm(curve[i,:]))
    n = gs.zeros(2)
    # Compute the sequence of angles
    angles = gs.zeros(nb_frames)
    for i in range(1, nb_frames, 1):
        angle_vert = unite_curve[i,:] - gs.dot(unite_curve[i,:], unite_curve[i-1,:])*unite_curve[i-1,:]
        n[0] = - unite_curve[i-1,1]
        n[1] = unite_curve[i-1,0]
        sinus_angle = gs.dot(angle_vert, n)
        dot_val = gs.dot(unite_curve[i, :], unite_curve[i-1, :])
        dot_val = np.clip(dot_val, -1.0, 1.0)  # This is added to avoir numerical errors
        if sinus_angle >= 0:
            angles[i] = angles[i-1] + gs.arccos(dot_val)
        else:
            angles[i] = angles[i-1] - gs.arccos(dot_val)
    if any(np.isnan(angles)):
        raise ValueError("Something wrong happened during the arcosine calculus.")
    return angles

def curvature_plot(f, new_curve, curvdel, signed_curvature):
    """ Plot the original curve, reparameterized curve, curvature-weighted arc-length parameter, and signed curvature.
        Parameters
    ----------
    f : np.array
        Original curve of shape (N, 2).
    new_curve : np.array
        Reparameterized curve of shape (N, 2).
    curvdel : np.array
        Curvature-weighted arc-length parameter of shape (N,). 
    signed_curvature : np.array
        Signed curvature of shape (N,).
    """
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Testing projection_prop_curv_lambda_length_in_R2", fontsize=14)

    # Plot original vs. reparameterized curve
    axs[0].plot(f[:, 0], f[:, 1], 'b-', label="Original curve", linewidth=1.2)
    axs[0].plot(new_curve[:,0], new_curve[:,1], 'r*', label="Reparameterized curve",)
    #axs[0].plot(new_curve[:, 0], new_curve[:, 1], 'r--', label="Reparameterized curve", linewidth=1.2)
    axs[0].axis("equal")
    axs[0].legend()
    axs[0].set_title("Curve comparison")

    # Plot curvdel
    axs[1].plot(curvdel, 'k.-')
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("Curvature-weighted arc-length parameter")
    axs[1].set_title("curvdel")

    # Plot signed curvature
    axs[2].plot(signed_curvature, 'm-', linewidth=1.2)
    axs[2].set_xlabel("Index")
    axs[2].set_ylabel("Signed curvature")
    axs[2].set_title("Signed curvature after reparametrization")
    plt.tight_layout()
    plt.show()

######### MAIN ###########

if __name__ == "__main__":
    import scipy.io
    PATH = r"C:\Users\LONGA\Downloads\leaves_parameterized.mat"
    #path = #INSERT THE PATH TO YOUR DATA
    A = scipy.io.loadmat(PATH)
    leaves = A['leaves_parameterized']
    curve = leaves[67,:,:]

    
    nb_frames, dim = curve.shape
    N = nb_frames
    curve2 = gs.zeros((N, dim))
    if compute_area(curve, absolute = False) < 0:
        curve = np.flipud(curve)
    newdelta, curve2 = reparametrize_by_arc_length(curve, N)
    curve2 = get_max_y_and_roll(curve2)
    area = compute_area(curve2)
    curve2 = translate_center_of_mass(curve2)
    curve3 = curve2/np.sqrt(np.abs(area))
    plt.plot(curve3[:,0], curve3[:,1], 'b-')
    curve3 = rotate_axis(curve3)
    #projection_clock_a_lmbda(curve3, n_subdivision = 0, lmbda = 0.5, visualize = True )
    testing_curve = reparametrize_by_curvature(curve3, lmbda = 1, unif_subset = nb_frames )
    plt.plot(testing_curve[:,0], testing_curve[:,1], 'g*')
    plt.axis('equal')
    plt.show()
    
    #_,_ = projection_clock(curve, number_of_angles = 10, lmbda = 0.5, visualize = True)