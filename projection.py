import geomstats.backend as gs
from utils import compute_length, get_max_y, translate_center_of_mass, rotate_ellipse, compute_area, rotate_ellipse_surface, rotate_axis, get_max_y_and_roll
from  scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import numpy as np
from visualize import colorbar_rainbow

def proj_clock_etapesbis(curve, N, a, lmbda):
    """ TO DO """
    #STUBB
    nb_frames, dim = curve.shape
    param2 = gs.zeros((nb_frames, dim))
    clock_parametrized_function_a = gs.zeros((nb_frames, dim))
    fig_nb = 0
    
    long = compute_length(curve)

    length = long # IS THIS USEFUL? 
    delta = gs.zeros(nb_frames)
    for s in range(2,nb_frames,1):
        delta[s] = gs.linalg.norm(curve[s,:]-curve[s-1,:]) + 1e-6
    
    cumdelta = gs.cumsum(delta)/length #arch length parameterization
    newdelta = gs.linspace(0,1, 1/nb_frames) #uniform parameterization

    for i in range(dim):
        param2 = gs.interp(newdelta, cumdelta, curve[:,i])
    etape0 = param2.copy()

    #Centering by highest point (or just the first one?)
    c, etape2_first_point = get_max_y(param2)

    #centering by center of mass
    etape2_center_of__mass = translate_center_of_mass(param2)

    #Normalize by area
    area = compute_area(param2)
    etape2_area = param2/gs.sqrt(gs.abs(area))

    #gravity centering
    center = gs.zeros(2)
    for i in range(nb_frames-1):
        center[0] += 0.5*etape2_area[i,:]**2*(etape2_area[i+1]-etape2_area[i,1])
        center[1] -= 0.5*etape2_area[i,1]**2*(etape2_area[i+1]-etape2_area[i,0])
    param3 = etape2_area - center
    param3 = param3*gs.sqrt(abs(area))

    #Ellips and axis rotation
    etape4_ellipse = rotate_ellipse(param3)
    etape4_ellipse_surface = rotate_ellipse_surface(param3)
    etape4 = rotate_axis(etape4_ellipse_surface)
    # Final step: clock parametrization with lambda
    clock_parameterized_a_lambda = gs.zeros_like(clock_parametrized_function_a)
    indices = gs.linspace(0, nb_frames-1, a, dtype=int)


    pass


    return True
    #return (etape0, etape1, etape2_first_point, etape2_center_of_mass,
    #etape2_gravity, etape3_length, etape3_area,
    #etape4_ellipse, etape4_ellipse_surface, etape4, etape5)

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

    nb_frame, dim = curve.shape
    assert dim == 2, "The input curve must be 2D."

    # --- First Step: Arc Length Parameterization ---
    length = compute_length(curve)
    newdelta, arc_length_parametrized_curve = reparametrize_by_arc_length(curve, N, normalized = normalized)

    # --- Step 2: Second derivative (curvature numerator)
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
        inter_func2 = interp1d(curvedelta, arc_length_parametrized_curve[:,i], kind='cubic', fill_value="extrapolate")
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
    curve2 = gs.zeros((nb_frames, dim))
    N = nb_frames

    newdelta, curve2 = reparametrize_by_arc_length(curve, nb_frames)    
    curve2 = get_max_y_and_roll(curve2)
    area = compute_area(curve2)
    curve2 = translate_center_of_mass(curve2)
    curve3 = curve2/np.sqrt(np.abs(area))

    # If the point at 1/4 is on the right, flip the curve, can I check just the first point? Or withthe Area?
    if curve3[int(np.ceil(nb_frames/4)) - 1, 0] > 0:
        curve3 = np.flipud(curve3)
    angles = extract_angle_sequence(curve3)
    indices = extract_uniform_angles(number_of_angles, angles) #Indixes of the curve points closest to the unif ang

    for k in range(len(indices) - 1):
        argument = newdelta[indices[k]:indices[k + 1] + 1] #Taking a portion of the uniform param.

        # Create uniform subdivision
        unif_subset = int(np.floor(N/number_of_angles))
        newdel2 = np.linspace(argument[0], argument[-1], num = unif_subset)
        clock_param_fun_a = gs.zeros((nb_frames, dim))
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
            clock_param_fun_a_lambda[(k)*unif_subset:(k+1)*unif_subset], _, _ = projection_prop_curv_lambda_length_in_R2(curve3[indices[k]:indices[k + 1]], unif_subset, lmbda, normalized = True)
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
    nb_frames = angles.shape[0]
    angles_uniform = gs.zeros(a)
    indices = gs.zeros(a, dtype=int)
    indices[0] = 0 # TOCHECK
    for s in range(a-1):
        angles_uniform[s] = angles[0] + (s+1)* 2*np.pi/(a-1)
    
    j = 0
    for i in range(1, nb_frames-1, 1):
        if np.max(angles[:i]) > angles_uniform[j] and np.max(angles[:i]) < 2*np.pi:
            indices[j+1] = i
            j += 1

    indices[j+1] = nb_frames
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
        if sinus_angle >= 0:
            angles[i] = angles[i-1] + gs.arccos(gs.dot(unite_curve[i,:], unite_curve[i-1,:]))
        else:
            angles[i] = angles[i-1] - gs.arccos(gs.dot(unite_curve[i,:], unite_curve[i-1,:]))
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
    curve = leaves[75,:,:]
    N_input = 200 
    N_output = 300
    lmbda = 50

    # Example curve: ellipse
    t = np.linspace(0, 2*np.pi, N_input)
    a, b = 3.0, 1.0
    x = a * np.cos(t)
    y = b * np.sin(t)
    #curve = np.column_stack([x, y])
    f = curve

    new_curve, curvdel, signed_curvature = projection_prop_curv_lambda_length_in_R2(curve, N_output, lmbda)
    curvature_plot(f, new_curve, curvdel, signed_curvature)

    _,_ = projection_clock(curve, number_of_angles = 12, lmbda = lmbda)