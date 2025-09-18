import geomstats.backend as gs
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import colormaps

def clustering_visualization(X, Y, index = 0):
    X = X.reshape(X.shape[0],-1)
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(X)
    for j in range(X_embedded.shape[0]):
        plt.plot(X_embedded[j][0],X_embedded[j][1], '*', color=colormaps['hsv'](Y[j]/15))
    plt.title(f"TSNE plot for the step{index}")
    plt.axis('equal')
    plt.show()
    return X_embedded

    
def colorbar_rainbow(u):
    """
    Map a scalar u in [0, 1] to an RGB rainbow color.
    
    Parameters:
        u : float or array-like
            Input scalar(s) in the range [0, 1].
            
    Returns:
        col : array of shape (3,) or (len(u), 3)
            RGB color(s) in range [0, 1].
    """
    # Scale u to range [1, 1530]
    x = gs.array(u) * 1529 + 1
    
    r = gs.zeros_like(x, dtype=float)
    g = gs.zeros_like(x, dtype=float)
    b = gs.zeros_like(x, dtype=float)
    
    # Red → Yellow
    mask = (x >= 0) & (x < 255)
    r[mask] = 255
    g[mask] = x[mask]
    b[mask] = 0
    
    # Yellow → Green
    mask = (x >= 255) & (x < 510)
    r[mask] = 510 - x[mask]
    g[mask] = 255
    b[mask] = 0
    
    # Green → Cyan
    mask = (x >= 510) & (x < 765)
    r[mask] = 0
    g[mask] = 255
    b[mask] = x[mask] - 510
    
    # Cyan → Blue
    mask = (x >= 765) & (x < 1020)
    r[mask] = 0
    g[mask] = 1020 - x[mask]
    b[mask] = 255
    
    # Blue → Magenta
    mask = (x >= 1020) & (x < 1275)
    r[mask] = x[mask] - 1020
    g[mask] = 0
    b[mask] = 255
    
    # Magenta → Red
    mask = (x >= 1275) & (x <= 1530)
    r[mask] = 255
    g[mask] = 0
    b[mask] = 1530 - x[mask]
    
    # Normalize to [0,1]
    R = r / 255
    G = g / 255
    B = b / 255
    
    # Stack to RGB
    col = gs.stack([R, G, B], axis=-1)
    
    return col

def plot_curve(curve):
    """
    Return the plot of the curve
    """
    plt.plot(curve[:,0], curve[:,1], "g-")
    plt.axis('equal')
    plt.show()