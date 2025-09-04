from normalizer import Normalizer
import scipy.io

if __name__ == "__main__":

    path = #INSERT THE PATH TO YOUR DATA
    A = scipy.io.loadmat(path)
    my_first_class = Normalizer(A['leaves_parameterized'])
    print(my_first_class.curve_unite_length(0))
    print('my_first_class')