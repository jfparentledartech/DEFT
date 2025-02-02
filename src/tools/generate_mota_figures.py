import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    mota_overalls = [0.318369, 0.244772, 0.339375, 0.351938, 0.320365, 0.350326, 0.341289, 0.341310, 0.337836, 0.371260,
                     0.333749, 0.362327, 0.328862, 0.353779, 0.325981, 0.335216, 0.356586, 0.338512, 0.350420, 0.346624,
                     0.367963, 0.352281, 0.364188, 0.344960, 0.358001, 0.343202]

    mota_cars = [0.336402, 0.203054, 0.354559, 0.372608, 0.352105, 0.342430, 0.349686, 0.351527, 0.373691, 0.395387,
                 0.309472, 0.383979, 0.305538, 0.345210, 0.318605, 0.357483, 0.375280, 0.349217, 0.387192, 0.343585,
                 0.348603, 0.315681, 0.372356, 0.347448, 0.363945, 0.335788]

    mota_trucks = [0.138966, 0.124302, -0.104749, 0.108240, 0.136872, 0.130587, -0.067039, 0.042598, 0.050978, 0.018855,
                   -0.093575, 0.017458, 0.150838, 0.171788, 0.076816, 0.035615, 0.053771, 0.074022, 0.114525, 0.053073,
                   0.046788, -0.030028, 0.031425, -0.039106, 0.120112, 0.187151]

    mota_bus = [-0.012259, 0.089317, -0.056042, -0.147110, -0.110333, -0.021016, -0.010508, -0.138354, -0.001751,
                -0.084063, 0.043783, 0.001751, -0.056042, 0.019264, 0.049037, -0.017513, -0.029772, 0.077058, 0.003503,
                -0.175131, -0.098074, -0.073555, -0.159370, 0.021016, -0.035026, 0.036778]

    mota_pedestrian = [0.347722, 0.281415, 0.365755, 0.366440, 0.330407, 0.374780, 0.368480, 0.357856, 0.343054,
                       0.384474, 0.371091, 0.376901, 0.353629, 0.374796, 0.350757, 0.347967, 0.368790, 0.359733,
                       0.356910, 0.370161, 0.398427, 0.390773, 0.382531, 0.363584, 0.390185, 0.369606]

    mota_motorcyclist = [0.019481, -0.050325, 0.150974, 0.154221, 0.087662, 0.202922, 0.275974, 0.284091, 0.185065,
                         0.314935, 0.298701, 0.277597, 0.159091, 0.112013, 0.199675, 0.323052, 0.293831, 0.290584,
                         0.370130, 0.267857, 0.292208, 0.201299, 0.250000, 0.116883, -0.092532, 0.254870]

    mota_cyclist = [0.056507, 0.117223, 0.161707, 0.256988, 0.150887, 0.241960, 0.115720, 0.280433, 0.256387, 0.255185,
                    0.203186, 0.206192, 0.318305, 0.308687, 0.216111, 0.200781, 0.272317, 0.130147, 0.157800, 0.302675,
                    0.276525, 0.314097, 0.301773, 0.333033, 0.063120, 0.131951]

    mota_van = [-0.004858, 0.064777, -0.039676, 0.036437, 0.094737, 0.106073, 0.080162, 0.051822, 0.059919, 0.134413,
                0.024291, 0.182996, 0.120648, 0.110931, 0.038057, 0.081781, 0.119838, 0.058300, 0.146559, -0.013765,
                0.162753, 0.081781, 0.123887, 0.106073, 0.104453, 0.135223]

    plt.title('Validation MOTA')
    epochs = range(1, len(mota_overalls)+1)
    plt.plot(epochs, mota_overalls, label='overall')
    plt.plot(epochs, mota_cars, label='car')
    plt.plot(epochs, mota_trucks, label='truck')
    plt.plot(epochs, mota_bus, label='bus')
    plt.plot(epochs, mota_pedestrian, label='pedestrian')
    plt.plot(epochs, mota_motorcyclist, label='motorcyclist')
    plt.plot(epochs, mota_cyclist, label='cyclist')
    plt.plot(epochs, mota_van, label='van')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MOTA')
    plt.show()

    print('Best epoch selecting max overall')
    print('                         num_frames      mota      motp  precision    recall \n\
    car 10                15663  0.395387  0.692369   0.758644  0.656631 \n\
    truck 10              15663  0.018855  0.963396   0.540804  0.310056 \n\
    bus 10                15663 -0.084063  1.012684   0.465296  0.316988 \n\
    pedestrian 10         15663  0.384474  0.709613   0.860114  0.583703 \n\
    motorcyclist 10       15663  0.314935  0.843753   0.830303  0.444805 \n\
    cyclist 10            15663  0.255185  0.595353   0.658320  0.624286 \n\
    van 10                15663  0.134413  0.865970   0.629583  0.403239 \n\
    OVERALL              109641  0.371260  0.704932   0.808352  0.597250')

    print('Second best epoch selecting max overall')
    print('                         num_frames      mota      motp  precision    recall \n\
    car 22                15663  0.372356  0.723010   0.744374  0.650783 \n\
    truck 22              15663  0.031425  0.853227   0.550523  0.331006 \n\
    bus 22                15663 -0.159370  1.008200   0.422274  0.318739 \n\
    pedestrian 22         15663  0.382531  0.705255   0.848389  0.584552 \n\
    motorcyclist 22       15663  0.250000  0.801717   0.737838  0.443182 \n\
    cyclist 22            15663  0.301773  0.596168   0.715347  0.578599 \n\
    van 22                15663  0.123887  0.672361   0.613048  0.441296 \n\
    OVERALL              109641  0.364188  0.709513   0.798622  0.595326')
