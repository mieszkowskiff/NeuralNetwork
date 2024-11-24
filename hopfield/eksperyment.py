import display
import hopfield_net
import read_csv
import numpy as np

np.random.seed(0)
CONST_ACTIVATION_FUNCTIONS = ['signum', 'heavy_side']
CONST_DYNAMICS_TYPE = ['asynchronous', 'synchronous']


datasets = ["animals-14x9.csv",
                "large-25x25.csv",
                "large-25x50.csv",
                "letters-14x20.csv",
                "letters-abc-8x12.csv",
                "OCRA-12x30-cut.csv",
                "small-7x7.csv",
                "cats-200x200.csv",
                "gen_pattern-5x5.csv"
        ]

path = "./data/hopfield/eksperyment6/"


train, dims = read_csv.read_patterns(path + "/../" + datasets[8])
num_of_patterns = train.shape[0]

print("Data loaded. Number of patterns: ", num_of_patterns)

# save training data as .png
for i in range(num_of_patterns):
    display.save_img(train[i], dims, path + "train/p" + str(i+1) + ".png")

print("Training data saved as .png.")
# save noise data as .png
X = []
for i in range(num_of_patterns):
    X.append(read_csv.noise(train[i], 0.0))
    display.save_img(X[-1], dims, path + "noise/n" + str(i+1) + ".png")

print("Noise data saved as .png.")
n = dims[0] * dims[1]
# activation function 0 - signum, 1 - heaviside
activation = 0
# dynamics type 0 - asynchronous, 1 - synchronous
dynamics = 1

HN = hopfield_net.HopfieldNet(
        n = n,
        activation = CONST_ACTIVATION_FUNCTIONS[activation],
        dynamics = CONST_DYNAMICS_TYPE[dynamics]
    )


HN.HEBB_training(train)
for i in range(num_of_patterns):
    last_x = HN.forward(dims, init_x = X[i], animation = False)
    display.save_img(last_x, dims, path + "hebb/h" + str(i+1) + ".png")

print("Hebb data saved as .png.")


HN2 = hopfield_net.HopfieldNet(
    n = n,
    activation = CONST_ACTIVATION_FUNCTIONS[activation],
    dynamics = CONST_DYNAMICS_TYPE[dynamics]
)

HN2.OJA_training(train, 30, eta=0.0001)
for i in range(num_of_patterns):
    #wait = input()
    last_x = HN2.forward(dims, init_x = X[i], animation = False)
    display.save_img(last_x, dims, path + "oja/o" + str(i+1) + ".png")
