import display
import hopfield_net
import read_csv

CONST_ACTIVATION_FUNCTIONS = ['signum', 'heavy_side']
CONST_DYNAMICS_TYPE = ['asynchronous', 'synchronous']


if __name__ == '__main__':
    datasets = ["animals-14x9.csv",
                "large-25x25.csv",
                "large-25x50.csv",
                "letters-14x20.csv",
                "letters-abc-8x12.csv",
                "OCRA-12x30-cut.csv",
                "small-7x7.csv"]
    # datasets: 0 - animals, 1 - large(25), 2 - large(50), 3 - letters, 4 - letters abc
    # 5 - OCRA, 6 - small
    train, dims = read_csv.read_patterns('./data/hopfield/' + datasets[5])
    #train = train[[True, False, False, False, True]]
    num_of_patterns = train.shape[0]
    
    for i in range(num_of_patterns):
        display.save_img(train[i], dims, "./data/hopfield/test/train/p" + str(i+1) + ".png")

    X = []
    for i in range(num_of_patterns):
        X.append(read_csv.noise(train[i], 0.1))
        display.save_img(X[-1], dims, ".data/hopfied/test/noise/n" + str(i+1) + ".png")
    
    #train = train[]
    # display chosen training pattern
    # display.display(train[display.user_choose_display(num_of_patterns)], dims)

    # Hopfield network

    n = dims[0] * dims[1]
    # activation function 0 - signum, 1 - heaviside
    activation = 0
    # dynamics type 0 - asynchronous, 1 - synchronous
    dynamics = 0

    HN = hopfield_net.HopfieldNet(
        n = n,
        activation = CONST_ACTIVATION_FUNCTIONS[activation],
        dynamics = CONST_DYNAMICS_TYPE[dynamics]
    )
    
    HN.HEBB_training(train)
    for i in range(num_of_patterns):
        last_x = HN.forward(dims, init_x = X[i], epochs = 200, animation = False)
        display.save_img(last_x, dims, "./data/hopfield/test/hebb/h" + str(i+1) + ".png")

    HN = hopfield_net.HopfieldNet(
        n = n,
        activation = CONST_ACTIVATION_FUNCTIONS[activation],
        dynamics = CONST_DYNAMICS_TYPE[dynamics]
    )


    HN.OJA_training(train, 30)
    for i in range(num_of_patterns):
        #wait = input()
        last_x = HN.forward(dims, init_x = X[i], epochs = 100, animation = False)
        display.save_img(last_x, dims, "./data/hopfield/test/oja/o" + str(i+1) + ".png")
        #wait = input()
        #last_x = HN.forward(dims, init_x = last_x, epochs = 10, animation = False)
        #display.save_img(last_x, dims, "../test/oja/o" + str(i+1) + ".png")
           