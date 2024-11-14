import display
import hopfield_net
import read_csv

CONST_LEARNING_METHODS = ['HEBB', 'OJA']
CONST_ACTIVATION_FUNCTIONS = ['signum', 'heavy_side']
CONST_DYNAMICS_TYPE = ['asynchronous', 'synchronous']

if __name__ == '__main__':
    train, dims = read_csv.read_patterns('./data/hopfield/letters-14x20.csv')
    num_of_patterns = train.shape[0]
    
    # display chosen training pattern
    
    display.display(train[display.user_choose_display(num_of_patterns)], dims)
    

    # Hopfield network

    n = dims[0]*dims[1]
    # learning method 0 - Hebb, 1 - Oja
    method = 0
    # activation function 0 - signum, 1 - heaviside
    activation = 0
    # dynamics type 0 - asynchronous, 1 - synchronous
    dynamics = 0

    HN = hopfield_net.HopfieldNet(lm = CONST_LEARNING_METHODS[method],
                                n = n,
                                activation = CONST_ACTIVATION_FUNCTIONS[activation],
                                dynamics = CONST_DYNAMICS_TYPE[dynamics])
    
    HN.training(train)
    #display network weights after training 
    display.display(HN.W, [n, n])

    init_x = train[2]
    last_x = HN.forward(dims, init_x = init_x, epochs = 10, show_vis = 0)
    display.display(last_x, dims)