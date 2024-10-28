import numpy as np
import csv
import math
import string

x=[0.001*i for i in range(0, 1001)]
x=np.array(x)
y=[5*xx-5*pow(xx,2)+0.05*math.sin(70*xx) for xx in x]
# Open the text file for writing
with open('./data_gen_noise.txt', mode='w') as txt_file:
    for i in range(0, len(x)):
        # Join each row into a single line of text
        tmp = (str(x[i]), ", ", str(y[i]), '\n')
        #print(",".join(tmp))
        txt_file.write("".join(tmp))
txt_file.close()

