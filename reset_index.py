import pickle
import numpy as np

def main():
    file = open("state", "rb")
    data = pickle.load(file)
    file.close()
    data["index"] = 1
    file = open('state', 'wb')
    pickle.dump(data, file)
    file.close()




if __name__ == "__main__":
    main()
