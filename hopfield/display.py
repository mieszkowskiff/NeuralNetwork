import matplotlib.pyplot as plt
from PIL import Image

def display(x, dims):
    plt.imshow(x.reshape(dims[1], dims[0]), cmap='gray', norm='linear')
    plt.show()

def save_img(x, dims, save_as):
    #plt.imshow(x.reshape(dims[1], dims[0]), cmap='gray', norm='linear')
    fig, ax = plt.subplots()
    ax.imshow(x.reshape(dims[1], dims[0]), cmap='gray', norm='linear')
    fig.savefig(save_as)
    plt.close(fig)

def user_choose_display(num_of_patterns):
    question = 'Which training pattern would you like to see? Choose [1-' + str(num_of_patterns) + '] : '
    bool = True
    
    while bool:
        try:
            tmp = int(input(question))
            if (tmp < 1 or tmp>num_of_patterns):
                print("Invalid integer range.")
            else:
                bool = False
        except ValueError:
            print("Invalid input. Please enter an integer.")
    return tmp-1



from PIL import Image

def convert_to_bitmap(image_path, output_size):
    """
    Converts a .png image to a bitmap of the given size.
    
    Parameters:
        image_path (str): Path to the input .png image.
        output_size (tuple): Desired size (width, height) of the bitmap.
    
    Returns:
        list: A 2D list representing the bitmap.
    """
    try:
        # Open the image
        img = Image.open(image_path).convert('L')  # Convert image to grayscale
        
        # Resize the image using the LANCZOS resampling method
        img = img.resize(output_size, Image.Resampling.LANCZOS)
        
        # Convert image to binary (black and white) based on a threshold
        threshold = 128  # Threshold value for binary conversion
        bitmap = img.point(lambda p: 1 if p > threshold else 0)
        
        # Convert to a 2D list
        bitmap_list = [[bitmap.getpixel((x, y)) for x in range(bitmap.width)] for y in range(bitmap.height)]
        
        return bitmap_list
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def save_bitmap_to_file(bitmap, output_file):
    """
    Saves a 2D bitmap to a text file.
    
    Parameters:
        bitmap (list): A 2D list representing the bitmap.
        output_file (str): Path to the output file to save the bitmap.
    """
    try:
        with open(output_file, 'w') as file:
            for row in bitmap:
                file.write(' '.join(map(str, row)) + '\n')
        print(f"Bitmap saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    input_image = "cat.png"  # Path to your .png image
    output_size = (32, 32)  # Desired size of the bitmap (width, height)
    output_file = "cat_bitmap.txt"  # Path to save the bitmap
    
    # Convert image to bitmap
    bitmap = convert_to_bitmap(input_image, output_size)
    
    if bitmap:
        # Save the bitmap to a file
        save_bitmap_to_file(bitmap, output_file)
