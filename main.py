import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fcm
import fgfcm
import enfcm
import fcm_s1


def validate(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError(f"{f} does not exist!")
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seg", type=int, help="Number of segments to use in the image")
    parser.add_argument("--meth", choices=["fcm", "fgfcm", "enfcm", "fcm_s1"], help="Different flavors of FCM that can be used")
    parser.add_argument("--img", type=validate, help="Specify the path to the image file")
    parser.add_argument("--fuzzy", default=1.6, type=float, help="Specify the fuzziness factor")
    parser.add_argument("--save", action="store_true", help="Output images will be saved in results folder with name {file_name}_{method}.png")
    parser.add_argument("--iter", default = 20, type = int, help="Number of iterations")

    args = parser.parse_args()
    no_of_segments = args.seg
    type_of_method = args.meth
    image_file_name = args.img
    fuzzy_num = args.fuzzy
    save = args.save
    no_of_iterations = args.iter

    if fuzzy_num <= 1:
        sys.exit("ERROR: Fuzziness Factor must be greater than 1.")

    if no_of_iterations <= 0:
        sys.exit("ERROR: Number of iterations must be greater than 0.")

    file_extension = os.path.splitext(os.path.basename( image_file_name ))[1]
    if file_extension in [".png", ".jpeg", ".jpg"]:
        original_image  = cv2.imread(image_file_name)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        original_image = original_image / 255
        binary_mask = np.ones(original_image.shape)

    else:
        sys.exit("ERROR: Invalid image file format. Please provide a valid image file.")


    if type_of_method == 'fcm':
        segmented_image, cost_array = fcm.c_means_clustering_algo(original_image, binary_mask, no_of_segments, fuzzy_num, no_of_iterations)
    elif type_of_method == "fgfcm":
        segmented_image, cost_array = fgfcm.c_means_clustering_algo(original_image, binary_mask, no_of_segments, fuzzy_num, no_of_iterations)
    elif type_of_method == 'enfcm':
        segmented_image, cost_array = enfcm.c_means_clustering_algo(original_image, binary_mask, no_of_segments, fuzzy_num, no_of_iterations)
    elif type_of_method == 'fcm_s1':
        segmented_image, cost_array = fcm_s1.c_means_clustering_algo(original_image, binary_mask, no_of_segments, fuzzy_num, no_of_iterations)
    else:
        sys.exit("ERROR: Invalid method name specified.")

    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    if save:
        plt.savefig(f"./Result/{os.path.splitext(os.path.basename(image_file_name))[0]}_{type_of_method}.png")
    plt.show()

    plt.plot(cost_array)
    plt.title("Cost vs Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost Values")
    plt.grid()
    if save:
        plt.savefig(f"./Result/{os.path.splitext(os.path.basename(image_file_name))[0]}_{type_of_method}_cost.png")
    plt.show()



