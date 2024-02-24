from matplotlib import pyplot as plt
import numpy as np
import cv2

def j_divergence(membership_matrix, pixel_values, cluster_centers, fuzzy_num):
    return np.sum(np.power(membership_matrix,fuzzy_num)*np.square(pixel_values-cluster_centers.T))

def class_means(membership_matrix, pixel_values, fuzzy_num) :

    powered_membership_matrix = membership_matrix ** fuzzy_num
    c_divergence = powered_membership_matrix.T@pixel_values
    c_divergence = c_divergence.T / np.sum(powered_membership_matrix, axis = 0)
    return c_divergence.T

def update_membership_values(pixel_values, cluster_centers, no_of_segments, fuzzy_num):

    size_of_pixel_values = pixel_values.size

    distance_matrix = np.zeros((size_of_pixel_values, no_of_segments))
    for i in range(no_of_segments):
        distance_matrix[:, i] = (pixel_values**2  - 2 * cluster_centers[i] * pixel_values + cluster_centers[i] ** 2).flatten()

    distance_matrix[distance_matrix <= 0] = 1e-10

    reversed_distance_matrix = ( 1 / distance_matrix ) ** (1 / (fuzzy_num - 1)) 
    sum_of_distances = np.sum(reversed_distance_matrix, axis = 1)

    membership_values = np.zeros((size_of_pixel_values, no_of_segments))

    for i in range(no_of_segments):
        membership_values[:, i] = reversed_distance_matrix[:, i] / sum_of_distances
     
    return membership_values


def c_means_clustering_algo(original_image, binary_mask, no_of_segments, fuzzy_num = 1.6, no_of_iterations = 20):
    np.random.seed(0)
    original_image = original_image*binary_mask
    pixel_values = np.float32(original_image.reshape((-1,1)))
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    return_val, labels, centers = cv2.kmeans(pixel_values, no_of_segments, None, term_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    kmeans_labels = np.copy(labels)
    kmeans_centers = np.copy(centers)
    ground_truth = np.copy(labels).flatten()

    initial_membership_matrix = np.random.rand(pixel_values.shape[0], centers.shape[0])
    initial_membership_matrix = initial_membership_matrix / initial_membership_matrix.sum(axis=1)[ :,None]
    membership_matrix = initial_membership_matrix

    cost_value = 0
    cost_array = []

    for i in range(no_of_iterations):
        membership_matrix = update_membership_values(pixel_values,centers,no_of_segments,fuzzy_num)
        centers = class_means(membership_matrix,pixel_values,fuzzy_num)
        cost_value = j_divergence(membership_matrix,pixel_values,centers,fuzzy_num)
        cost_array.append(cost_value)
        print(f"Iteration {i}: { cost_value }")

    labels = np.argmax(membership_matrix, axis = 1)
    segmented_labels = np.copy(labels).flatten()
    if np.all(centers >= 0) and np.all(centers <= 1):
        centers = centers * 255
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((original_image.shape))

    if np.all(kmeans_centers >= 0) and np.all(kmeans_centers <= 1):
        kmeans_centers = kmeans_centers * 255
    kmeans_centers = np.uint8(kmeans_centers)
    kmeans_segmented_data = kmeans_centers[kmeans_labels.flatten()]
    kmeans_segmented_data = kmeans_segmented_data.reshape((original_image.shape))
    dice_coefficients = np.zeros(no_of_segments)
    for i in range(no_of_segments):
        dice_value = 0
        for j in range(no_of_segments) :
            dice_value = max(dice_value,np.sum(segmented_labels[ground_truth==i]==j)*2.0 / (np.sum(segmented_labels[segmented_labels==j]==j) + np.sum(ground_truth[ground_truth==i]==i)))
        dice_coefficients[i] = dice_value
    print("Mean Dice Coefficient: ", np.mean(dice_coefficients))
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(segmented_image, cmap='gray')
    ax[1].set_title("FCM Algorithm")
    ax[2].imshow(kmeans_segmented_data, cmap = 'gray')
    ax[2].set_title("K-Means Clustering Algorithm")
    plt.show()

    return segmented_image, cost_array