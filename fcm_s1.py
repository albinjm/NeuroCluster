from matplotlib import pyplot as plt
import numpy as np
import cv2

def j_divergence(membership_matrix, pixel_values, cluster_centers, fuzzy_num,average_pixel_values, alpha):
    return np.sum(np.power(membership_matrix,fuzzy_num)*np.square(pixel_values-cluster_centers.T) + alpha*np.power(membership_matrix, fuzzy_num)*np.square(average_pixel_values-cluster_centers.T))

def class_means(membership_matrix, pixel_values, fuzzy_num, average_pixel_values,alpha) :

    powered_membership_matrix = membership_matrix ** fuzzy_num
    c_divergence = powered_membership_matrix.T@(pixel_values+alpha*average_pixel_values)
    c_divergence = c_divergence.T/((1+alpha)*np.sum(powered_membership_matrix,axis = 0))
    return c_divergence.T

def update_membership_values(pixel_values, cluster_centers, no_of_segments, fuzzy_num,average_pixel_values, alpha):
    
    size_of_pixel_values = pixel_values.size

    distance_matrix = np.zeros((size_of_pixel_values, no_of_segments))
    for i in range(no_of_segments):
        distance_matrix[:, i] = (pixel_values**2  - 2 * cluster_centers[i] * pixel_values + cluster_centers[i] ** 2).flatten()
        distance_matrix[:, i] += alpha*(average_pixel_values**2  - 2 * cluster_centers[i] * average_pixel_values + cluster_centers[i] ** 2).flatten()

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
    average_image = np.zeros(original_image.shape)

    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            sum_of_pixel_values = 0 
            count = 0
            if i != 0 :
                sum_of_pixel_values += original_image[i-1][j]
                count += 1
            if j != 0 :
                sum_of_pixel_values += original_image[i][j-1]
                count += 1
            if i != original_image.shape[0]-1:
                sum_of_pixel_values += original_image[i+1][j]
                count += 1
            if j != original_image.shape[1]-1:
                sum_of_pixel_values += original_image[i][j+1]
                count += 1
            average_image[i,j] = sum_of_pixel_values/count
     
    pixel_values = np.float32(original_image.reshape((-1,1)))
    average_pixel_values = np.float32(average_image.reshape((-1,1)))
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    return_val, labels, centers = cv2.kmeans(pixel_values, no_of_segments, None, term_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    kmeans_labels = np.copy(labels)
    kmeans_centers = np.copy(centers)
    ground_truth = np.copy(labels).flatten()

    initial_membership_matrix = np.random.rand(pixel_values.shape[0],centers.shape[0])
    initial_membership_matrix = initial_membership_matrix/initial_membership_matrix.sum(axis=1)[:,None]

    membership_matrix = initial_membership_matrix

    cost_value = 0
    cost_array = []

    #hyperparameter that controls the amount of influence the average image has on the segmentation process
    alpha = 0.2

    for i in range(no_of_iterations):
        centers = class_means(membership_matrix, pixel_values, fuzzy_num, average_pixel_values, alpha)
        membership_matrix = update_membership_values(pixel_values, centers, no_of_segments, fuzzy_num, average_pixel_values, alpha)
        cost_value = j_divergence(membership_matrix, pixel_values, centers, fuzzy_num, average_pixel_values, alpha)   
        cost_array.append(cost_value)
        print(f"Iteration {i}: { cost_value }")

    labels = np.argmax(membership_matrix,axis = 1)
    seg = np.copy(labels).flatten()
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
        for j in range(no_of_segments):
            dice_value = max(dice_value,np.sum(seg[ground_truth==i]==j)*2.0 / (np.sum(seg[seg==j]==j) + np.sum(ground_truth[ground_truth==i]==i)))
        dice_coefficients[i] = dice_value
    print("Mean Dice Coefficient: ", np.mean(dice_coefficients))
    fig, axs = plt.subplots(1, 3 )
    axs[0].imshow(original_image,cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(segmented_image,cmap='gray')
    axs[1].set_title("FCM_S1 Algorithm")
    axs[2].imshow(kmeans_segmented_data, cmap = 'gray')
    axs[2].set_title("K-Means Clustering Algorithm")
    plt.show()

    return segmented_image, cost_array