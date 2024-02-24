from matplotlib import pyplot as plt
import numpy as np
import cv2

def j_divergence(membership_matrix, pixel_values, cluster_centers, fuzzy_num,gamma):
    distance = pixel_values-cluster_centers.T
    divergence_value = np.sum(np.power(membership_matrix,fuzzy_num)*(np.square(distance)*gamma))
    return divergence_value

def class_means(membership_matrix, pixel_values, fuzzy_num, gamma) :
    pixel_values = pixel_values.reshape((-1,1))
    powered_membership_matrix = membership_matrix ** fuzzy_num
    c_divergence = powered_membership_matrix.T@(pixel_values*gamma)
    c_divergence = c_divergence/(powered_membership_matrix.T@gamma)
    return c_divergence

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
    original_image = original_image * binary_mask
    average_image = np.float32(np.zeros(original_image.shape))

    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            sum_of_squared_differences = 0.0
            count = 0
            if i != 0 :
                sum_of_squared_differences += (original_image[i-1][j]-original_image[i][j])**2
                count += 1
            if j != 0 :
                sum_of_squared_differences += (original_image[i][j-1]-original_image[i][j])**2
                count += 1
            if i != original_image.shape[0]-1:
                sum_of_squared_differences += (original_image[i+1][j]-original_image[i][j])**2
                count += 1
            if j != original_image.shape[1]-1:
                sum_of_squared_differences += (original_image[i][j+1]-original_image[i][j])**2
                count += 1
            variance = sum_of_squared_differences/count
            if variance == 0: 
                variance = 1e-10
            sum_of_similarity_ij = 0
            if i != 0 :
                similarity_ij = np.exp(-1*((original_image[i-1][j]-original_image[i][j])**2)/(2*variance))*np.exp(-1/3)
                average_image[i,j] += similarity_ij*original_image[i-1][j]
                sum_of_similarity_ij += similarity_ij
            if j != 0 :
                similarity_ij = np.exp(-1*((original_image[i][j-1]-original_image[i][j])**2)/(2*variance))*np.exp(-1/3)
                average_image[i,j] += similarity_ij*original_image[i][j-1]
                sum_of_similarity_ij += similarity_ij
            if i != original_image.shape[0]-1:
                similarity_ij = np.exp(-1*((original_image[i+1][j]-original_image[i][j])**2)/(2*variance))*np.exp(-1/3)
                average_image[i,j] += similarity_ij*original_image[i+1][j]
                sum_of_similarity_ij += similarity_ij
            if j != original_image.shape[1]-1:
                similarity_ij = np.exp(-1*((original_image[i][j+1]-original_image[i][j])**2)/(2*variance))*np.exp(-1/3)
                average_image[i,j] += similarity_ij*original_image[i][j+1]
                sum_of_similarity_ij += similarity_ij
            average_image[i,j] /= sum_of_similarity_ij

    pixel_values = np.float32(average_image.reshape((-1,1)))
    unique_pixel_values, pixel_indices, pixel_counts = np.unique(pixel_values,return_inverse=True,return_counts=True)
    average_pixel_values = np.float32(average_image.reshape((-1,1)))
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    return_val, labels, centers = cv2.kmeans(pixel_values, no_of_segments, None, term_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    kmeans_labels = np.copy(labels)
    kmeans_centers = np.copy(centers)
    ground_truth = np.copy(labels).flatten()

    initial_membership_matrix = np.random.rand(unique_pixel_values.shape[0],centers.shape[0])
    initial_membership_matrix = initial_membership_matrix/initial_membership_matrix.sum(axis=1)[:,None]
    membership_matrix = initial_membership_matrix

    cost_value = 0
    cost_array = []

    pixel_counts = pixel_counts.reshape((-1,1))

    for i in range(no_of_iterations):
        centers = class_means(membership_matrix,unique_pixel_values,fuzzy_num,pixel_counts)
        membership_matrix = update_membership_values(unique_pixel_values, centers, no_of_segments, fuzzy_num)
        cost_value = j_divergence(membership_matrix, unique_pixel_values.reshape((-1,1)),centers.reshape((-1,1)), fuzzy_num, pixel_counts.reshape((-1,1)))
        cost_array.append(cost_value)
        print(f"Iteration {i}: { cost_value }")

    labels = np.argmax(membership_matrix,axis = 1)
    segmented_labels = np.copy(labels[pixel_indices]).flatten()
    
    if np.all(centers >= 0) and np.all(centers <= 1):
        centers = centers * 255
    centers = np.uint8(centers)

    segmented_data = centers[labels.flatten()]
    segmented_data = segmented_data[pixel_indices]
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
    fig, ax = plt.subplots(1, 3 )
    ax[0].imshow(original_image,cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(segmented_image,cmap='gray')
    ax[1].set_title("FGFCM Algorithm")
    ax[2].imshow(kmeans_segmented_data, cmap = 'gray')
    ax[2].set_title("K-Means Clustering Algorithm")
    plt.show()
    
    return segmented_image, cost_array