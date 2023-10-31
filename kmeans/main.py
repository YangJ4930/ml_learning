import random

import numpy as np
import cv2

k = 4
max_time = 40


def deal_img(path):
    img_array = cv2.imread(path, cv2.IMREAD_COLOR)
    img_array = img_array / 255.0
    return img_array


# here need to calculate to vector's distance
def calculate_distance(vector1, vector2):
    dist = np.linalg.norm(vector1 - vector2)
    return dist


def calculate_k_center(data_set, center):
    # size = len(data_set)
    da = np.array(data_set)
    return np.mean(da, axis=0)


# random 质心
def generate_k_center(data_set, k):
    he = data_set.shape[0]
    dim = data_set.shape[1]
    centers = np.zeros((k, 3))
    for _ in range(k):
        index1 = np.random.randint(0, he - 1)
        index2 = np.random.randint(0, dim - 1)

        centers[i] = data_set[index1][index2]
    return centers


if __name__ == '__main__':
    img_arr = deal_img("./dog.jpg")
    cen = generate_k_center(img_arr, k)
    height = img_arr.shape[0]
    length = img_arr.shape[1]
    loop_over = False
    cluster = [[] for _ in range(k)]
    time = 0
    while (not loop_over) and (time < max_time):
        time += 1
        loop_over = True
        for i in range(height):
            for j in range(length):
                min_p = 10.0
                min_index = -1
                for m in range(k):
                    if min_p > calculate_distance(cen[m], img_arr[i][j]):
                        min_p = calculate_distance(cen[m], img_arr[i][j])
                        min_index = m

                if time == max_time:
                    img_arr[i][j] = cen[min_index]
                else:
                    cluster[min_index].append(img_arr[i][j])
        if time != max_time:
            for i in range(k):
                cen[i] = calculate_k_center(cluster[i], cen[i])

            for i in range(k):
                size = len(cluster[i])
                min_p = 10.0
                min_index = -1
                for j in range(size):
                    if min_p > calculate_distance(cen[i], cluster[i][j]):
                        min_p = calculate_distance(cen[i], cluster[i][j])
                        min_index = j

                if min_index != i:
                    loop_over = False
            if time != max_time:
                cluster.clear()
                cluster = [[] for _ in range(k)]
                print(cluster)

            print("---------------------")
            print(time)
            for i in range(k):
                print(cen[i])

    cv2.imwrite('dogk5.jpg', img_arr * 255)
