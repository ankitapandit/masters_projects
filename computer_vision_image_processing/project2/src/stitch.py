import cv2
import numpy as np
import os
import sys


def apply_ransac(source, destination):
    count = 0
    max_inliers = -1
    min = sys.maxsize
    final_matrix = None

    while(count < 5000):
        source_sample_points = source[np.random.choice(source.shape[0], 4, replace=False), :]
        destination_sample_points = list()
        for i in source_sample_points:
            result = np.where((source==i).all(axis=1))
            if len(result) > 0 and len(result[0]) > 0:
                destination_sample_points.append(destination[result[0][0]])
        destination_sample_points = np.asarray(destination_sample_points)
        homography_matrix = calculate_homography(source_sample_points,destination_sample_points)
        inliers = 0
        for i in range(len(source)):
            distance = calculate_distance(source[i], destination[i], homography_matrix)
            if distance < 4:
                inliers += 1
        if inliers > max_inliers:
            max_inliers = inliers
            final_matrix = homography_matrix
        count += 1
    return final_matrix


def calculate_distance(source, destination, matrix):
    p1 = np.transpose(np.matrix([source.item(0), source.item(1), 1]))
    p2 = (1 / (np.dot(matrix, p1)).item(2)) * (np.dot(matrix, p1))
    p3 = np.transpose(np.matrix([destination.item(0), destination.item(1), 1]))
    error = p3 - p2
    distance = np.linalg.norm(error)
    return distance


def calculate_homography(source, destination):
    RND_POINTS = 4
    dlt_matrix = list()
    for i in range(0, RND_POINTS):
        temp = list()
        p1 = np.matrix(source[i])
        p2 = np.matrix(destination[i])
        line1 = [0, 0, 0, -1.0 * p1.item(0), -1.0 * p1.item(1), -1.0,
                 p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1)]
        temp.append(line1)
        line2 = [-1.0 * p1.item(0), -1.0 * p1.item(1), -1.0, 0, 0, 0,
                 p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0)]
        temp.append(line2)
        dlt_matrix += temp
    dlt_matrix = np.asarray(dlt_matrix)
    u,s,vh = np.linalg.svd(dlt_matrix)
    result_matrix = vh[8].reshape(3,3)
    result_matrix = (1 / result_matrix.item(8)) * result_matrix
    return result_matrix


def stitch(good_match, img1, img2, kp_1, kp_2):
    MIN_MATCH_COUNT = 10
    if len(good_match) > MIN_MATCH_COUNT:
        source_pts = np.float32([kp_1[m[2]].pt for m in good_match]).reshape(-1, 1, 2)
        destination_pts = np.float32([kp_2[m[1]].pt for m in good_match]).reshape(-1, 1, 2)
        source_pts = source_pts.reshape(source_pts.shape[0], source_pts.shape[2])
        destination_pts = destination_pts.reshape(destination_pts.shape[0], destination_pts.shape[2])
        ransac_homography = apply_ransac(source_pts, destination_pts)
        w1, h1 = img1.shape[:2]
        w2, h2 = img2.shape[:2]
        pts1 = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)
        pts3 = cv2.perspectiveTransform(pts2, ransac_homography)
        pts = np.concatenate((pts3, pts1), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    else:
        print("REQUIRED: Matching keypoints should be more than " + str(MIN_MATCH_COUNT))
    res = cv2.warpPerspective(img1, Ht.dot(ransac_homography), (xmax - xmin, ymax - ymin))
    res[t[1]:w2 + t[1], t[0]:h2 + t[0]] = img2
    result = res
    return result


def match_keypoints(des_1, des_2):
    matches = list()
    for i in range(0, len(des_1)):
        distance = list()
        for j in range(0, len(des_2)):
            dist = cv2.norm(des_1[i], des_2[j], cv2.NORM_HAMMING)
            temp = [dist, j, i, j]
            distance.append(temp)
        distance.sort()
        matches.append(distance[:2])
    matches_ll = list()
    for k in matches:
        temp = list()
        for l in k:
            temp.append([l[0], l[1], l[2]])
        matches_ll.append(temp)
    good_matches = []
    for k, l in matches_ll:
        if k[0] < 0.6 * l[0]:
            good_matches.append(k)
    return good_matches


def detect_keypoints(img1, img2):
    orb = cv2.ORB_create()
    kp_1 = orb.detect(img1, None)
    kp_1, d_1 = orb.compute(img1, kp_1)
    kp_2 = orb.detect(img2, None)
    kp_2, d_2 = orb.compute(img2, kp_2)
    return kp_1, d_1, kp_2, d_2


def main():
    arg = sys.argv
    input_dir = arg[1]
    images = []

    if os.path.exists(input_dir + '/panorama.jpg'):
        os.remove(input_dir + '/panorama.jpg')
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                file_path = os.path.join(input_dir, file)
                image_matrix = cv2.imread(file_path)
                images.append(image_matrix)
    result=images[0]
    for k in range(1, len(images)):
        image_1 = result
        image_2 = images[k]
        keypoint_1, descriptor_1, keypoint_2, descriptor_2 = detect_keypoints(image_1, image_2)
        good_matches = match_keypoints(descriptor_1, descriptor_2)
        result = stitch(good_matches, image_1, image_2, keypoint_1, keypoint_2)
    cv2.imwrite(input_dir + '/panorama.jpg', result)


if __name__ == '__main__':
    main()