import numpy as np
import cv2
import pickle
import math
import sys
import os
import json
from nms import nms

final_weights = list()
final_classifiers = list()
features = list()
json_list = list()
coord = list()
scores = list()


def compute_polarity(classifier, int_img):
    feature = classifier[0]
    threshold = classifier[1]
    polarity = classifier[2]
    S1 = 0
    S2 = 0
    for white, black in feature:
        if len(white) == 4:
            i, j, w, h = white
            D = int_img[i][j]
            if (i - h >= 0):
                B = int_img[i - h][j]
            else:
                B = 0
            if j - w >= 0:
                C = int_img[i][j - w]
            else:
                C = 0
            if i - h >= 0 and j - w >= 0:
                A = int_img[i - h][j - w]
            else:
                A = 0
            S1 = (D + A) - (B + C)
        else:
            for p in white:
                i, j, w, h = p
                D = int_img[i][j]
                if (i - h >= 0):
                    B = int_img[i - h][j]
                else:
                    B = 0
                if j - w >= 0:
                    C = int_img[i][j - w]
                else:
                    C = 0
                if i - h >= 0 and j - w >= 0:
                    A = int_img[i - h][j - w]
                else:
                    A=0
                S1 += (D + A) - (B + C)
        if len(black) == 4:
            i, j, w, h = black
            D = int_img[i][j]
            if (i - h >= 0):
                B = int_img[i - h][j]
            else:
                B = 0
            if j - w >= 0:
                C = int_img[i][j - w]
            else:
                C = 0
            if i - h >= 0 and j - w >= 0:
                A = int_img[i - h][j - w]
            else:
                A = 0
            S2 = (D + A) - (B + C)
        else:
            for n in black:
                i, j, w, h = n
                D = int_img[i][j]
                if (i - h >= 0):
                    B = int_img[i - h][j]
                else:
                    B = 0
                if j - w >= 0:
                    C = int_img[i][j - w]
                else:
                    C = 0
                if i - h >= 0 and j - w >= 0:
                    A = int_img[i - h][j - w]
                else:
                    A=0
                S2 += (D + A) - (B + C)
    Score = S2-S1
    return 1 if polarity * Score < polarity *threshold else 0


def get_feature_scores(features, int_img):
    Scores = list()
    for white, black in features:
        S1 = 0
        S2 = 0
        if len(white) == 4:
            i, j, w, h = white
            D = int_img[i][j]
            if (i - h >= 0):
                B = int_img[i - h][j]
            else:
                B = 0
            if j - w >= 0:
                C = int_img[i][j - w]
            else:
                C = 0
            if i - h >= 0 and j - w >= 0:
                A = int_img[i - h][j - w]
            else:
                A = 0
            S1 = (D + A) - (B + C)
        else:
            for p in white:
                i, j, w, h = p
                D = int_img[i][j]
                if (i - h >= 0):
                    B = int_img[i - h][j]
                else:
                    B = 0
                if j - w >= 0:
                    C = int_img[i][j - w]
                else:
                    C = 0
                if i - h >= 0 and j - w >= 0:
                    A = int_img[i - h][j - w]
                else:
                    A=0
                S1 += (D + A) - (B + C)
        if len(black) == 4:
            i, j, w, h = black
            D = int_img[i][j]
            if (i - h >= 0):
                B = int_img[i - h][j]
            else:
                B = 0
            if j - w >= 0:
                C = int_img[i][j - w]
            else:
                C = 0
            if i - h >= 0 and j - w >= 0:
                A = int_img[i - h][j - w]
            else:
                A = 0
            S2 = (D + A) - (B + C)
        else:
            for n in black:
                i, j, w, h = n
                D = int_img[i][j]
                if (i - h >= 0):
                    B = int_img[i - h][j]
                else:
                    B = 0
                if j - w >= 0:
                    C = int_img[i][j - w]
                else:
                    C = 0
                if i - h >= 0 and j - w >= 0:
                    A = int_img[i - h][j - w]
                else:
                    A = 0
                S2 += (D + A) - (B + C)
        Score = S2-S1
        Scores.append(Score)
    return Scores


def compute_features(image):
    height, width = image.shape
    features = []
    count = 0
    for i in range(0, height):
        for j in range(0, width):
            for w in range(1, width+1):
                for h in range(1, height+1):
                    # W
                    # B
                    if ((i + h - 1) < height) and ((j + 2 * w - 1) < width):
                       white = [i + h - 1, j + w - 1, w, h]
                       black = [i + h - 1, j + 2 * w - 1, w, h]
                       features.append((white, black))
                    if ((i + 2 * h - 1) < height) and ((j + w - 1) < width):
                        white = [i + h - 1, j + w - 1, w, h]
                        black = [i + 2 * h - 1, j + w - 1, w, h]
                        features.append((white, black))
                    #W-B-W
                    if ((i + h - 1) < height) and ((j + 3 * w - 1) < width):
                        white1 = [i + h - 1, j + w - 1, w, h]
                        black = [i + h - 1, j + 2 * w - 1, w, h]
                        white2 = [i + h - 1, j + 3 * w - 1, w, h]
                        features.append(([white1, white2], black))
                    #W
                    #B
                    #W
                    if ((i + 3 * h - 1) < height) and ((j + w - 1) < width):
                        white1 = [i + h - 1, j + w - 1, w, h]
                        black = [i + 2 * h - 1, j + w - 1, w, h]
                        white2 = [i + 3 * h - 1, j + w - 1, w, h]
                        features.append(((white1, white2), black))
                    #W-B
                    #B-W
                    if ((i + 2 * h - 1) < height) and ((j + 2 * w - 1) < width):
                        white1 = [i + h - 1, j + w - 1, w, h]
                        black1 = [i + h - 1, j + 2 * w - 1, w, h]
                        white2 = [i + 2 * h - 1, j + 2 * w - 1, w, h]
                        black2 = [i + 2 * h - 1, j + w - 1, w, h]
                        features.append(((white1, white2),(black1, black2)))
    return features


def apply_haar_feature(features, training_data):
    x = np.zeros((len(features), len(training_data)))
    y = [img[1] for img in training_data]
    i = 0
    for j in range(len(training_data)):
        score = get_feature_scores(features, training_data[j][0])
        for i in range(len(score)):
            x[i][j] = score[i]
    return x, y


def compute_integral_image(img):
    result = np.zeros(img.shape)
    sum = np.zeros(img.shape)
    for i in range(0, len(result)):
        for j in range(0, len(result[0])):
            sum[i][j] = (sum[i - 1][j] + img[i][j] if i-1>=0 else img[i][j])
            result[i][j] = (result[i][j - 1] + sum[i][j] if j-1>=0 else sum[i][j])
    return result


def get_best_weak_classifier(x, y, features, weights):
    total_pos = 0
    total_neg = 0
    for w, label in zip(weights, y):
        if label == 1:
            total_pos += w
        else:
            total_neg += w
    classifier = list()
    t_f = x.shape[0]
    for index, feature in enumerate(x):
        applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
        pos_seen, neg_seen = 0, 0
        pos_weights, neg_weights = 0, 0
        min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
        for w, f, label in applied_feature:
            error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
            if error < min_error:
                min_error = error
                best_feature = features[index]
                best_threshold = f
                best_polarity = 1 if pos_seen > neg_seen else -1
            if label == 1:
                pos_seen += 1
                pos_weights += w
            else:
                neg_seen += 1
                neg_weights += w
        classifier.append(([best_feature], best_threshold, best_polarity))
    return classifier


def find_best_classifier(weakClassifier, weights, training_data):
    best_clf, best_error, best_accuracy = None, float('inf'), None
    for clf in weakClassifier:
        error, accuracy = 0, []
        for data, w in zip(training_data, weights):
            correctness = abs(compute_polarity(clf, data[0]) - data[1])
            accuracy.append(correctness)
            error += w * correctness
        error = error / len(training_data)
        if error < best_error:
            best_clf, best_error, best_accuracy = clf, error, accuracy
    return best_clf, best_error, best_accuracy


def training_images(training, count, no_iter=20):
    weights = np.zeros(len(training))
    training_data = []
    global features
    features = compute_features(training[0][0])
    for imgCount in range(len(training)):
        training_data.append((compute_integral_image(training[imgCount][0]), training[imgCount][1]))
        if training[imgCount][1] == 1:
            weights[imgCount] = 1.0/(2*count[0])
        else:
            weights[imgCount] = 1.0/(2*count[1])
    x, y = apply_haar_feature(features, training_data)
    weights = weights / np.linalg.norm(weights)
    for i in range(no_iter):
        wclf = get_best_weak_classifier(x, y, features, weights)
        classifier, error, acc = find_best_classifier(wclf, weights, training_data)
        B = error / (1.0-error)
        for i in range(0,len(acc)):
            weights[i] *= (B**(1-acc[i]))
        A = math.log(1/B)
        final_weights.append(A)
        final_classifiers.append(classifier)
        weights = weights / np.linalg.norm(weights)


def detect_face(test, o_classifiers, o_alphas):
    total = 0
    ii = compute_integral_image(test)
    for alpha, clf in zip(o_alphas, o_classifiers):
        total += alpha * compute_polarity(clf, ii)
    alpha_sum = sum(o_alphas)
    if total >= 0.55 * alpha_sum:
        result = 1
    else:
        result = 0
    return result, total


def testing_images(img, file, output_clfs, output_alphas):
    classification = list()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img1.shape
    box_height = 100
    box_width = 100
    while(box_height < height / 2 and box_width < width / 2):
        for x in range(0, height, 20):
            for y in range(0, width, 20):
                block = img1[x:x+box_height, y:y+box_width]
                window = cv2.resize(block, (19, 19))
                flag, total = detect_face(window, output_clfs, output_alphas)
                if flag == 1:
                    classification.append([(y, x), (y + block.shape[1], x + block.shape[0]), total])
        box_height += 20
        box_width += 20
    coord = list()
    scores = list()
    for res in classification:
        coord.append((res[0][0], res[0][1], res[1][0], res[1][1]))
        scores.append(res[2])
    output = nms.boxes(coord, scores)
    output = output[:2]
    for x in output:
        if classification[x][2] > 205:
            json_dict = dict()
            cv2.rectangle(img, classification[x][0], classification[x][1], (255, 255, 0), 2)
            print(file)
            json_dict['iname'] = str(file)
            width = classification[x][1][0] - classification[x][0][0]
            height = classification[x][1][1] - classification[x][0][1]
            json_dict['bbox'] = [classification[x][0][0], classification[x][0][1], width, height]
            # print(json_dict)
            json_list.append(json_dict)
    cv2.imwrite("output/" + file, img)


def main():
    args = sys.argv
    dir = args[1]
    pickleDict = pickle.load(open('classifiers.p', 'rb'))
    classifiers = pickleDict['clfs']
    weights = pickleDict['alphas']
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(dir, file))
                testing_images(img, file, classifiers, weights)
    with open('results.json', 'w') as fout:
        json.dump(json_list, fout)


if __name__=='__main__':
    main()