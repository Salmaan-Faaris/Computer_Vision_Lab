import cv2
import numpy as np
import matplotlib.pyplot as plt

def unified_feature_pipeline(ref_path, target_path, scene_path):

    # ---------------------------------------------------
    # 1️⃣ Load images
    # ---------------------------------------------------
    img_ref = cv2.imread(ref_path, 0)       # Reference image
    img_target = cv2.imread(target_path, 0) # Image to register
    img_scene = cv2.imread(scene_path, 0)   # Scene for object detection

    orb = cv2.ORB_create(5000)

    # ---------------------------------------------------
    # 2️⃣ Feature Detection & Matching (Registration)
    # ---------------------------------------------------
    kp1, des1 = orb.detectAndCompute(img_ref, None)
    kp2, des2 = orb.detectAndCompute(img_target, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Estimate Affine Transformation (Image Registration)
    M_affine, mask = cv2.estimateAffine2D(pts2, pts1, method=cv2.RANSAC)
    h, w = img_ref.shape
    registered = cv2.warpAffine(img_target, M_affine, (w, h))

    print("Affine Matrix (Registration):\n", M_affine)

    # ---------------------------------------------------
    # 3️⃣ Manual Affine Transformation
    # ---------------------------------------------------
    manual_M = np.float32([
        [1.1, 0.2, 40],
        [0.1, 1.2, 30]
    ])

    manual_affine = cv2.warpAffine(img_ref, manual_M, (w, h))

    print("Manual Affine Matrix:\n", manual_M)

    # ---------------------------------------------------
    # 4️⃣ Object Recognition using Homography
    # ---------------------------------------------------
    kp_obj, des_obj = orb.detectAndCompute(img_ref, None)
    kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

    bf2 = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches2 = bf2.knnMatch(des_obj, des_scene, k=2)

    good = []
    for m, n in matches2:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    detected_scene = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)

    if len(good) > 10:
        pts_obj = np.float32([kp_obj[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        H, mask = cv2.findHomography(pts_obj, pts_scene, cv2.RANSAC, 5.0)

        h_obj, w_obj = img_ref.shape
        corners = np.float32([[0,0],[0,h_obj],[w_obj,h_obj],[w_obj,0]]).reshape(-1,1,2)
        projected = cv2.perspectiveTransform(corners, H)

        cv2.polylines(detected_scene, [np.int32(projected)], True, (0,255,0), 3)

        print("Homography Matrix:\n", H)
    else:
        print("Object not detected")

    # ---------------------------------------------------
    # 5️⃣ Display Results
    # ---------------------------------------------------
    plt.figure(figsize=(15,10))

    plt.subplot(2,2,1)
    plt.title("Reference Image")
    plt.imshow(img_ref, cmap='gray')

    plt.subplot(2,2,2)
    plt.title("Registered Image (Affine)")
    plt.imshow(registered, cmap='gray')

    plt.subplot(2,2,3)
    plt.title("Manual Affine Transform")
    plt.imshow(manual_affine, cmap='gray')

    plt.subplot(2,2,4)
    plt.title("Object Detection (Homography)")
    plt.imshow(cv2.cvtColor(detected_scene, cv2.COLOR_BGR2RGB))

    plt.show()


unified_feature_pipeline(
    "D:\\Salman\\Semesters\\Sem8\\Computer Vision\\Class Work\\Image Registration\\image1.jfif",   # reference
    "D:\\Salman\\Semesters\\Sem8\\Computer Vision\\Class Work\\Image Registration\\image2.jpg",   # image to register
    "D:\\Salman\\Semesters\\Sem8\\Computer Vision\\Class Work\\Image Registration\\Scene.jpg"     # scene containing object
)
