import cv2
import numpy as np
import matplotlib.pyplot as plt

def block_matching_sad(img1, img2, block_size=16, search_range=7):
    h, w = img1.shape
    motion_vectors = []

    # Output image for visualization
    output_img = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):

            best_sad = float('inf')
            best_dx = 0
            best_dy = 0

            # Extract current block
            block = img1[y:y+block_size, x:x+block_size]

            # Search in neighborhood
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):

                    ny = y + dy
                    nx = x + dx

                    if (0 <= ny < h - block_size) and (0 <= nx < w - block_size):
                        candidate = img2[ny:ny+block_size, nx:nx+block_size]

                        sad = np.sum(np.abs(block - candidate))

                        if sad < best_sad:
                            best_sad = sad
                            best_dx = dx
                            best_dy = dy

            motion_vectors.append((x, y, best_dx, best_dy))

            # Draw motion vector
            cv2.arrowedLine(output_img,
                            (x + block_size//2, y + block_size//2),
                            (x + block_size//2 + best_dx,
                             y + block_size//2 + best_dy),
                            (0, 0, 255), 1)

    return output_img, motion_vectors



img1 = cv2.imread("D:\\Salman\\Semesters\\Sem8\\Computer Vision\\Class Work\\frame1.jfif", 0)
img2 = cv2.imread("D:\\Salman\\Semesters\\Sem8\\Computer Vision\\Class Work\\frame2.jpg", 0)

result, motion_vectors = block_matching_sad(img1, img2)

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Motion Vectors using SAD")
plt.axis("off")
plt.show()