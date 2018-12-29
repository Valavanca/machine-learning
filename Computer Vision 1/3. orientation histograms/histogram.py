import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import string
import heapq


def read_image_grayscale(filename):
    src = cv.imread(filename, 0)
    if src is None:
        print ("Error opening image. The image %s does not exist." % filename)
        return -1
    else:
        print ('Image %s loaded' % filename)
        return src


def show_image_greyscale(image, caption="image"):
    plt.imshow(image, cmap='gray')
    plt.title(caption)
    plt.show()

def show_image(image, caption="image"):
    cv.imshow(caption, image)
    k = cv.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv.imwrite(format_filename(caption)+".jpg", image)
        cv.destroyAllWindows()

def format_filename(s):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')
    return filename


def rotate_image(image, angle):
    rows, cols = image.shape
    rotation_matrix = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv.warpAffine(image, rotation_matrix, (cols, rows))


def magnitude_and_angle_of(image):
    """ Gradient of image
    cv2.CV_64F values are represented by positive values. Positive slope
    np.uint8. White/Black. Negative slope
    Args:
        image (Mat::data): In the case of color images, the decoded images will have the channels stored in B G R order.
        filter (ndarray):  The gradient can be calculated using a simple linear (-1, 0, 1) and (-1, 0, 1)T filter.
    """
    
    # operator_x = np.array([[0, 0, 0],
    #                    [-1, 0, 1],
    #                    [0, 0, 0]])
    # operator_y = np.array([[0, -1, 0],
    #                    [0, 0, 0],
    #                    [0, 1, 0]])
    # calculate the gradient images
    image = cv.GaussianBlur(image, (3, 3), 0)

    gx_64f = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1) # or cv.filter2D(image, cv.CV_32F, operator_x)
    gy_64f = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1) # or cv.filter2D(image, cv.CV_32F, operator_y)
    
    abs_gx_64f = np.absolute(gx_64f)
    abs_gy_64f = np.absolute(gy_64f)

    gx = np.float32(abs_gx_64f)
    gy = np.float32(abs_gy_64f)

    # magnitude and direction of gradient. Arctan(gy/gx)
    mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)

    show_image_greyscale(gx, caption="gx")
    show_image_greyscale(gy, caption="gy")

    return mag, angle


def histogram_from(data, x_axis="Angle", y_axis="Count(%)", title='histogram', color='r'):
    # norm = cv.normalize(data, None)
    row = data.flatten()
    values, bins, _ = plt.hist(row, normed=True, bins=int(row.max())+1, alpha=0.5, color=color)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.show()
    return values, bins


def get_largest_values(targetArray, mirrorArray, k):
    """ find the largest k values from target array and bind them 
    with values from mirror array on the same index
    
    Args:
        targetArray (list): List for evaluating
        mirrorArray (list): List for meta values
        k (int): k-largest values from targetArray
    """
    heap = []
    for idx, val in enumerate(targetArray):
        # the smallest item on the heap is heap[0]
        if len(heap) < k or val > heap[0][0]:
            # If the heap is full, remove the smallest element on the heap.
            if len(heap) == k:
                heapq.heappop(heap)
            # add the current element as the new smallest.
            heapq.heappush(heap, [val, mirrorArray[idx]])

    k_mirror_values = [heapq.heappop(heap)[1] for i in range(k)]

    return k_mirror_values[::-1]


def most_prevalent_angle(values, bins):
    most_prevalents_angle = get_largest_values(values, bins, 3)
    for angle in most_prevalents_angle:
        if angle == 0:
            continue
        else:
            return angle, (angle+180)%360
    

if __name__ == "__main__":
    # path to the selected image
    IMG_PATH = "3. orientation histograms\lines2.png"
    # IMG_PATH = "data\dog.jpg"
    # IMG_PATH = "data/blackandwhite.jpg"

    img = read_image_grayscale(IMG_PATH)
    show_image(img, caption="original")
    _, angle = magnitude_and_angle_of(img)
    values, bins = histogram_from(angle)
    angle, _ = most_prevalent_angle(values, bins)
    print("Angle: %s" % angle)

    rotated_img = rotate_image(img, angle)
    show_image(rotated_img, caption="rotate to %s" % angle)
