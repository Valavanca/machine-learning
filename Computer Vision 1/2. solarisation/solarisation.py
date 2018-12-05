import cv2
import numpy as np
import matplotlib.pyplot as plt
import string


def read_image_grayscale(filename):
    src = cv2.imread(filename, 0)
    if src is None:
        print ("Error opening image. The image %s does not exist." % filename)
        return -1
    else:
        print ('Image %s loaded' % filename)
        return src


def show_image(image, caption="image"):
    cv2.imshow(caption, image)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite(format_filename(caption)+".jpg", image)
        cv2.destroyAllWindows()


def format_filename(s):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')
    return filename


def calculate_polynomial(x1, y1, x2, y2):
    # p(x) = ax ^ 3 + bx ^ 2 + cx ^ 1 + d
    a = np.array([[x1*x1*x1, x1*x1, x1, 1],
                  [x2*x2*x2, x2*x2, x2, 1],
                  [3*x1*x1, 2*x1, 1, 0],
                  [3*x2*x2, 2*x2, 1, 0]])
    b = np.array([y1, y2, 0, 0])
    e = np.linalg.solve(a, b)
    return e


def adjust_image(polynomial, img):
    assert len(polynomial) == 4

    # image size
    h = img.shape[0]
    w = img.shape[1]

    for y in range(0, h):
        for x in range(0, w):
            c = img[y, x]
            value = polynomial[0] * pow(c, 3) + polynomial[1] * \
                pow(c, 2) + polynomial[2] * c + polynomial[3]
            img[y, x] = value

    return img


def plot_polynomial_curve(polynomial, low=0, high=255):
    values = [polynomial[0] * pow(c, 3) + polynomial[1] * pow(c, 2) + polynomial[2] * c + polynomial[3]
              for c in range(low, high)]
    plt.plot(values)
    plt.show()


if __name__ == "__main__":
    IMG_PATH = "data/dog.jpg"  # path to the selected image
    img = read_image_grayscale(IMG_PATH)
    show_image(img, caption=IMG_PATH)

    polynomial = calculate_polynomial(10, 20, 180, 255)
    solarized_img = adjust_image(polynomial, img)
    plot_polynomial_curve(polynomial)
    show_image(solarized_img, caption="10, 20, 180, 255")


    polynomial = calculate_polynomial(0, 255, 255, 0)
    solarized_img = adjust_image(polynomial, img)
    plot_polynomial_curve(polynomial)
    show_image(solarized_img, caption="0, 255, 255, 0")
