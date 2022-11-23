import cv2 as cv
import numpy as np

def compute_weights(images, time_decay):
    (w_c, w_s, w_e) = (1, 1, 1) #contrast, saturation, exposedness

    if time_decay is not None:
        print("\rset time", end="")
        times = np.array(range(len(images)-1, -1, -1))
        decay = np.exp(-((times)**2)/(2*((len(images)**2)/(np.float32(time_decay)**2))))

    print("now set image type", end="")
    arr = []
    weights = []
    if images[0].shape[1] is not None:
        for j in range(2):
            arr.append(images[0].shape[j])
    elif images[0].shape[0] is not None:
        arr.append(images[0].shape[0])
    weights_sum = np.zeros(arr, dtype=np.float32)

    i = 0
    j = 0
    for image_uint in images:
        image = np.float32(image_uint)/(2**8 - 1)
        W = np.ones(image.shape[:2], dtype=np.float32)

        #contrast, saturation, exposedness calc
        W_contrast = np.absolute(cv.Laplacian(cv.cvtColor(image, cv.COLOR_RGB2GRAY), cv.CV_32F)) ** w_c + 1
        W_saturation = image.std(axis=2, dtype=np.float32) ** w_s + 1
        W_exposedness = np.prod(np.exp(-((image - 0.5)**2)/0.8), axis=2, dtype=np.float32) ** w_e + 1
        W = np.multiply(np.multiply(np.multiply(W, W_contrast), W_saturation), W_exposedness)

        if time_decay is not None:
            W *= decay[i]
            i += 1

        print(f"\rnew time decay : {i + 1} / calc repeat : {j + 1}", end="")
        weights_sum += W
        weights.append(W)
        j += 1

    #normalization
    nonzero = weights_sum > 0
    for i in range(len(weights)):
        weights[i][nonzero] /= weights_sum[nonzero]
        weights[i] = np.uint8(weights[i]*255)

        if(i < len(images)):
            mylist = weights[i]
            mylist *= len(weights)
            cv.imwrite(f"output/weightmap/weightmap{i}.jpg", mylist)

    print(f"\rtime decay sync repeat : {i + 1}\nCSE calc repeat : {j + 1}")
    return weights


def gaussian_kernel(size=5, sigma=0.4):
    print("\rnow calc gaussian kernal", end="")
    return cv.getGaussianKernel(ksize=size, sigma=sigma)


def image_reduce(image):
    print("\rnow reduce image", end="")
    kernel = gaussian_kernel()
    out_image = cv.filter2D(image, cv.CV_8UC3, kernel)
    out_image = cv.resize(out_image, None, fx=0.5, fy=0.5)
    return out_image


def image_expand(image):
    print("\rnow expand image", end="")
    kernel = gaussian_kernel()
    out_image = cv.resize(image, None, fx=2, fy=2)
    out_image = cv.filter2D(out_image, cv.CV_8UC3, kernel)
    return out_image


def gaussian_pyramid(img, depth):
    print("\rnow break down to gaussian pyramid", end="")
    G = img.copy()
    gp = [G]
    for i in range(depth):
        G = image_reduce(G)
        gp.append(G)
    return gp


def laplacian_pyramid(img, depth):
    print("\rnow break down to laplacian pyramid", end="")
    gp = gaussian_pyramid(img, depth+1)
    lp = [gp[depth-1]]
    for i in range(depth-1, 0, -1):
        GE = image_expand(gp[i])
        L = cv.subtract(gp[i-1], GE)
        lp = [L] + lp
    return lp


def pyramid_collapse(pyramid):
    print("\rnow collapsing pyramid", end="")
    depth = len(pyramid)
    collapsed = pyramid[depth-1]
    for i in range(depth-2, -1, -1):
        collapsed = cv.add(image_expand(collapsed), pyramid[i])
    return collapsed


def exposure_fusion(images, depth=3, time_decay=None):

    #err check
    if not isinstance(images, list) or len(images) < 2:
        print("Input has to be a list of at least two images")
        return None

    size = images[0].shape
    for i in range(len(images)):
        if not images[i].shape == size:
            print("Input images have to be of the same size")
            return None

    #compute weights, G/L pyramid
    weights = compute_weights(images, time_decay)
    gaussian_pyramids = []
    laplacian_pyramids = []
    for (image, weight) in zip(images, weights):
        gaussian_pyramids.append(gaussian_pyramid(weight, depth))
        laplacian_pyramids.append(laplacian_pyramid(image, depth))

    #combine pyramids * weight
    combines = []
    for l in range(depth):
        combine = np.zeros(laplacian_pyramids[0][l].shape, dtype=np.uint8)
        for k in range(len(images)):
            my = np.float32(gaussian_pyramids[k][l])/255
            gaussian = np.dstack((my, my, my))
            laplacian = laplacian_pyramids[k][l]
            combine = cv.add(combine, cv.multiply(laplacian, gaussian, dtype=cv.CV_8UC3))
        combines.append(combine)
    #익명의 인도인 개발자

    #collapse pyramid
    fusion = pyramid_collapse(combines)

    return fusion


def align_images(images, n):
    #err check
    print("\rerror checking", end="")
    if not isinstance(images, list) or len(images) < 2:
        print("\n<error> why 012 images?")
        return None

    size = images[0].shape
    for i in range(len(images)):
        if not images[i].shape == size:
            print("\n<error> why diff images?")
            return None

    #grayscale make
    grayscale = []
    for i in range(n):
        print(f"\rnow make grayscale {i+1}/{n}", end="")
        grayscale.append(cv.cvtColor(images[i], cv.COLOR_BGR2GRAY))
    model_image = grayscale[0]

    #find size, define model
    print("\rnow find image size", end="") #my modify
    size_img = model_image.shape
    print("\rnow define image's motion model", end="") #my modify
    warp_mode = cv.MOTION_TRANSLATION

    #define 2x3 or 3x3 matrices and initialize the matrix to identity
    print("\rnow define matrix", end="") #my modify
    if warp_mode == cv.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    #define termination criteria
    print("\rnow define termination criteria", end="") # my modify
    roof = 5000 #반복 횟수
    accu = 1e-10 #목표 정확도
    #반복 횟수 초과로 돌거나, 목표 정확도 이상의 정확도이면 반복 종료
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, roof, accu)
    #find size ~ define termination criteria 제작자: 깃허브에 거주하는 익명의 인도인

    #ECC run
    aligned_images = [images[0]]
    for i in range(1, len(images)):
        print(f"\rnow run ECC {i+1}/{len(images)}", end="")
        (cc, warp_matrix) = cv.findTransformECC(model_image, grayscale[i], warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=3)

        if warp_mode == cv.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography
            aligned_image = cv.warpPerspective(images[i], warp_matrix, (size_img[1], size_img[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            aligned_image = cv.warpAffine(images[i], warp_matrix, (size_img[1], size_img[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
        #if, else문 내용 제작자: 깃허브에 거주하는 익명의 인도인

        aligned_images.append(aligned_image)
        cv.imwrite(f"output/aligned/img{i-1}.jpg", aligned_image)

    print(f"\rimage resolution : {size_img[0]} x {size_img[1]}")

    return aligned_images