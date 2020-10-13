import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def Otsu(array):
    min_value = array.min().item()
    max_value = array.max().item()
    hist = cv2.calcHist([array],[0],None,[256],[min_value,max_value]).squeeze() / np.prod(array.shape)
    bins = np.linspace(min_value, max_value, 256)
    order1 = hist * bins

    omega_0 = np.cumsum(hist)
    omega_1 = 1 - omega_0
    cum_order = np.cumsum(order1)
    cum_order_r = cum_order[-1] - cum_order

    mu_0 = np.zeros_like(omega_0)
    # avoid 0 omega_0 in the denominator, set the average of the first several 0 valued bins as 0
    start_id = 0
    for i in range(256):
        if omega_0[i] == 0:
            start_id+=1
        else:
            break
    mu_0[start_id:] = (1 / omega_0[start_id:]) * cum_order[start_id:] 
    mu_1 = (1 / omega_1) * cum_order_r

    sigmaB = omega_0 * omega_1 * (mu_0 - mu_1) * (mu_0 - mu_1)
    idx = np.argmax(sigmaB) 
    K = bins[idx]
    return K


def iterOtsu(img, iter, l=False, reverse=False):
    array = img.ravel()
    for i in range(iter):
        k = Otsu(array)
        idx = np.where(array <= k) if l else np.where(array >= k)
        print(k, idx[0].shape)
        array = array[idx]

    mask = np.zeros_like(img, dtype=np.float32)
    if reverse:
        mask[np.where(img <= k)] = 1
    else:
        mask[np.where(img >= k)] = 1
    return mask, k


def BGRseg(img, iters, l=False, reverse=False):
    masks = []
    t = []
    for i in range(3):
        gray = img[:,:,i].squeeze()
        mask, threshold = iterOtsu(gray, iters[i], l, reverse)
        masks.append(mask)
        t.append(int(threshold))
    combined_mask = cv2.bitwise_and(masks[0], masks[1], masks[2])
    return combined_mask, t


def varmap(img, window_size):
    N = window_size
    b = int((N - 1) / 2)
    expand_im = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_REPLICATE)

    h, w = img.shape
    grids = np.zeros((h, w, N*N))
    for i in range(b, h+b):
        for j in range(b, w+b):
            grids[i-b, j-b] = expand_im[i-b:i+b+1, j-b:j+b+1].ravel()
    var_img = np.var(grids, axis=2, keepdims=False).astype(np.float32)
    return var_img


def contextseg(img, windows, iters, l=True, reverse=False):
    masks = []
    for i in range(len(iters)):
        var_img = varmap(img, windows[i])
        mask, threshold = iterOtsu(var_img, iter=iters[i], l=l, reverse=reverse)
        masks.append(mask)
    combined_mask = cv2.bitwise_and(masks[0], masks[1], masks[2])
    return combined_mask


def findContour(mask):
    expanded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    neighbor_mask = np.zeros_like(expanded_mask)
    # cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
    tl = cv2.copyMakeBorder(mask, 0, 2, 0, 2, cv2.BORDER_REPLICATE)
    t = cv2.copyMakeBorder(mask, 0, 2, 1, 1, cv2.BORDER_REPLICATE)
    tr = cv2.copyMakeBorder(mask, 0, 2, 2, 0, cv2.BORDER_REPLICATE)
    l = cv2.copyMakeBorder(mask, 1, 1, 0, 2, cv2.BORDER_REPLICATE)
    r = cv2.copyMakeBorder(mask, 1, 1, 2, 0, cv2.BORDER_REPLICATE)
    bl = cv2.copyMakeBorder(mask, 2, 0, 0, 2, cv2.BORDER_REPLICATE)
    b = cv2.copyMakeBorder(mask, 2, 0, 1, 1, cv2.BORDER_REPLICATE)
    br = cv2.copyMakeBorder(mask, 2, 0, 2, 0, cv2.BORDER_REPLICATE)
    accu = tl + t + tr + l + r + bl + b + br
    neighbor_mask[np.where(accu < 8)] = 1
    contour = cv2.bitwise_and(neighbor_mask, expanded_mask)
    return contour[1:-1, 1:-1]


def largestComponent(mask, connectivity=8):
    new_img = np.zeros_like(mask, dtype=np.float32)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity)
    max_label, _ = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
    new_img[np.where(output==max_label)] = 1
    return new_img


folder_task = '/home/xingguang/Documents/ECE661/hw6/images'
imcat = cv2.imread(os.path.join(folder_task, 'cat.jpg'))
impigeon = cv2.imread(os.path.join(folder_task, 'pigeon.jpeg'))
imfox = cv2.imread(os.path.join(folder_task, 'fox.jpg'))

graycat = cv2.cvtColor(imcat, cv2.COLOR_BGR2GRAY)
graypigeon = cv2.cvtColor(impigeon, cv2.COLOR_BGR2GRAY)
grayfox = cv2.cvtColor(imfox, cv2.COLOR_BGR2GRAY)

# ---------------------------cat image-------------------------------
mask, t = BGRseg(imcat, iters=[2, 1, 3], l=True, reverse=False)
print("BGR thresholds for cat image:", t)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(mask,kernel,iterations = 7)
final_mask = cv2.erode(dilation,kernel,iterations = 6)

contour = findContour(final_mask)
cv2.imwrite("BGR_cat.jpeg", final_mask*255)
cv2.imwrite("BGR_cat_contour.jpeg", contour*255)

mask = contextseg(graycat, windows=[7, 11, 15], iters=[5, 4, 4], l=True)
cv2.imwrite("contextmask_cat.jpeg", mask*255)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(mask,kernel,iterations = 2)
final_mask = cv2.erode(dilation,kernel,iterations = 3)
contour = findContour(final_mask)
cv2.imwrite("context_cat.jpeg", final_mask*255)
cv2.imwrite("context_cat_contour.jpeg", contour*255)

# ------------------------pigeon image-------------------------------
mask, t = BGRseg(impigeon, iters=[1, 1, 1], l=False, reverse=False)
print("BGR thresholds for ipgeon image:", t)

kernel = np.ones((5,5),np.uint8)
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
dilation = cv2.dilate(opened,kernel,iterations = 1)
final_mask = cv2.erode(dilation,kernel,iterations = 1)
largest = largestComponent(final_mask)
contour = findContour(largest)
cv2.imwrite("BGR_pigeon.jpeg", final_mask*255)
cv2.imwrite("BGR_pigeon_comp.jpeg", largest*255)
cv2.imwrite("BGR_pigeon_contour.jpeg", contour*255)

mask = contextseg(graypigeon, windows=[3, 5, 7], iters=[4, 3, 3], l=True, reverse=True)
cv2.imwrite("contextmask_pigeon.jpeg", mask*255)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(mask,kernel,iterations = 1)
final_mask = cv2.dilate(erosion,kernel,iterations = 1)
contour = findContour(final_mask)
cv2.imwrite("context_pigeon.jpeg", final_mask*255)
cv2.imwrite("context_pigeon_contour.jpeg", contour*255)

# -------------------------- fox image-------------------------------
mask, t = BGRseg(imfox, iters=[2, 1, 2], l=False, reverse=False)
print("BGR thresholds for fox image:", t)

kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(mask,kernel,iterations = 2)
final_mask = cv2.erode(dilation,kernel,iterations = 1)
largest = largestComponent(final_mask)
contour = findContour(largest)
cv2.imwrite("BGR_fox.jpeg", final_mask*255)
cv2.imwrite("BGR_fox_comp.jpeg", largest*255)
cv2.imwrite("BGR_fox_contour.jpeg", contour*255)

mask = contextseg(grayfox, windows=[3, 5, 7], iters=[4, 3, 3], l=True, reverse=False)
cv2.imwrite("contextmask_fox.jpeg", mask*255)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(mask,kernel,iterations = 1)
final_mask = cv2.dilate(erosion,kernel,iterations = 1)
largest = largestComponent(final_mask)
largest = cv2.dilate(largest,kernel,iterations = 2)
largest = cv2.erode(largest,kernel,iterations = 2)
contour = findContour(largest)
cv2.imwrite("context_fox.jpeg", final_mask*255)
cv2.imwrite("context_fox_comp.jpeg", largest*255)
cv2.imwrite("context_fox_contour.jpeg", contour*255)