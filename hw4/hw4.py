import numpy as np
import cv2
import os
import scipy.signal as scisig


def HarrisCornerDet(img, sigma, local=1, k=0.05, filter_ratio=0.1):
    '''
    Description: Find the corners by Harris corner detector
    Input:
        img: input image
        sigma: variance of the Gaussian filter
        local: local neighborhood size for Non-maximum suppression
        k: the ratio of det(C) and Tr(C)^2
        filter_ratio: the threshold for Harris Response is set to filter_ratio * max(Response)
    Output:
        markerImg: Harris Response Image
        pList: detected corner list
    '''
    def HaarFilter(sigma=1):
        # Output the Haar filter along both x and y directions
        s = round(np.ceil(4 * sigma))
        if s % 2: s += 1
        Fx = np.ones((s, s))
        Fx[:, :int(s/2)] = -1
        Fy = -1 * np.copy(Fx.T)
        return Fx, Fy

    h, w = img.shape
    Fx, Fy = HaarFilter(sigma)
    Dx = scisig.correlate2d(img, Fx, mode="same")  # compute the derivative along x axis
    Dy = scisig.correlate2d(img, Fy, mode="same")  # compute the derivative along y axis
    Dx2 = Dx * Dx
    Dxy = Dx * Dy
    Dy2 = Dy * Dy
    sumSize = int(round(np.ceil(5*sigma)/2.0)*2+1)
    sumFilter = np.ones((sumSize, sumSize)) 
    # sum the derivative squares in neighborhoods together
    Dx2 = scisig.correlate2d(Dx2, sumFilter, mode='same') 
    Dxy = scisig.correlate2d(Dxy, sumFilter, mode='same')
    Dy2 = scisig.correlate2d(Dy2, sumFilter, mode='same')
    # find the determination and trace of C matrix
    detC = Dx2 * Dy2 - Dxy * Dxy
    trC = Dx2 + Dy2
    R = detC - k * trC * trC   # Harris response map
    thresh = filter_ratio * R.max() # threshold to get rid of the small responses
    markerImg = np.zeros_like(R)
    # Non-maximum suppression to avoid redundant corners
    for i in range(local, h-local):
        for j in range(local, w-local):
            neighbor = R[i-local:i+local+1, j-local:j+local+1]
            if R[i, j] == np.amax(neighbor) and R[i, j] >= 0:
                markerImg[i, j] = R[i, j]

    i, j = np.where(markerImg >= thresh) 
    pList = [_ for _ in zip(j, i)]
    return markerImg, pList


def DrawMarker(Img, pList, color=(0,0,255)):
    result = np.copy(Img)
    for j, i in pList:
        result = cv2.circle(result, (j, i), 5, color, 1)
    return result


def HarrisFeatureMaking(img, pList, window_size):
    '''
    Description: The feature for correspondence making. It is the pixel values of the neighborhood
        of size window_size x window_size
    '''
    ws = window_size
    feature = np.zeros((len(pList), (2*ws+1)*(2*ws+1)))
    new_img = cv2.copyMakeBorder(img, ws, ws, ws, ws, cv2.BORDER_REPLICATE)
    for k , (j, i) in enumerate(pList):
        i, j = i+ws, j+ws
        neighbors = new_img[i-ws:i+ws+1, j-ws:j+ws+1]
        feature[k, :] = neighbors.reshape((1, -1))
    return feature


def cpMatching(f1, f2, method):
    '''
    Description: find the correspondence scores between points in f1 and f2. 
    '''
    d1 = f1.shape[0]
    d2 = f2.shape[0]
    if method == 'SSD': 
        # sum of squared difference (f1-f2)^2 = f1^2 - 2*f1*f2 + f2^2
        f11 = np.sum(f1*f1, axis=1).reshape((d1, 1))
        f12 = np.matmul(f1, f2.T)
        f22 = np.sum(f2*f2, axis=1).reshape((1, d2))
        cp = -f11.repeat(d2, axis=1) + 2 * f12 - f22.repeat(d1, axis=0)
    elif method == 'NCC':
        # normalized cross-correlation
        m1 = np.sum(f1, axis=1, keepdims=True)/f1.shape[1]
        m2 = np.sum(f2, axis=1, keepdims=True)/f2.shape[1]
        nf1 = f1 - m1
        nf2 = f2 - m2
        numerator = np.matmul(nf1, nf2.T)
        denom_1 = np.sqrt(np.sum(nf1*nf1, axis=1).reshape((d1, 1)))
        denom_2 = np.sqrt(np.sum(nf2*nf2, axis=1).reshape((1, d2)))
        denominator = np.matmul(denom_1, denom_2)
        cp = numerator / denominator
    else: print("\'Method\' has to be SSD or NCC")
    f1_Matching = np.argmax(cp, axis=1)   # find the indices of features in f2 that match f1 features the most
    f2_Matching = np.argmax(cp, axis=0)   # find the indices of features in f1 that match f2 features the most
    return cp, f1_Matching, f2_Matching


def matchedImg(img1, img2, pList1, pList2, f1_f2, f2_f1):
    '''
    Description: find the correspondence pairs and mark them on the concated image
    Input: 
        f1_f2: the indices of features in f2 that match f1 features the most
        f2_f1: the indices of features in f1 that match f2 features the most
    '''
    def drawLines(img, p_start, p_end, color=(0,255,0)):
        image = np.copy(img)
        for i in range(len(p_start)):
            image = cv2.circle(image, p_start[i], 5, color, 1)
            image = cv2.circle(image, p_end[i], 5, color, 1)
            image = cv2.line(image, p_start[i], p_end[i], color, 2) 
        return image

    im1 = np.copy(img1)
    im2 = np.copy(img2)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # the "if-else"s: align the sizes of image 1 and 2.
    if h1 > h2:
        im2 = cv2.copyMakeBorder(im2, 0, h1-h2, 0, 0, cv2.BORDER_REPLICATE)
        h = h1
    elif h1 < h2:
        im1 = cv2.copyMakeBorder(im1, 0, h2-h1, 0, 0, cv2.BORDER_REPLICATE)
        h = h2
    else: h = h1

    if w1 > w2:
        im2 = cv2.copyMakeBorder(im2, 0, 0, 0, w1-w2, cv2.BORDER_REPLICATE)
        w = w1
    elif w1 < w2:
        im1 = cv2.copyMakeBorder(im1, 0, 0, 0, w2-w1, cv2.BORDER_REPLICATE)
        w = w2
    else: w = w1

    # Mark the detected corners on the concated image
    concatIm = np.concatenate((im1, im2), axis=1)
    npList2 = [(j+w, i) for (j, i) in pList2]
    concatIm = DrawMarker(concatIm, pList1, color=(255,0,0))
    concatIm = DrawMarker(concatIm, npList2, color=(255,0,0))

    # find the corners in f1 and f2 that have a good match between each other
    bestMatch_1 = [] # the indices of corners in image 1 that have a good match in f2
    bestMatch_2 = [] # the corresponding indices of corners in image 2 
    for i, idx2 in enumerate(f1_f2):
        if i == f2_f1[idx2]:
            bestMatch_1.append(pList1[i])
            bestMatch_2.append(npList2[idx2])
    print('detected '+str(len(pList1))+ ' corners in image 1')
    print('detected '+str(len(pList2))+ ' corners in image 2')
    print('found '+str(len(bestMatch_1))+ ' matched corners between image 1 and 2')
    matchedIm = drawLines(concatIm, bestMatch_1, bestMatch_2, color=(0,255,0))
    return concatIm, matchedIm


def taskHarris(img1, img2, sigmas, folder_result, filter_ratio=0.05):
    g_1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g_2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    for sigma in sigmas:
        _, responses_1 = HarrisCornerDet(g_1, sigma=sigma, local=7, k=0.05, filter_ratio=filter_ratio)
        _, responses_2 = HarrisCornerDet(g_2, sigma=sigma, local=7, k=0.05, filter_ratio=filter_ratio)
        f1 = HarrisFeatureMaking(g_1, pList=responses_1, window_size=20)
        f2 = HarrisFeatureMaking(g_2, pList=responses_2, window_size=20)
        for M in ['SSD', 'NCC']:
            print(M, ', sigma:', sigma)
            cp, f1_f2, f2_f1 = cpMatching(f1, f2, method=M)
            concatIm, matchedIm = matchedImg(img1, img2, responses_1, responses_2, f1_f2, f2_f1)
            concatIm = cv2.cvtColor(concatIm, cv2.COLOR_RGB2BGR)
            concat_out = os.path.join(folder_result, 'Harris_'+str(sigma)+'.jpg')
            matchedIm = cv2.cvtColor(matchedIm, cv2.COLOR_RGB2BGR)
            match_out = os.path.join(folder_result, 'HarrisMatch_'+str(sigma)+'_'+M+'.jpg')
            cv2.imwrite(match_out, matchedIm)
        cv2.imwrite(concat_out, concatIm)


def taskSIFT(img1, img2, nfeatures):
    g_1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g_2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures)
    kp1, des1 = sift.detectAndCompute(g_1, None)
    kp2, des2 = sift.detectAndCompute(g_2, None)
    pList1 = [(int(kp1[i].pt[0]), int(kp1[i].pt[1])) for i in range(len(kp1))]
    pList2 = [(int(kp2[i].pt[0]), int(kp2[i].pt[1])) for i in range(len(kp2))]
    for M in ['SSD', 'NCC']:
        print(M)
        _, f1_f2, f2_f1 = cpMatching(des1, des2, method=M)
        concatIm, matchedIm = matchedImg(img1, img2, pList1, pList2, f1_f2, f2_f1)
        concatIm = cv2.cvtColor(concatIm, cv2.COLOR_RGB2BGR)
        concat_out = os.path.join(folder_result, 'SIFT.jpg')
        matchedIm = cv2.cvtColor(matchedIm, cv2.COLOR_RGB2BGR)
        match_out = os.path.join(folder_result, 'SIFTmatch_'+M+'.jpg')
        cv2.imwrite(match_out, matchedIm)
    cv2.imwrite(concat_out, concatIm)


def taskSURF(img1, img2, hessianThreshold):
    g_1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g_2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    print(hessianThreshold)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold)
    kp1, des1 = surf.detectAndCompute(g_1, None)
    kp2, des2 = surf.detectAndCompute(g_2, None)
    pList1 = [(int(kp1[i].pt[0]), int(kp1[i].pt[1])) for i in range(len(kp1))]
    pList2 = [(int(kp2[i].pt[0]), int(kp2[i].pt[1])) for i in range(len(kp2))]
    for M in ['SSD', 'NCC']:
        print(M)
        _, f1_f2, f2_f1 = cpMatching(des1, des2, method=M)
        concatIm, matchedIm = matchedImg(img1, img2, pList1, pList2, f1_f2, f2_f1)
        concatIm = cv2.cvtColor(concatIm, cv2.COLOR_RGB2BGR)
        concat_out = os.path.join(folder_result, 'SURF'+str(hessianThreshold)+'.jpg')
        matchedIm = cv2.cvtColor(matchedIm, cv2.COLOR_RGB2BGR)
        match_out = os.path.join(folder_result, 'SURFmatch_'+M+str(hessianThreshold)+'.jpg')
        cv2.imwrite(match_out, matchedIm)
    cv2.imwrite(concat_out, concatIm)


folder_task = '/home/xingguang/Documents/ECE661/hw4/hw4_Task1_Images'
folder_task2 = '/home/xingguang/Documents/ECE661/hw4/hw4_Task2_Images'
path_1_1 = os.path.join(folder_task, 'pair1', '1.JPG')
path_1_2 = os.path.join(folder_task, 'pair1', '2.JPG')
path_2_1 = os.path.join(folder_task, 'pair2', '1.JPG')
path_2_2 = os.path.join(folder_task, 'pair2', '2.JPG')
path_3_1 = os.path.join(folder_task, 'pair3', '1.jpg')
path_3_2 = os.path.join(folder_task, 'pair3', '2.jpg')
path_4_1 = os.path.join(folder_task2, 'pair1', '1.jpg')
path_4_2 = os.path.join(folder_task2, 'pair1', '2.jpg')
path_5_1 = os.path.join(folder_task2, 'pair2', '1.jpg')
path_5_2 = os.path.join(folder_task2, 'pair2', '2.jpg')

img_1_1 = cv2.cvtColor(cv2.imread(path_1_1), cv2.COLOR_BGR2RGB)
img_1_2 = cv2.cvtColor(cv2.imread(path_1_2), cv2.COLOR_BGR2RGB)
img_2_1 = cv2.cvtColor(cv2.imread(path_2_1), cv2.COLOR_BGR2RGB)
img_2_2 = cv2.cvtColor(cv2.imread(path_2_2), cv2.COLOR_BGR2RGB)
img_3_1 = cv2.cvtColor(cv2.imread(path_3_1), cv2.COLOR_BGR2RGB)
img_3_2 = cv2.cvtColor(cv2.imread(path_3_2), cv2.COLOR_BGR2RGB)
img_4_1 = cv2.cvtColor(cv2.imread(path_4_1), cv2.COLOR_BGR2RGB)
img_4_2 = cv2.cvtColor(cv2.imread(path_4_2), cv2.COLOR_BGR2RGB)
img_5_1 = cv2.cvtColor(cv2.imread(path_5_1), cv2.COLOR_BGR2RGB)
img_5_2 = cv2.cvtColor(cv2.imread(path_5_2), cv2.COLOR_BGR2RGB)
sigmas = [0.5, 0.707, 1, 1.414, 2]



folder_result = os.path.join('/home/xingguang/Documents/ECE661/hw4/Results', 'pair1')
img1 = img_1_1
img2 = img_1_2
taskHarris(img1, img2, sigmas, folder_result, filter_ratio=0.02)
taskSIFT(img1, img2, 300)
taskSURF(img1, img2, 2000)


folder_result = os.path.join('/home/xingguang/Documents/ECE661/hw4/Results', 'pair2')
img1 = img_2_1
img2 = img_2_2
taskHarris(img1, img2, sigmas, folder_result, filter_ratio=0.15)
taskSIFT(img1, img2, 200)
taskSURF(img1, img2, 20000)


folder_result = os.path.join('/home/xingguang/Documents/ECE661/hw4/Results', 'pair3')
img1 = img_3_1
img2 = img_3_2
taskHarris(img1, img2, sigmas, folder_result, filter_ratio=0.05)
taskSIFT(img1, img2, 200)
taskSURF(img1, img2, 7500)


folder_result = os.path.join('/home/xingguang/Documents/ECE661/hw4/Results', 'pair4')
img1 = img_4_1
img2 = img_4_2
taskHarris(img1, img2, sigmas, folder_result, filter_ratio=0.02)
taskSIFT(img1, img2, 200)
taskSURF(img1, img2, 15000)


folder_result = os.path.join('/home/xingguang/Documents/ECE661/hw4/Results', 'pair5')
img1 = img_5_1
img2 = img_5_2
taskHarris(img1, img2, sigmas, folder_result, filter_ratio=0.01)
taskSIFT(img1, img2, 200)
taskSURF(img1, img2, 5000)
