import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

path_root = '/home/xingguang/Documents/ECE661/hw9/Files'

# for my own images:
# img_dataset = os.path.join(path_root, 'Dataset')
# images = []
# for i in range(len(os.listdir(img_dataset))):
#     im_path = os.path.join(img_dataset, str(i+1)+'.jpg')
#     images.append(cv2.resize(cv2.imread(im_path), (750, 1000)))

# for the given images
img_dataset = os.path.join(path_root, 'Dataset')
images = []
for i in range(len(os.listdir(img_dataset))):
    im_path = os.path.join(img_dataset, 'Pic_'+str(i+1)+'.jpg')
    images.append(cv2.imread(im_path))


def make_pattern(grid_size, hline_num, vline_num):
    # ma
    x = np.linspace(0, grid_size*(vline_num-1), vline_num)
    y = np.linspace(0, grid_size*(hline_num-1), hline_num)
    xv, yv = np.meshgrid(x, y)
    return np.concatenate([xv.reshape((-1,1)), yv.reshape((-1,1))], axis=1)


def cvtPoint(lines, r):
    rho = lines[:, 0]
    theta = lines[:, 1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = np.array([x0 + r * (-b), y0 + r * a]).T
    pt2 = np.array([x0 - r * (-b), y0 - r * a]).T
    return pt1, pt2


def rearrange(p1, p2, v=True):
    p1_list = []
    p2_list = []
    for i in range(p1.shape[0]):
        if v:
            if p1[i, 1] > 0 and p2[i, 1] <0:
                p1_list.append(p1[i])
                p2_list.append(p2[i])
            elif p1[i, 1] < 0 and p2[i, 1] > 0:
                p1_list.append(p2[i])
                p2_list.append(p1[i])               
        else:
            if p1[i, 0] > 0 and p2[i, 0] <0:
                p1_list.append(p1[i])
                p2_list.append(p2[i])
            elif p1[i, 0] < 0 and p2[i, 0] > 0:
                p1_list.append(p2[i])
                p2_list.append(p1[i])
    return np.concatenate(p1_list, axis=0).reshape((-1, 2)),\
            np.concatenate(p2_list, axis=0).reshape((-1, 2))


def nmsLines(lines, p1, p2, nms_ratio=0.25, v=True):
    if v:
        # d = np.abs(lines[:, 0] * np.cos(lines[:, 1]))
        d = lines[:, 0] * np.cos(lines[:, 1])
        d_abs = np.abs(d)
        nms_thres = nms_ratio * (np.max(d_abs) - np.min(d_abs)) / 7
    else:
        # d = np.abs(lines[:, 0] * np.sin(lines[:, 1]))
        d = lines[:, 0] * np.sin(lines[:, 1])
        d_abs = np.abs(d)
        nms_thres = nms_ratio * (np.max(d_abs) - np.min(d_abs)) / 9
    
    idx = np.argsort(d, axis=0)
    d_sort = d[idx]
    valid_id = []
    temp_ids = []
    for i in range(d_sort.shape[0]-1):
        if i == 0:
            temp_ids.append(idx[i])
        if d_sort[i+1] - d_sort[i] < nms_thres:
            temp_ids.append(idx[i+1])
        else:
            valid_id.append(temp_ids)
            temp_ids = [idx[i+1]]
        if i == d_sort.shape[0]-2:
            valid_id.append(temp_ids)

    p1_list = []
    p2_list = []
    for ids in valid_id:
        p1_list.append(np.average(p1[ids], axis=0))
        p2_list.append(np.average(p2[ids], axis=0))
    return np.concatenate(p1_list, axis=0).reshape((-1, 2)),\
            np.concatenate(p2_list, axis=0).reshape((-1, 2))


def drawLines(img, p1, p2, c):
    out = img.copy()
    for i in range(0, p1.shape[0]):
        pt1 = (int(p1[i, 0].item()), int(p1[i, 1].item()))
        pt2 = (int(p2[i, 0].item()), int(p2[i, 1].item()))
        cv2.line(out, pt1, pt2, c, 2)
    return out


def drawPoints(img, p, with_text=True, c=(0, 0, 255)):
    out = img.copy()
    for i in range(p.shape[0]):
        point = (int(p[i,0].item()), int(p[i,1].item()))
        out = cv2.circle(out, point, radius=3, color=c, thickness=-1)
        if with_text:
            out = cv2.putText(out, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA) 
    return out


def find_itsc(vp1, vp2, hp1, hp2):
    vp1 = np.append(vp1, np.ones((vp1.shape[0],1)), axis=1)
    vp2 = np.append(vp2, np.ones((vp1.shape[0],1)), axis=1)
    hp1 = np.append(hp1, np.ones((hp1.shape[0],1)), axis=1)
    hp2 = np.append(hp2, np.ones((hp2.shape[0],1)), axis=1)
    v_lines = np.cross(vp1, vp2)
    h_lines = np.cross(hp1, hp2)
    points = []
    for i in range(h_lines.shape[0]):
        itscs = np.cross(v_lines, h_lines[i])
        points.append(itscs[:,:2]/itscs[:, 2].reshape((-1,1)))
    return np.concatenate(points, axis=0)


def extract_intersections_from_image(in_img, nms_ratio, i, r=0.5, t=50, save_imgs=True):
    raw_img = np.copy(in_img) 
    gray = cv2.GaussianBlur(cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), (3, 3), 1.4)
    edges = cv2.Canny(gray, 255*1.5,255)
    lines = cv2.HoughLines(edges, 1, r*np.pi / 180, t)
    lines = np.squeeze(lines)   
    # cos(theta)^2 > cos(pi/2)^2 ~ verticle
    vlines = lines[np.where(np.cos(lines[:,1]) ** 2 > 0.5)]
    vp1, vp2 = cvtPoint(vlines, 1000)
    vp1, vp2 = rearrange(vp1, vp2, v=True)
    vp1, vp2 = nmsLines(vlines, vp1, vp2, nms_ratio, v=True)
    if vlines is not None:
        img = drawLines(raw_img, vp1, vp2, c=(0,255,0))

    # cos(theta)^2 <= cos(pi/2)^2 ~ horizontal
    hlines = lines[np.where(np.cos(lines[:,1]) ** 2 <= 0.5)]
    hp1, hp2 = cvtPoint(hlines, 1000)
    hp1, hp2 = rearrange(hp1, hp2, v=False)
    hp1, hp2 = nmsLines(hlines, hp1, hp2, nms_ratio, v=False)
    if hlines is not None:
        img = drawLines(img, hp1, hp2, c=(255,0,0))
    intersections = find_itsc(vp1, vp2, hp1, hp2)
    img_with_points = drawPoints(raw_img, p=intersections)
    if save_imgs:
        p1, p2 = cvtPoint(lines, 1000)
        img_Hough = drawLines(raw_img, p1, p2, c=(255,255,255))
        cv2.imwrite(os.path.join(path_root, "HoughTrans", "h_"+str(i)+'.jpg'), img_Hough)
        cv2.imwrite(os.path.join(path_root, "Hough_filtered", "hf_"+str(i)+'.jpg'), img)
        cv2.imwrite(os.path.join(path_root, "Canny", "edge_"+str(i)+'.jpg'), edges)
        cv2.imwrite(os.path.join(path_root, "Intersections", "itsc_"+str(i)+'.jpg'), img_with_points)
    return intersections


def findHomoproj(source, target):
    # target = source * H^T
    def F_unit(source_point, target_point):
        x, y = source_point[0], source_point[1]
        x_, y_ = target_point[0], target_point[1]
        return np.asarray([[x, y, 1, 0, 0, 0, -x*x_, -y*x_], 
                        [0, 0, 0, x, y, 1, -x*y_, -y*y_]])
    F_list = [F_unit(source[i], target[i]) for i in range(source.shape[0])]
    F = np.concatenate(F_list, axis=0)
    T_span = target.reshape((-1,1))
    H_param = np.dot(np.linalg.pinv(F), T_span)
    H = np.ones((9, 1))
    H[:8, :] = H_param
    return H.reshape((3, 3))

def findOmega(H_list):
    def V_unit(H):
        h11, h12, h13 = (H[0,0], H[1,0], H[2,0])
        h21, h22, h23 = (H[0,1], H[1,1], H[2,1])
        return np.asarray([[h11*h21, h11*h22+h12*h21, h12*h22, \
                h13*h21+h11*h23, h13*h22+h12*h23, h13*h23],\
                [h11**2-h21**2, 2*h11*h12-2*h21*h22, h12**2-h22**2, \
                2*h11*h13-2*h21*h23, 2*h12*h13-2*h22*h23, h13**2-h23**2]])
    V_list = [V_unit(H) for H in H_list]
    V = np.concatenate(V_list, axis=0)
    _, _, v = np.linalg.svd(V)
    b = v[-1]
    Omega = np.array([[b[0],b[1],b[3]], [b[1],b[2],b[4]], [b[3],b[4],b[5]]])
    return Omega

def findK(omega):
    w = np.copy(omega)
    v0 = (w[0,1]*w[0,2] - w[0,0]*w[1,2]) / (w[0,0]*w[1,1] - w[0,1]**2)
    lamda = w[2,2] - (w[0,2]**2 + v0 * (w[0,1]*w[0,2] - w[0,0]*w[1,2])) / w[0,0]
    a_x = np.sqrt(lamda / w[0,0])
    a_y = np.sqrt(lamda * w[0,0] / (w[0,0]*w[1,1] - w[0,1]**2))
    s = -w[0,1] * a_x**2 * a_y / lamda
    u0 = s * v0 / a_y - w[0,2] * a_x**2 / lamda
    K = np.array([[a_x, s, u0], [0, a_y, v0], [0, 0, 1]])
    return K

def findRt(H_list, K):
    R_list = []
    t_list = []
    for H in H_list:
        r12_t = np.dot(np.linalg.inv(K), H)
        lamda = 1 / np.linalg.norm(r12_t[:,0])
        r12_t = lamda * r12_t
        r3 = np.cross(r12_t[:,0], r12_t[:, 1])
        Q = np.copy(r12_t)
        Q[:, 2] = r3
        u, _, v = np.linalg.svd(Q)
        R = np.dot(u, v)
        R_list.append(R)
        t_list.append(r12_t[:, 2].copy())
    return R_list, t_list


def projTransform(H, source):
    nps = source.shape[0]
    source_rep = np.concatenate((source, np.ones((nps,1))), axis=1)
    t_homo = np.dot(H, source_rep.T).T
    t_norm = t_homo[:,:2] / t_homo[:,2].reshape((nps,1))
    return t_norm

def construct_params(R_list, t_list, K):
    Rt_list = []
    for R, t in zip(R_list, t_list):
        phi = np.arccos((np.trace(R)-1)/2)
        w = phi / (2 * np.sin(phi)) * np. asarray([
            R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
        Rt_list.append(np.append(w, t))
    K_param = np.asarray([K[0,0], K[0,1], K[0,2], K[1,1], K[1,2]])
    params = np.append(K_param, np.concatenate(Rt_list))
    return params

def reconstruct_mat(lm_params):
    N = int((lm_params.shape[0]-5) / 6)
    k = lm_params[:5]
    K = np.array([[k[0], k[1], k[2]], [0, k[3], k[4]], [0, 0, 1]])
    R_list = []
    t_list = []
    for i in range(N):
        w = lm_params[5+i*6:8+i*6]
        t = lm_params[8+i*6:11+i*6]
        phi = np.linalg.norm(w)
        wx = np.array([[0, -w[2], w[1]], [w[2],0,-w[0]], [-w[1],w[0],0]])
        R = np.eye(3) + np.sin(phi)/phi*wx + (1-np.cos(phi))/(phi**2) * np.dot(wx, wx)
        R_list.append(R)
        t_list.append(t)
    return R_list, t_list, K


def radial_distort(itscs, k1, k2, x0, y0):
    # Remove radial distortions
    x = itscs[:,0]
    y = itscs[:,1]
    r = (x-x0)**2 + (y-y0)**2
    x_rad = x + (x-x0) * (k1*r + k2*(r**2)) 
    y_rad = y + (y-y0) * (k1*r + k2*(r**2)) 
    return np.hstack([x_rad.reshape((-1, 1)), y_rad.reshape((-1, 1))])


def cost_Func(params, itsc_list, pattern, with_rd=False):
    num_Img = len(itsc_list)
    if with_rd:
        R_list, t_list, K = reconstruct_mat(params[:-2])
        k1 = params[-2]
        k2 = params[-1]
        x0 = params[2]
        y0 = params[4]
    else:
        R_list, t_list, K = reconstruct_mat(params)
    Proj_pattern = []
    for R, t in zip(R_list, t_list):
        Rt = np.concatenate([R[:,0:1], R[:,1:2], t.reshape((-1,1))], axis=1)
        H = np.dot(K, Rt)
        reconst_p = projTransform(H, pattern)
        if with_rd:
            reconst_p = radial_distort(reconst_p, k1, k2, x0, y0)
        Proj_pattern.append(reconst_p)
    projec_itscs = np.concatenate(Proj_pattern, axis=0)
    gt_ptrns = np.concatenate(itsc_list, axis=0)
    diff = projec_itscs - gt_ptrns
    return diff.flatten()

def error(diff):
    diff = diff.reshape((-1, 2))
    diff_norm = np.linalg.norm(diff, axis=1)
    e = np.average(diff_norm)
    var = np.var(diff_norm)
    max_d = np.max(diff_norm)
    return np.array([e, var, max_d])

def measure(diff):
    diff = diff.reshape((-1, 2))
    diff_norm = np.linalg.norm(diff, axis=1)
    num_Img = int(diff_norm.shape[0]/80)
    measured = []
    for i in range(num_Img):
        measure_imgi = {}
        current_d = diff_norm[i*80:i*80+80]
        measure_imgi["Means"] = np.average(current_d)
        measure_imgi["Variances"] = np.var(current_d)
        measure_imgi["max_distance"] = np.max(current_d)
        measured.append(measure_imgi)
    return measured


def measure_proj(pattern, itsc_list, valid_idlist, images, params, status='before'):
    if params.shape[0] % 6 == 1:
        R_list, t_list, K = reconstruct_mat(params[:-2])
        k1 = params[-2]
        k2 = params[-1]
        x0 = params[2]
        y0 = params[4]
    else:
        R_list, t_list, K = reconstruct_mat(params)
    H_list = []
    for R, t in zip(R_list, t_list):
        Rt = np.concatenate([R[:,0:1], R[:,1:2], t.reshape((-1,1))], axis=1)
        H = np.dot(K, Rt)
        H_list.append(H)

    diff_list = []
    for i, H in enumerate(H_list):
        img_idx = valid_idlist[i]
        img_i = images[img_idx]
        projed = projTransform(H, pattern)
        if params.shape[0] % 6 == 1:
            projed = projTransform(H, pattern)
            projed = radial_distort(projed, k1, k2, x0, y0)
        else:
            projed = projTransform(H, pattern)
        proj_img = drawPoints(img_i, projed, with_text=True, c=(0, 255, 255))
        cv2.imwrite(os.path.join(path_root, "proj_"+status,\
                     "to"+str(img_idx+1)+'.jpg'), proj_img)
        diff = itsc_list[i] - projed
        diff_list.append(diff)
    e = error(np.array(diff_list).flatten())
    measure_params = measure(np.array(diff_list).flatten())
    return e, measure_params    

def reproject(static_idx, valid_idlist, images, params, itsc_list, status='before'):
    img_idx = valid_idlist[static_idx]
    static_img = np.copy(images[img_idx])
    static_itscs = itsc_list[static_idx]
    static_img = drawPoints(static_img, static_itscs)
    if status == 'withrad':
        R_list, t_list, K = reconstruct_mat(params[:-2])
        k1 = params[-2]
        k2 = params[-1]
        x0 = params[2]
        y0 = params[4]
    else:
        R_list, t_list, K = reconstruct_mat(params)
    H_list = []
    for R, t in zip(R_list, t_list):
        Rt = np.concatenate([R[:,0:1], R[:,1:2], t.reshape((-1,1))], axis=1)
        H = np.dot(K, Rt)
        H_list.append(H)
    sH = H_list[static_idx]
    diff_list = []
    for i, H in enumerate(H_list):
        if i == static_idx:
            continue
        img_i = valid_idlist[i]
        H_i_s = np.dot(sH, np.linalg.inv(H))
        reprojed = projTransform(H_i_s, itsc_list[i])
        reproj_img = drawPoints(static_img, reprojed, with_text=False, c=(0, 255, 0))
        cv2.imwrite(os.path.join(path_root, "reproj_"+status, \
                    str(img_i+1)+'to'+str(valid_idlist[static_idx]+1)+'.jpg'), reproj_img)
        diff = static_itscs - reprojed
        diff_list.append(diff)
    e = error(np.array(diff_list).flatten())
    measure_params = measure(np.array(diff_list).flatten())
    return e, measure_params

# --------------------------------Main function -------------------------------
# -----------------------------------------------------------------------------
# Load all images, extract the intersections and compute H's
# -----------------------------------------------------------------------------
gt_pattern = make_pattern(grid_size=10, hline_num=10, vline_num=8)

H_list = []
valid_image_ids = []
itsc_list = []

# for the given images:
nms = 0.25
rsl = 0.5
thres = 50

# for my own images:
# nms = 0.28
# rsl = 0.6
# thres = 70
for i, image in enumerate(images):
    itscs = extract_intersections_from_image(image, nms, i+1, rsl, thres, save_imgs=False)
    if itscs.shape[0] == 80:
        H = findHomoproj(gt_pattern, itscs)
        H_list.append(H)
        itsc_list.append(itscs)
        valid_image_ids.append(i)
print("Total number of images being detected 80 intersections:", len(valid_image_ids))
print("They are (indices):", valid_image_ids)

# -----------------------------------------------------------------------------
# Compute the intrinsic matrix K and rotation matrices
# -----------------------------------------------------------------------------
omega = findOmega(H_list)
K = findK(omega)
R_list, t_list = findRt(H_list, K)

# -----------------------------------------------------------------------------
# LM optimization to refine the camera matrices
# -----------------------------------------------------------------------------
params = construct_params(R_list, t_list, K)
loss = cost_Func(params, itsc_list, pattern=gt_pattern)
sol = least_squares(cost_Func, params, method = 'lm', args=[itsc_list, gt_pattern])
R_list_refined, t_list_refined, K_refined = reconstruct_mat(sol.x)
sol_rad = least_squares(cost_Func, np.append(params, np.array([0, 0])), \
                        method = 'lm', args=[itsc_list, gt_pattern, True])
R_list_refined_wr, t_list_refined_wr, K_refined_wr = reconstruct_mat(sol_rad.x[:-2])
print("k1, k2:", sol_rad.x[-2:]) 
print("K before refinement:", K)
print("K after refinement without radial distortion:", K_refined)
print("K after refinement with radial distortion:", K_refined_wr)

# -----------------------------------------------------------------------------
# Project the ground truth pattern to every image
# -----------------------------------------------------------------------------
_, measure_nolm= measure_proj(gt_pattern, itsc_list, valid_image_ids, images, params, "before")
_, measure_lm= measure_proj(gt_pattern, itsc_list, valid_image_ids, images, sol_rad.x, "after")
# measure overall accuracy
print(error(cost_Func(params, itsc_list, pattern=gt_pattern)))
print(error(cost_Func(sol.x, itsc_list, pattern=gt_pattern)))
# measure accuracy of selected views
print(error(cost_Func(sol_rad.x, itsc_list, pattern=gt_pattern, with_rd=True)))
print("measure the projection on image", valid_image_ids[10]+1, measure_nolm[10], measure_lm[10])
print("measure the projection on image", valid_image_ids[36]+1, measure_nolm[36], measure_lm[36])
# [0, 7, 18, 36] for given images
# [3, 5, 8, 15] for my own images
for i in [0, 7, 18, 36]:
    print("Project the gt pattern to image:", valid_image_ids[i]+1, "the ratation matrices:")
    print(R_list[i], t_list[i])
    print(R_list_refined[i], t_list_refined[i])
    print(R_list_refined_wr[i], t_list_refined_wr[i])

# -----------------------------------------------------------------------------
# Project the corners of all views to the fixed image
# -----------------------------------------------------------------------------

fix_id = 1
e_before, measured_before = reproject(fix_id, valid_image_ids, images, params, itsc_list, status='before')
print("Overall reprojection error, var, and max distance before refinement:", e_before)

e_after, measured_after = reproject(fix_id, valid_image_ids, images, sol.x, itsc_list, status='after')
print("Overall reprojection error, var, and max distance after refinement:", e_after)

# [3, 7, 9, 36] for the given images
# [2, 4, 7, 14] for my own images
for i in [3, 7, 9, 36]:
    if i < fix_id:
        print("image:", valid_image_ids[i]+1)
    else:
        print("image:", valid_image_ids[i]+2)
    print(measured_before[i])
    print(measured_after[i])