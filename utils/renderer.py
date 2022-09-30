"""
X-ray renderer module.
Need kornia
"""

import numpy as np
import torch
import kornia

PARAM_IRON = [
    [0, 0, 104.3061111111111], 
    [-199.26894460884833, -1.3169138497713286, 227.17803542009827], 
    [-21.894450101465132, 0.20336113292167177, 274.63740523563814]
]

PARAM_IRON_FIX = [
    [0, 0, 104.3061111111111], 
    [0, 0, 226.1507], 
    [0, 0, 225.2509],
]


PARAM_PLASTIC = [
    [0, 0, 16.054857142857145], 
    [-175.96004580018538, -0.02797999280157535, 226.59010365257998], 
    [-1.1977592197679745, 0.03212775118421846, 251.99895369583868]
]

PARAM_ALUMINUM = [
    [0, 0, 44.9139739229025], 
    [-162.0511635446029, -0.1537546525499077, 169.87370033743895],
    [68.9094475565913, -0.14688815701438654, 174.05450704994433]
]

MIN_EPS = 1e-6

def load_from_file(path, mean=0.5, std=1/2.4):
    """
    Load vertices and faces from an .obj file.
    coordinates will be normalized to N(0.5, 0.5)
    """
    with open(path, "r") as fp:
        xlist = fp.readlines()

    vertices = []
    faces = []

    for elm in xlist:
        if elm[0] == "v":
            vertices.append([float(elm.split(" ")[1]), float(elm.split(" ")[2]), float(elm.split(" ")[3])])
        elif elm[0] == "f":
            faces.append([int(elm.split(" ")[1]) - 1, int(elm.split(" ")[2]) - 1, int(elm.split(" ")[3]) - 1])

    vertices = torch.Tensor(vertices).type(torch.cuda.FloatTensor)
    faces = np.array(faces, dtype=np.int32)
    return (vertices * std) + mean, faces

def save_to_file(path, vertices, faces):
    """
    Save vertices and faces to an .obj file.
    """
    with open(path, "w") as fp:
        for i in range(vertices.shape[0]):
            fp.write("v {} {} {}\n".format(vertices[i][0], vertices[i][1], vertices[i][2]))
        for i in range(faces.shape[0]):
            fp.write("f {} {} {}\n".format(faces[i][0]+1, faces[i][1]+1, faces[i][2]+1))

def get_func_hsv(params):
    def func(x, a, b, c):
        return a * torch.exp(b * x) + c

    return lambda x: torch.cat((func(x, *params[0]), func(x, *params[1]), func(x, *params[2])), 1)

def simulate(img, material="iron"):
    """
    img: Tensor (N, 1, H, W) range(0, 1) depth image
    return: (N, 3, H, W) range(0, 1) rgb image
    """
    if material == "iron":
        max_depth = 8
        params = PARAM_IRON
    if material == "iron_fix":
        max_depth = 8
        params = PARAM_IRON_FIX
    elif material == "plastic":
        max_depth = 40
        params = PARAM_PLASTIC
    elif material == "aluminum":
        max_depth = 10
        params = PARAM_ALUMINUM
    sim = get_func_hsv(params)
    img_xray = sim(img * max_depth)
    img_xray = torch.clamp(img_xray, 0, 255) / 255
    img_xray[0][0] = img_xray[0][0] * 255 * np.pi / 90
    img_xray = kornia.color.hsv_to_rgb(img_xray)
    img_xray = torch.flip(img_xray, [1])
    img_xray = torch.clamp(img_xray, 0, 1)
    img = torch.cat((img, img, img), 1)
    mask = (img!=0)
    return img_xray, mask

def simulate_blur(img, material="iron"):
    """
    img: Tensor (N, 1, H, W) range(0, 1) depth image
    return: (N, 3, H, W) range(0, 1) rgb image
    """
    if material == "iron":
        max_depth = 8
        params = PARAM_IRON
    elif material == "plastic":
        max_depth = 40
        params = PARAM_PLASTIC
    elif material == "aluminum":
        max_depth = 10
        params = PARAM_ALUMINUM
    img = kornia.filters.gaussian_blur2d(img, (5, 5), (1, 1))
    sim = get_func_hsv(params)
    img_xray = sim(img * max_depth)
    img_xray = torch.clamp(img_xray, 0, 255) / 255
    img_xray[0][0] = img_xray[0][0] * 255 * np.pi / 90
    img_xray = kornia.color.hsv_to_rgb(img_xray)
    img_xray = torch.flip(img_xray, [1])
    img_xray = torch.clamp(img_xray, 0, 1)
    img = torch.cat((img, img, img), 1)
    mask = (img!=0)
    return img_xray, mask

def rotate_matrix(matrix):
    """
    Rotate vertices in a obj file.
    matrix: a three-element list [Rx, Ry, Rz], R for rotate degrees (angle system)
    """
    x = matrix[0] * np.pi / 180
    y = matrix[1] * np.pi / 180
    z = matrix[2] * np.pi / 180
    rx = torch.Tensor([
        [1, 0, 0],
        [0, torch.cos(x), -torch.sin(x)],
        [0, torch.sin(x), torch.cos(x)]
    ]).cuda()
    ry = torch.Tensor([
        [torch.cos(y), 0, torch.sin(y)],
        [0, 1, 0],
        [-torch.sin(y), 0, torch.cos(y)]
    ]).cuda()
    rz = torch.Tensor([
        [torch.cos(z), -torch.sin(z), 0],
        [torch.sin(z), torch.cos(z), 0],
        [0, 0, 1]
    ]).cuda()
    M = torch.mm(torch.mm(rx, ry), rz)
    return M

def is_in_triangle(point, tri_points):
    """
    Judge whether the point is in the triangle
    """
    tp = tri_points

    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = torch.dot(v0.T, v0)
    dot01 = torch.dot(v0.T, v1)
    dot02 = torch.dot(v0.T, v2)
    dot11 = torch.dot(v1.T, v1)
    dot12 = torch.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 < 1e-4:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v <= 1) & (inverDeno != 0)

def get_point_weight(point, tri_points):
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = torch.dot(v0.T, v0)
    dot01 = torch.dot(v0.T, v1)
    dot02 = torch.dot(v0.T, v2)
    dot11 = torch.dot(v1.T, v1)
    dot12 = torch.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 < 1e-4:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2

def are_in_triangles(points, tri_points):
    """
    Judge whether the points are in the triangles
    assume there are n points, m triangles
    points shape: (n, 2)
    tri_points shape: (m, 3, 2)
    """
    tp = tri_points
    n = points.shape[0]
    m = tp.shape[0]

    # vectors
    # shape: (m, 2)
    v0 = tp[:, 2, :] - tp[:, 0, :]
    v1 = tp[:, 1, :] - tp[:, 0, :]
    # shape: (n, m, 2)
    v2 = points.unsqueeze(1).repeat(1, m, 1) - tp[:, 0, :]

    # dot products
    # shape: (m, 2) =sum=> (m, 1)
    dot00 = torch.mul(v0, v0).sum(dim=1)
    dot01 = torch.mul(v0, v1).sum(dim=1)
    dot11 = torch.mul(v1, v1).sum(dim=1)
    # shape: (n, m, 2) =sum=> (n, m, 1)
    dot02 = torch.mul(v2, v0).sum(dim=2)
    dot12 = torch.mul(v2, v1).sum(dim=2)

    # barycentric coordinates
    # shape: (m, 1)
    inverDeno = dot00*dot11 - dot01*dot01
    zero = torch.zeros_like(inverDeno)
    inverDeno = torch.where(inverDeno < MIN_EPS, zero, 1 / inverDeno)

    # shape: (n, m, 1)
    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno
    
    w0 = 1 - u - v
    w1 = v
    w2 = u

    # check if point in triangle
    return (u >= -MIN_EPS) & (v >= -MIN_EPS) & (u + v <= 1+MIN_EPS) & (inverDeno != 0), w0, w1, w2

def ball2depth(vertices, faces, h, w):
    """
    Save obj file as a depth image, z for depth and x,y for position
    a ball with coord in [0, 1]
    h, w: the output image height and width
    return: a depth image in shape [h, w]
    """
    vertices = torch.clamp(vertices, 0, 1)
    vs = vertices.clone()
    vs[:, 0] = vertices[:, 0] * w
    vs[:, 1] = vertices[:, 1] * h
    vertices = vs
    faces = torch.LongTensor(faces).cuda()
    
    points = torch.Tensor([(i, j) for i in range(h) for j in range(w)]).cuda()
    tri_points = vertices[faces, :2]
    in_triangle, w0, w1, w2 = are_in_triangles(points, tri_points)
    
    point_depth = w0 * vertices[faces[:, 0], 2] + w1 * vertices[faces[:, 1], 2] + w2 * vertices[faces[:, 2], 2]
    
    min_depth = torch.min(torch.where(in_triangle, point_depth, torch.full_like(point_depth, 9999)), dim=1).values
    max_depth = torch.max(torch.where(in_triangle, point_depth, torch.full_like(point_depth, -9999)), dim=1).values

    image = torch.clamp(max_depth - min_depth, 0, 1).view(h, w)
    
    return image


def plane2depth(vertices, faces, h, w):
    """
    Save obj file as a depth image, x,y for position
    h, w: the output image height and width
    """
    vertices = torch.clamp(vertices, 0, 1)
    vs = vertices.clone()
    vs[:, 0] = vertices[:, 0] * w
    vs[:, 1] = vertices[:, 1] * h
    vertices = vs

    depth = torch.zeros((h, w))

    for i in range(faces.shape[0]):
        tri = faces[i, :]

        # get rectangular bounding box
        umin = torch.floor(torch.min(vertices[tri, 0]))
        umin = torch.clamp(umin, 0, h-1)
        umax = torch.ceil(torch.max(vertices[tri, 0]))
        umax = torch.clamp(umax, 0, h-1)
        
        vmin = torch.floor(torch.min(vertices[tri, 1]))
        vmin = torch.clamp(vmin, 0, w-1)
        vmax = torch.ceil(torch.max(vertices[tri, 1]))
        vmax = torch.clamp(vmax, 0, w-1)

        # get depth value for each pixel in the triangular
        for u in range(int(umin.item()), int(umax.item())+1):
            for v in range(int(vmin.item()), int(vmax.item())+1):
                if is_in_triangle(torch.Tensor([u, v]).cuda(), vertices[tri, :2]):
                    w0, w1, w2 = get_point_weight(torch.Tensor([u, v]).cuda(), vertices[tri, :2])
                    point_depth = w0 * vertices[tri[0], 2] + w1 * vertices[tri[1], 2] + w2 * vertices[tri[2], 2]
                    if point_depth > depth[v, u]:
                        depth[v, u] = point_depth
    
    image = torch.clamp(depth, 0, 1)
    
    return image


def adj_list(vertices, faces):
    adj_ls = [[] for _ in range(vertices.shape[0])]
    for i in range(faces.shape[0]):
        adj_ls[faces[i][0]].extend([faces[i][1], faces[i][2]])
        adj_ls[faces[i][1]].extend([faces[i][0], faces[i][2]])
        adj_ls[faces[i][2]].extend([faces[i][0], faces[i][1]])
        
    for i in range(len(adj_ls)):
        adj_ls[i] = list(set(adj_ls[i]))
        
    return adj_ls


def tvloss(vertices, vertices_adv, adj_ls, coe=0):
    total_loss = 0.0
    cnt = 0
    for i in range(vertices.shape[0]):
        vi = vertices_adv[i] - vertices[i]
        for q in adj_ls[i]:
            vq = vertices_adv[q] - vertices[q]
            norm = torch.norm(vi - vq) ** 2
            total_loss += norm
            cnt += 1
            
    total_loss /= cnt

    loss2 = 0.0
    for i in range(vertices.shape[0]):
        vi = vertices_adv[i] - vertices[i]
        loss2 += torch.norm(vi) ** 2
    loss2 /= vertices.shape[0]
            
    total_loss += coe * loss2
    
    return total_loss
        