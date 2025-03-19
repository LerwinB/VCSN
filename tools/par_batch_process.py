import cv2
import os
from tqdm import tqdm
import numpy as np
import argparse
from multiprocessing import Pool

cpus = 6
color2index = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (255, 255, 0): 3,
    (0, 0, 255): 4,
    (255, 0, 255): 5,
    (0, 0, 124): 6,
    (202, 202, 202): 7,
    (255, 255, 255): 8,
    (124, 0, 0): 9,
    (0, 185, 0): 10,
    (185, 185, 0): 11,
    (0, 124, 124): 12,
    (0, 255, 255): 13,
    (0, 200, 0): 14,
    (50, 255, 0): 15,
    (50, 124, 0): 16,
    (124, 255, 0): 17,
    (50, 200, 0): 18,
    (0, 255, 50): 19
}

def get_path(workdir):
    with open(workdir) as file:
        paths = file.readlines()
        paths_new = [workdir+path[1:-1] for path in paths]
    return paths


def rgb2gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    return img_blur
    
def rgb2gray_new(img, mean,sigma):
    noise = np.random.normal(mean,sigma,img.shape).astype(np.uint8)
    img_noise = cv2.add(img, noise)
    img_gray = cv2.cvtColor(img_noise, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    return img_blur

def src_process(paths,workdir):
    mean = 0
    sigma = 0.3
    paths_new = [workdir + path[1:-1] for path in paths]
    for path in tqdm(paths_new, position=0 ,leave=True):
        img = cv2.imread(path)
        img_blur = rgb2gray(img)
        cv2.imwrite(path, img_blur)

def fun(im):
    img = cv2.imread(im)
    img_blur = rgb2gray(img)
    cv2.imwrite(im, img_blur)

def parsrc_process(paths,workdir):
    paths_new = [workdir + path[1:-1] for path in paths]
    b = (im for im in paths_new)
    c = range(100)
    type(paths_new)
    with Pool(cpus) as p:
        r = list(tqdm(p.imap(fun, paths_new), total=len(paths), desc="src"))
    #@parfor(c, total=len(paths))
        

def label_vis(paths, workdir):
    paths_new = [workdir + path[1:-1] for path in paths]
    for path in tqdm(paths_new,position=0 ,leave=True):
        #print(path)
        img = cv2.imread(path)
        cv2.imwrite(path, img)

#@numba.jit(nopython=True)
def rgb2mask_new(img_src):
    # RGB值对应的标签值

    img = cv2.cvtColor(img_src.copy(), cv2.COLOR_BGR2RGB)
    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3
    mask = np.zeros(img.shape[:2])

    for color, label in color2index.items():
        color = np.array(color)
        mask[np.all(img == color, axis=-1)] =label
    return mask


def rgb2mask(img_src):
    # RGB值对应的标签值
    img = cv2.cvtColor(img_src.copy(), cv2.COLOR_BGR2RGB)
    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3
    W = np.power(256, [[0],[1],[2]])
    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)
    mask = np.zeros(img_id.shape)
    for i, c in enumerate(values):
        try:
            mask[img_id==c] = color2index[tuple(img[img_id==c][0])]
        except:
            pass
    return mask


def label_rename(paths, workdir):
    paths_new = [workdir + path[1:-1] for path in paths]
    for path in tqdm(paths_new):
        # print(path)
        label_index = path.find("label")
        path_label = path[:label_index+5]
        path_label_new = path_label + "_color"
        label_new = path_label_new + path[label_index+5:]
        if not os.path.exists(path_label_new):
            os.rename(path_label, path_label_new)
        if not os.path.exists(path_label):
            os.mkdir(path_label)
        #print(label_new)
        img = cv2.imread(label_new)
        mask = rgb2mask(img)
        cv2.imwrite(path, mask)

def label_fun(path):
    path_color = path.replace('label','label_color')
    img = cv2.imread(path_color)
    mask = rgb2mask(img)
    cv2.imwrite(path, mask)

def parlabel_process(paths, workdir):
    paths_new = [workdir + path[1:-1] for path in paths]
    with Pool(cpus) as p:
        r = list(tqdm(p.imap(label_fun, paths_new), total=len(paths), desc="labelfun"))
def label_process(paths, workdir):
    paths_new = [workdir + path[1:-1] for path in paths]
    for path in tqdm(paths_new, position=0 ,leave=True):
        # print(path)
        label_index = path.find("label")
        path_label = path[:label_index+5]
        path_label_new = path_label + "_color"
        label_new = path_label_new + path[label_index+5:]
        if not os.path.exists(path_label_new):
            os.mkdir(path_label_new)
        if not os.path.exists(path_label):
            os.mkdir(path_label)
        #print(label_new)
        img = cv2.imread(path)
        mask = rgb2mask(img)
        cv2.imwrite(label_new, img)
        cv2.imwrite(path, mask)

def process(work_dir, sats):
    for sat in sats:
        label_paths = get_path(work_dir+sat+"/"+"labelpath.txt")
        src_paths = get_path(work_dir+sat+"/"+"srcpath.txt")
        if len(label_paths) != 283500:
            print("error!")
        print(sat)
        src_process(src_paths, work_dir[:-1])
        label_process(label_paths, work_dir[:-1])




if __name__ == '__main__':
    # python batch_process.py satname
    try:
        parser = argparse.ArgumentParser(description='batch process')
        parser.add_argument('satname', type=str, help='name of sat',default="Gaofen13")
        parser.add_argument('workdir', type=str, help='name of sat',default="E:/Epic/documents/batchcam/Screenshots1024/")
        args = parser.parse_args() 
        sat = args.satname
        work_dir = args.workdir

    except:
        sat = "ROOSTER"
        work_dir = "data/UESAT_RGB_53/Screenshots0528/"
    
    label_paths = get_path(work_dir+sat+"/"+"labelpath.txt")
    src_paths = get_path(work_dir+sat+"/"+"srcpath.txt")
    if len(label_paths) != 283500:
        print("error!")
    print(sat)
    #label_vis(label_paths, work_dir[:-1])
    #parsrc_process(src_paths, work_dir[:-1])
    parlabel_process(label_paths, work_dir[:-1])
    #label_rename(label_paths, work_dir[:-1])

