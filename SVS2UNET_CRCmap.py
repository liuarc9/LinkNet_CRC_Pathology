import os
import glob
import cv2 as cv
import pandas as pd
import openslide
import numpy as np
import concurrent.futures
import torch
import torch.nn as nn
import json
import sys
sys.path.append("/data/lar/program/TIL_task")
import cv2
import torchvision.transforms as transforms
from PIL import Image

from IdenMatterLocation import *
from ImgDetail_unet import *

#models
import utils
import torchvision.models as models
from unet import Unet
from model_NCRF import MODELS
from NormMacenko import normal_Macenko

from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # using gpu where location is 0
cpu_num = 10
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(20)
train_transformer = transforms.Compose([
                            #transforms.ToPILImage(),
                            transforms.ToTensor(),
                        ])

"""
This code is used to visulaize uunet_model images' results
"""

if __name__ == '__main__':

    """
    slidingLength = 10
    uunet_model_img_size = 100
    poolNum = 40
    WSI_csv_path = "/data/lar/BRCA/brca_info.csv"
    SVSfolderPath = "/data/lar/BRCA/BRCA_data/"
    thumbs_paths = "/data/lar/BRCA/BRCA_thumb/"  # OTSU thumb paths
    img_save_type = ".png"
    umap_probability_save_folder_path = "/data/lar/umap/"
    """

    slidingLength = 10
    unet_model_img_size = 256
    poolNum = 40
    WSI_csv_path = "/data/lar/unet/U_net_map/crc_info.csv"
    SVSfolderPath = "/data/lar/CRC_data/gdc_linux/"
    thumbs_paths = "/data/lar/CRC_data/BRCA_gray/"  # OTSU thumb paths
    img_save_type = ".png"
    umap_probability_save_folder_path = "/data/lar/unet/U_net_map/"
    save_folder_path = "/data/lar/unet/U_net_map/"    # final result
    error_report_path = "/data/lar/unet/U_net_map/error_report2.txt"

    HERef = np.load("/data/lar/program/normalization/HERef.npy")
    maxCRef = np.load("/data/lar/program/normalization/maxCRef.npy")

    if not os.path.exists(umap_probability_save_folder_path):
        os.mkdir(umap_probability_save_folder_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU

    # classify
    resnet18_model = models.resnet18(pretrained=True)
    net1 = resnet18_model
    num_fits = resnet18_model.fc.in_features
    net1.fc = nn.Linear(num_fits, 9)
    net1.load_state_dict(
        torch.load('/data/lar/program/TIL_task/best_ResNetBCEWithLogitsLoss_crc.pth', map_location=device))
    net1.to(device=device)
    net1.eval()
    unet_model = Unet()
    # load model parameters
    # evaluation mode
    mode = "predict"
    print("Load model done")
    # test
    img = cv2.imread('/data/lar/lymphocyte/data_color_normal/im88.tif')

    # open csv
    data = pd.read_csv(WSI_csv_path)

    paths = [data['png_paths'].iloc[i] for i in range(len(data['png_paths']))]

    label = {'ADI': [141, 141, 141], 'BACK': [33, 23, 23], 'DEB': [189, 41, 153], 'LYM': [30, 149, 191], 'MUC': [250, 216, 206],
             'MUS': [50, 205, 50], 'NORM': [204, 102, 0], 'STR': [247, 188, 10], 'TUM': [230, 73, 22]}

    colors = [[141, 141, 141], [33, 23, 23], [189, 41, 153], [30, 149, 191], [250, 216, 206],
            [50, 205, 50], [204, 102, 0], [247, 188, 10], [230, 73, 22]]


    for index, path in enumerate(paths):
        if index <= 4:
            continue
        TILs = 0
        til = 0
        while TILs == 0:
            #path = '/data/lar/CRC_data/thumb/TCGA-CA-5255-01Z-00-DX1.77310ae2-9c5f-48c4-9754-c5b30d287089.png'
            midname = path[path.rindex("/") + 1:path.rindex(".")]
            print('midname', midname)
            unet_model_img_save_path = umap_probability_save_folder_path + "/" + midname + img_save_type

            # skip completed part
            try:
                unet_model_img_path_mark = glob.glob(unet_model_img_save_path)[0]
                if unet_model_img_path_mark:
                    continue

            except IndexError:

                pass
            # print("Midname:", midname)
            SVSpath = glob.glob(SVSfolderPath + "/*/" + midname + "*")[0]
            # data[data["svs_name"] == midname]["svs_paths"].iloc[0]
            # print("SVSPath:", SVSpath)
            slide = openslide.open_slide(SVSpath)
            # print("slide_property:", slide.properties)
            [w0, h0] = slide.level_dimensions[0]
            # print("DebugL:", w0)
            # some slides may don't have dimension 3
            try:

                [w3, h3] = slide.level_dimensions[3]

            except IndexError:

                [w1, h1] = slide.level_dimensions[1]
                w3 = int(w1 / 16)
                h3 = int(h1 / 16)
            # print("w3:", w3)
            slide.close()

            # OTSU_img adjustment
            unet_model_img_path = path  # "/data/lxy/MCO_thumb/MCO_thumb/" + midname + ".png"
            OTSU_img = cv.imread(unet_model_img_path, 0)
            OTSU_img = cv.resize(OTSU_img, (w3, h3))
            # print(OTSU_img)
            # print("MPP:", data[data["ParentSpecimen"] == midname]["MPP"].iloc[0])
            SVSmpp = float(data[data["fileName"] == midname]["MPP"].iloc[0])
            # print("SVSmpp:", SVSmpp)
            if np.isnan(SVSmpp):
                # print("NANMPP")
                continue

            wholePatch = unet_model_img_size
            # int(round(unet_model_img_size * 0.5 / SVSmpp))  # get the patch with 0.5 um/mpp
            thumbPatch = int(round(wholePatch * w3 / w0))  # patch length of thumb

            til_patch = -np.ones((int(h0 / wholePatch), int(w0 / wholePatch), 3), dtype=np.float32)
            classify = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            with concurrent.futures.ProcessPoolExecutor() as executor:

                rows = range(int(h0 / wholePatch))
                multiArg = [OTSU_img, slidingLength, w0, wholePatch, thumbPatch]
                matterLocation = executor.map(partial(IdenMatterLocation2, multiArg=multiArg), rows)

            roughLocation = []

            for i in matterLocation:
                roughLocation = roughLocation + i

            # print('roughLocation', len(roughLocation))
            pool = Pool(poolNum)

            # flag, count the length of roughLocation
            countFlag = 0
            full = False  # bool variable, if the shed is full, it is True; otherwise False.
            q = Manager().Queue(5000)

            while (True):

                multiArg = [roughLocation[countFlag], SVSmpp, w3, h3, w0,
                            path, SVSpath]  # roughLocation[i] i, j(row, column information)
                # print("multiArg:",multiArg)
                pool.apply_async(strip2Img, args=(multiArg, q,))
                if q.empty():
                    # tmp = 6 + 6
                    # print("It is empty!")
                    test_path = '/data/lar/lymphocyte/predex/im90.tif'
                    image = Image.open(test_path)
                    # pred+
                    pred2 = unet_model.detect_image(image)

                if not q.empty():
                    # print('It is not empty, shed length:', q.qsize())
                    location = q.get_nowait()  # get information from stack
                    length = len(location)
                    # print("length:", length)
                    if length > 0:

                        for flag in range(length):
                            # print("flag:", flag)
                            img = location[flag][0]
                            i = location[flag][1]
                            j = location[flag][2]
                            k = location[flag][3]
                            #print("debug:", img.shape)
                            # preprocessing

                            # change image to tensor
                            image = train_transformer(img)
                            img_tensor = image.reshape(1, 3, 256, 256) #result_image.shape[0] =256
                            # print("img_tensor_shape:", img_tensor.shape)
                            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
                            # classify network
                            pred_cl = net1(img_tensor)
                            prediction = torch.argmax(pred_cl, 1) # get the prediction label, prediction.item() is in 0~8
                            classify[prediction.item()] += 1
                            print('prediction', prediction.item())
                            print('classify', classify[prediction.item()])
                            til_patch[i, j * slidingLength + k, ] = colors[prediction.item()]


                countFlag += 1
                # print('countFlag', countFlag)
                if countFlag == len(roughLocation):
                    break

            while (True):

                if not q.empty():

                    location = q.get_nowait()

                    length = len(location)
                    # print('length', length)
                    if length >= 1:

                        for flag in range(length):
                            # print("flag:", flag)
                            img = location[flag][0]
                            i = location[flag][1]
                            j = location[flag][2]
                            k = location[flag][3]
                            #classify

                            image = train_transformer(img)
                            img_tensor = image.reshape(1, 3, 256, 256)
                            #print("img_tensor_shape:", img_tensor.shape)
                            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
                            # classify network
                            pred_cl = net1(img_tensor)
                            prediction = torch.argmax(pred_cl, 1)
                            classify[prediction.item()] += 1
                            print('prediction', prediction.item())
                            print('classify', classify[prediction.item()])
                            til_patch[i, j * slidingLength + k, ] = colors[prediction.item()]


                if q.empty():
                    break

            pool.close()
            pool.join()

            #img_size = (512, 512)
            img_type = ".png"
            np.save('/data/lar/CRC_data/prob_map/' + midname + 'tumor.npy', til_patch)
            whole = til_patch.astype(np.uint8)
            whole_img = Image.fromarray((whole).astype(np.uint8))
            whole_img.save('/data/lar/CRC_data/prob_map/' + midname + 'tumor.png')
            TILs = 1

            print('ADI:', classify[0])
            data.loc[data["fileName"] == midname, "ADI"] = classify[0]
            data.loc[data["fileName"] == midname, "BACK"] = classify[1]
            data.loc[data["fileName"] == midname, "DEB"] = classify[2]
            data.loc[data["fileName"] == midname, "LYM"] = classify[3]
            data.loc[data["fileName"] == midname, "MUC"] = classify[4]
            data.loc[data["fileName"] == midname, "MUS"] = classify[5]
            data.loc[data["fileName"] == midname, "NORM"] = classify[6]
            data.loc[data["fileName"] == midname, "STR"] = classify[7]
            data.loc[data["fileName"] == midname, "TUM"] = classify[8]

            data.to_csv(save_folder_path + 'crc_map.csv', index=False, sep=',')




