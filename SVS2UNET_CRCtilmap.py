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

#需要用到的模型
import utils
import torchvision.models as models
from unet1 import Unet
from model_NCRF import MODELS
from NormMacenko import normal_Macenko

from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # using gpu where location is 0
cpu_num = 40
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(30)

train_transformer = transforms.Compose([
                            transforms.ToPILImage(),
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
    poolNum = 20
    WSI_csv_path = "/data/lar/unet/U_net_map/crc_info.csv"
    SVSfolderPath = "/data/lar/CRC_data/gdc_linux/"
    thumbs_paths = "/data/lar/CRC_data/thumb_gray/"  # OTSU thumb paths
    img_save_type = ".png"
    umap_probability_save_folder_path = "/data/lar/unet/U_net_map/"
    save_folder_path = "/data/lar/unet/U_net_map/"    # final result
    error_report_path = "/data/lar/unet/U_net_map/error_report2.txt"

    HERef = np.load("/data/lar/program/normalization/HERef.npy")
    maxCRef = np.load("/data/lar/program/normalization/maxCRef.npy")

    if not os.path.exists(umap_probability_save_folder_path):
        os.mkdir(umap_probability_save_folder_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU

    #加载肿瘤分割网络NCRF
    resnet18_model = models.resnet18(pretrained=True)
    net1 = resnet18_model
    num_fits = resnet18_model.fc.in_features
    net1.fc = nn.Linear(num_fits, 9)
    net1.load_state_dict(
        torch.load('/data/lar/program/TIL_task/best_ResNetBCEWithLogitsLoss_crc.pth', map_location=device))
    net1.to(device=device)
    net1.eval()
    #til分割网络
    unet_model = Unet()
    # load model parameters
    # evaluation mode
    mode = "predict"
    print("Load model done")
    # 测试整个流程运行
    img = cv2.imread('/data/lar/lymphocyte/data_color_normal/im88.tif')

    # open csv
    data = pd.read_csv(WSI_csv_path)

    paths = [data['png_paths'].iloc[i] for i in range(len(data['png_paths']))]

    for index, path in enumerate(paths):
        if index <= 5:
            continue
        TILs = 0
        pred_flag = 0
        til = 0
        #path = '/data/lar/CRC_data/thumb_gray/TCGA-CA-5255-01Z-00-DX1.77310ae2-9c5f-48c4-9754-c5b30d287089.png'
        while TILs == 0:
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
            # some slides may don't have dimension 3shou
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

            wholePatch = unet_model_img_size  # int(round(unet_model_img_size * 0.5 / SVSmpp))  # get the patch with 0.5 um/mpp
            thumbPatch = int(round(wholePatch * w3 / w0))  # patch length of thumb

            unet_mat = -np.ones((int(h0 / wholePatch), int(w0 / wholePatch), 256, 256), dtype=np.float32)
            til_patch = -np.ones((int(h0 / wholePatch), int(w0 / wholePatch)), dtype=np.float32)
            # xception_probability_mat = np.zeros((int(h0 / wholePatch), int(w0 / wholePatch), 9), dtype=np.float32)
            classify = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            with concurrent.futures.ProcessPoolExecutor() as executor:

                rows = range(int(h0 / wholePatch))
                multiArg = [OTSU_img, slidingLength, w0, wholePatch, thumbPatch]
                matterLocation = executor.map(partial(IdenMatterLocation2, multiArg=multiArg), rows)

            roughLocation = []

            for i in matterLocation:
                roughLocation = roughLocation + i

            # print(roughLocation)
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
                    # 预测
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
                            # print("debug:", img.shape)
                            # preprocessing

                            #用来判断是肿瘤区域
                            image = train_transformer(img)
                            img_tensor = image.reshape(1, 3, img.shape[0], img.shape[1])
                            # print("img_tensor_shape:", img_tensor.shape)
                            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
                            pred_cl = net1(img_tensor)
                            prediction = torch.argmax(pred_cl, 1)
                            classify[prediction.item()] += 1
                            if prediction.item() == 3 or prediction.item() == 7 or prediction.item() == 8:
                                #是肿瘤区域再进行后续计算
                                # extract result
                                img = Image.fromarray(img)
                                #print('img.shape', img.shape)
                                pred = unet_model.detect_image(img)
                                pred = pred.convert('L')
                                transformer2 = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
                                result = transformer2(pred)[0]
                                unet_mat[i, j * slidingLength + k, :, :] = np.asarray(result)
                                # probability to image
                                result[result > 0] = 255
                                result[result < 0] = 0
                                # probability to image
                                # print("pred.shape:", pred.shape)
                                til_patch[i, j * slidingLength + k] = torch.sum(result == 255) / 65336

                                # if not np.sum(unet_mat[i, j * slidingLength + k, :, :] == -1) == 0:
                                #    ertxt.write(midname + '\n')
                                #    ertxt.write("model not run on this patch\n")
                                if torch.sum(result > 0.5) > 1:
                                    pred_flag = 1


                countFlag += 1

                if countFlag == len(roughLocation):
                    break

            while (True):

                if not q.empty():

                    location = q.get_nowait()

                    length = len(location)

                    if length >= 1:

                        for flag in range(length):
                            # print("flag:", flag)
                            img = location[flag][0]
                            i = location[flag][1]
                            j = location[flag][2]
                            k = location[flag][3]

                            #用来判断是肿瘤区域
                            image = train_transformer(img)
                            img_tensor = image.reshape(1, 3, img.shape[0], img.shape[1])
                            # print("img_tensor_shape:", img_tensor.shape)
                            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
                            pred_cl = net1(img_tensor)
                            prediction = torch.argmax(pred_cl, 1)
                            classify[prediction.item()] += 1
                            if prediction.item() == 3 or prediction.item() == 7 or prediction.item() == 8:
                                #是肿瘤区域再进行后续计算

                                img = Image.fromarray(img)
                                pred = unet_model.detect_image(img)
                                pred = pred.convert('L')
                                transformer2 = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
                                result = transformer2(pred)[0]
                                unet_mat[i, j * slidingLength + k, :, :] = np.asarray(result)
                                # probability to image
                                result[result > 0] = 255
                                result[result < 0] = 0
                                # probability to image
                                til_patch[i, j * slidingLength + k] = torch.sum(result == 255) / 65336
                                #unet_mat[i, j * slidingLength + k, :, :] = result
                                if torch.sum(result > 0.5) > 1:
                                    pred_flag = 1


                if q.empty():
                    break

            pool.close()
            pool.join()


            # get TILscore(use mean)

            unet_map = np.zeros((unet_mat.shape[0]*256, unet_mat.shape[1]*256), dtype=np.float32)
            for i in range(unet_mat.shape[0]):
                for j in range(unet_mat.shape[1]):
                    unet_map[i * 256: (i+1) * 256, j * 256: (j+1) * 256] = unet_mat[i, j, :, :]
            map1 = til_patch
            map1[map1 < 0] = 0
            til_patch_v2 = til_patch[til_patch > 0]
            til = np.sum(til_patch[til_patch > 0])
            TILs = np.mean(til_patch_v2)
            print('TILs:', TILs)
            data.loc[data["fileName"] == midname, "TIL"] = TILs
            np.save('/data/lar/CRC_data/prob_map/' + midname + 'til.npy', til_patch)
            np.save('/data/lar/CRC_data/prob_map/' + midname + 'umap.npy', unet_map)
            norm_map = np.sqrt(map1/np.max(map1)) * 255
            norm_map = np.asarray(norm_map, dtype=np.uint8)
            heat_img = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            cv2.imwrite('/data/lar/CRC_data/prob_map/'+midname+'til.png', heat_img)
            #aa = np.load('/data/lar/BRCA/BRCAheatmap/TCGA-B6-A0I1-01Z-00-DX1.86BEC5E4-A2D1-4039-8D08-0598FA8BCC2Bmap.npy')
            #bb = np.load('/data/lar/BRCA/BRCAheatmap/TCGA-B6-A0I1-01Z-00-DX1.86BEC5E4-A2D1-4039-8D08-0598FA8BCC2B.npy')


            data.to_csv(save_folder_path + 'crc_tilmap.csv', index=False, sep=',')



