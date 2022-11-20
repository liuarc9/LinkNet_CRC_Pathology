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
import cv2
import torchvision.transforms as transforms
from PIL import Image
from IdenMatterLocation import *
from ImgDetail_unet import *

#models
import utils
import torchvision.models as models
from unet import Unet
from nets.linknet import LinkNet, MSLinkNetBase
from model_NCRF import MODELS
from NormMacenko import normal_Macenko

from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # using gpu where location is 0

train_transformer = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                        ])

"""

"""

if __name__ == '__main__':

    slidingLength = 10
    unet_model_img_size = 256
    poolNum = 50
    WSI_csv_path = "/"
    img_save_type = ".png"
    umap_probability_save_folder_path = "/"
    save_folder_path = "/"    # final result

    HERef = np.load("/")
    maxCRef = np.load("/")

    if not os.path.exists(umap_probability_save_folder_path):
        os.mkdir(umap_probability_save_folder_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU

    #NCRF
    resnet18_model = models.resnet18(pretrained=True)
    net1 = resnet18_model
    num_fits = resnet18_model.fc.in_features
    net1.fc = nn.Linear(num_fits, 9)
    net1.load_state_dict(
        torch.load('/', map_location=device))
    net1.to(device=device)
    net1.eval()
    #til
    unet_model = Unet()
    # load model parameters
    # evaluation mode
    mode = "predict"
    print("Load model done")

    # open csv
    data = pd.read_csv(WSI_csv_path)

    paths = [data['png_paths'].iloc[i] for i in range(len(data['png_paths']))]

    for index, path in enumerate(paths):
        midname = path[path.rindex("/") + 1:path.rindex(".")]
        if pd.isna(data.loc[data["ParentSpecimen"] == midname, "TIL"]).item() == False:
            print(midname)
            continue
        TILs = 0
        pred_flag = 0
        til = 0
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

            SVSpath = data.loc[index]['svs_paths']

            slide = openslide.open_slide(SVSpath)

            [w0, h0] = slide.level_dimensions[0]
            try:

                [w3, h3] = slide.level_dimensions[3]

            except IndexError:

                [w1, h1] = slide.level_dimensions[1]
                w3 = int(w1 / 16)
                h3 = int(h1 / 16)
            slide.close()

            # OTSU_img adjustment
            unet_model_img_path = path
            OTSU_img = cv.imread(unet_model_img_path, 0)
            OTSU_img = cv.resize(OTSU_img, (w3, h3))
            SVSmpp = float(data[data["ParentSpecimen"] == midname]["MPP"].iloc[0])
            if np.isnan(SVSmpp):
                continue

            wholePatch = unet_model_img_size  # int(round(unet_model_img_size * 0.5 / SVSmpp))  # get the patch with 0.5 um/mpp
            thumbPatch = int(round(wholePatch * w3 / w0))  # patch length of thumb

            til_patch = -np.ones((int(h0 / wholePatch), int(w0 / wholePatch)), dtype=np.float32)
            til_patch8 = -np.ones((int(h0 / wholePatch), int(w0 / wholePatch)), dtype=np.float32)
            classify = [0, 0, 0, 0, 0, 0, 0, 0, 0]

            with concurrent.futures.ProcessPoolExecutor() as executor:

                rows = range(int(h0 / wholePatch))
                multiArg = [OTSU_img, slidingLength, w0, wholePatch, thumbPatch]
                matterLocation = executor.map(partial(IdenMatterLocation2, multiArg=multiArg), rows)

            roughLocation = []

            for i in matterLocation:
                roughLocation = roughLocation + i

            pool = Pool(poolNum)

            # flag, count the length of roughLocation
            countFlag = 0
            full = False  # bool variable, if the shed is full, it is True; otherwise False.
            q = Manager().Queue(5000)

            while (True):

                multiArg = [roughLocation[countFlag], SVSmpp, w3, h3, w0,
                            path, SVSpath]  # roughLocation[i] i, j(row, column information)
                pool.apply_async(strip2Img, args=(multiArg, q,))
                if q.empty():
                    test_path = '/'
                    image = Image.open(test_path)
                    # 预测
                    pred2 = unet_model.detect_image(image)

                if not q.empty():
                    location = q.get_nowait()  # get information from stack
                    length = len(location)
                    # print("length:", length)
                    if length > 0:

                        for flag in range(length):
                            img = location[flag][0]
                            i = location[flag][1]
                            j = location[flag][2]
                            k = location[flag][3]
                            # preprocessing
                            # color normalization
                            source_image = img
                            try:
                                result_image = normal_Macenko(source_image, HERef, maxCRef)
                            except (np.linalg.LinAlgError, ValueError):
                                result_image = img
                                print('numpy.linalg.LinAlgError: Eigenvalues did not converge')

                            # extract result
                            img = Image.fromarray(result_image)

                            image = train_transformer(result_image)
                            img_tensor = image.reshape(1, 3, result_image.shape[0], result_image.shape[1])
                            # print("img_tensor_shape:", img_tensor.shape)
                            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
                            pred_cl = net1(img_tensor)
                            prediction = torch.argmax(pred_cl, 1)
                            classify[prediction.item()] += 1
                            print('prediction', prediction.item())
                            print('classify', classify[prediction.item()])

                            if prediction.item() == 8:
                                pred = unet_model.detect_image(img)
                                pred = pred.convert('L')
                                transformer2 = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
                                result = transformer2(pred)[0]
                                # probability to image
                                result[result > 0] = 255
                                result[result < 0] = 0
                                # print("pred.shape:", pred.shape)
                                til_patch8[i, j * slidingLength + k] = torch.sum(result == 255) / 65336



                countFlag += 1
                if countFlag == len(roughLocation):
                    break

            while (True):

                if not q.empty():

                    location = q.get_nowait()

                    length = len(location)
                    if length >= 1:

                        for flag in range(length):
                            img = location[flag][0]
                            i = location[flag][1]
                            j = location[flag][2]
                            k = location[flag][3]
                            # preprocessing
                            # color normalization
                            source_image = img
                            try:
                                result_image = normal_Macenko(source_image, HERef, maxCRef)
                            except (np.linalg.LinAlgError, ValueError):
                                result_image = img
                            # extract result
                            img = Image.fromarray(result_image)

                            image = train_transformer(result_image)
                            img_tensor = image.reshape(1, 3, result_image.shape[0], result_image.shape[1])
                            #print("img_tensor_shape:", img_tensor.shape)
                            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
                            pred_cl = net1(img_tensor)
                            prediction = torch.argmax(pred_cl, 1)
                            classify[prediction.item()] += 1
                            print('prediction', prediction.item())
                            print('classify', classify[prediction.item()])
                            if prediction.item() == 8:
                                pred = unet_model.detect_image(img)
                                pred = pred.convert('L')
                                transformer2 = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
                                result = transformer2(pred)[0]
                                # probability to image
                                result[result > 0] = 255
                                result[result < 0] = 0
                                til_patch8[i, j * slidingLength + k] = torch.sum(result == 255) / 65336


                if q.empty():
                    break

            pool.close()
            pool.join()


            # get TILscore(use mean)
            til_patch_8 = til_patch8[til_patch8 > 0]
            TILs8 = np.mean(til_patch_8)
            TILs = TILs8
            print('TILs:', TILs)
            print('TUM:', classify[8])
            data.loc[data["ParentSpecimen"] == midname, "TIL"] = TILs8
            data.to_csv(save_folder_path + '/', index=False, sep=',')

    ertxt.close()