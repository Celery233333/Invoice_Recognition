import cv2
import os
import numpy as np
from imutils.perspective import four_point_transform
import paddlehub as hub
import xlwt
from paddleocr import PaddleOCR,draw_ocr
from tkinter import W
import imutils
from PIL import Image
import matplotlib.pyplot as plt


class OCRModel:
    def __init__(self, file_name ):
        '''
        file is an image 
        '''
        self.file_name = file_name
        self.ocr = hub.Module(name="chinese_ocr_db_crnn_server")
        file = cv2.imread(self.file_name)
        w = file.shape[1]
        h = file.shape[0]
        self.image = [cv2.resize(file,(int(w*1.3),int(h*1.3)))] 
        
    def get_result(self):
        results = self.ocr.recognize_text(
                    images = self.image,         
                    use_gpu = False,            
                    box_thresh = 0.8,             
                    text_thresh = 0.8)         
                
        text = []
        for result in results:
            data = result['data']
            for infomation in data:
                text.append(infomation['text'])            
        return text

def FindContours(img_path):
    src_img = cv2.imread(img_path)
    src_img0 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    src_img0 = cv2.GaussianBlur(src_img0,(5,5),0)
    src_img1 = cv2.bitwise_not(src_img0)
    AdaptiveThreshold = cv2.adaptiveThreshold(src_img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()
    scale = 30

    horizontalSize = int(horizontal.shape[1]/scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # cv2.imshow("horizontal", horizontal)
    # cv2.waitKey(0)

    verticalsize = int(vertical.shape[1]/scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    # cv2.imshow("verticalsize", vertical)
    # cv2.waitKey(0)

    mask = horizontal + vertical
    #np.count_nonzero(mask, axis=0) #对列统计白色（非零值）
    #np.count_nonzero(mask, axis=1) #对行统计白色（非零值）
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    Net_img = cv2.bitwise_and(horizontal, vertical)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Net_img", Net_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    final_kernel=np.ones((13,13), np.uint8)
    img_bin_final=cv2.dilate(mask,final_kernel,iterations=1)
    ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    stats=np.delete(stats,0,axis=0)
    stats=np.delete(stats,0,axis=0)



    output = np.zeros((src_img.shape[0], src_img.shape[1], 3), np.uint8)
    for i in range(1, ret):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 255)
        output[:, :, 1][mask] = np.random.randint(0, 255)
        output[:, :, 2][mask] = np.random.randint(0, 255)

    
    # cv2.imshow('oginal', output)
    cv2.imwrite("original.jpg",output)
    # cv2.waitKey()
    return stats,labels,Net_img,contours

def preProcess(path):
    #loads in the image
    image = cv2.imread(path)
    b, g, r = cv2.split(image)
    r = cv2.adaptiveThreshold(r,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,21)


    #Resizes and duplicates to compare
    originImg = r
    editImg = r

    #Sets window size to apply the operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    #Inverts the bits (black->white, white->black)
    notThresh = cv2.bitwise_not(r)

    #Applies dilation to image for contour preparation
    dilated = cv2.dilate(notThresh, kernel, iterations = 8)

    ret, img = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)

    img_line = np.zeros_like(img)

    cv2.polylines(img_line, contours, isClosed=True, color=(255,255,255), thickness=2)  
    # Image.fromarray(img_line).show()


    cv_contours = [] 
    for contour in contours: 
        area = cv2.contourArea(contour)
        if area >= 1000:
            cv_contours.append(contour)

    img_poly = np.zeros_like(img)
    cv2.fillPoly(img_poly, cv_contours, (255,255,255))
    # Image.fromarray(img_poly).show()

    mask = cv2.bitwise_not(img_poly)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    img1 = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,11)
    # binary = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,9)
    out = cv2.bitwise_or(binary, mask)
    cv2.imwrite("ocr.png", out)


if __name__ == '__main__':
    preProcess("type3.png")
    input_Path = 'ocr.png'
    cutImg_path = 'C:/Users/Celery/Desktop/'
    cutImg_name = input_Path.split('/')[-1][:-4]
    stats,labels,Net_img,contours = FindContours(input_Path)
    wb = xlwt.Workbook()
    ws = wb.add_sheet("qc")
    img = cv2.imread("ocr.png")
    ocr = PaddleOCR(use_angle_cls = True,lang = "ch")

    k=0
    j=0
    wb = xlwt.Workbook()
    ws = wb.add_sheet("qc")
    ocr = PaddleOCR(use_angle_cls = True,lang = "ch")
    test = []
    #Loop to draw rectangle around contour, crop, pass to tesseract and save the text
    for i in range(len(stats)): 
        #Separates the contours into coordinates x,y,start,end
        x1 = stats[i][1]-5
        x2 = stats[i][1]+stats[i][3]+5
        y1 = stats[i][0]-5
        y2 = stats[i][0]+stats[i][2]+5
        test.append([y1,y2,x1,x2])
        j=j+1

        if j == 8:
            test = np.array(test)
            test = test[np.lexsort(test[:,::-1].T)]
            for z in range(8):
                cropped = img[test[z][2]: test[z][3], test[z][0]: test[z][1]]
                # cv2.imshow("temp",cropped)
                # cv2.waitKey()
                cv2.imwrite("temp.png",cropped)
                ocr1 = OCRModel('temp.png')
                result = ocr1.get_result()
                print(result)
                if len(result) == 0:
                    result = ocr.ocr("temp.png",cls=True)
                    try:
                        print(result[0][1][0])
                        ws.write(k,z,result[0][1][0])
                    except:
                        ws.write(k,z,"")
                else:
                    ws.write(k,z,"".join(result))
            test = []
            j=0
            k=k+1
    wb.save("text2.xls")