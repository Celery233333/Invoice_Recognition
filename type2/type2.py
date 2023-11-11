import cv2
import os
import numpy as np
from imutils.perspective import four_point_transform
import paddlehub as hub
import xlwt
from paddleocr import PaddleOCR,draw_ocr


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
        self.image = [cv2.resize(file,(int(w*1.2),int(h*1.2)))] 
        
        
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
    horizontalStructure1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure1)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours = [] 
    for contour in contours: 
        area = cv2.contourArea(contour)
        if area >= 1000:
            cv_contours.append(contour)
    
    qc1 = 0
    qc2 = 0
    qc3 = 0
    qc4 = 0
    temp = 0
    for i in range(len(cv_contours[0])):
        if temp < cv_contours[0][i][0][0]:
            temp = cv_contours[0][i][0][0]
            qc1 = i

    temp = 10000
    for i in range(len(cv_contours[0])):
        if temp > cv_contours[0][i][0][0]:
            temp = cv_contours[0][i][0][0]
            qc2 = i

    temp = 0
    for i in range(len(cv_contours[1])):
        if temp < cv_contours[1][i][0][0]:
            temp = cv_contours[1][i][0][0]
            qc3 = i

    temp = 10000
    for i in range(len(cv_contours[1])):
        if temp > cv_contours[1][i][0][0]:
            temp = cv_contours[1][i][0][0]
            qc4 = i

    qc1 = [cv_contours[0][qc1][0][0],cv_contours[0][qc1][0][1]]
    qc2 = [cv_contours[0][qc2][0][0],cv_contours[0][qc2][0][1]]
    qc3 = [cv_contours[1][qc3][0][0],cv_contours[1][qc3][0][1]]
    qc4 = [cv_contours[1][qc4][0][0],cv_contours[1][qc4][0][1]]
    qc = [0,0,0,0]
    qc[0] = qc4
    qc[1] = qc2
    qc[2] = qc3
    qc[3] = qc1
    # print(qc)

    warped = four_point_transform(src_img0, np.array(qc))
    # cv2.imshow("a",warped)
    cv2.imwrite("text.png",warped)
    # cv2.waitKey()

    return warped

def get_Affine_Location(input_Path,Net_img,contours):
    src_img = cv2.imread(input_Path)
    witdh = src_img.shape[0]
    height = src_img.shape[1]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i in range(len(contours)):
        area0 = cv2.contourArea(contours[i])
        if area0<20:continue

        # =======================查找每个表的关节数====================
        epsilon = 0.1 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)  # 获取近似轮廓
        x1, y1, w1, h1 = cv2.boundingRect(approx)
        roi = Net_img[int(y1):int(y1+h1) ,int(x1):int(x1+w1)]
        roi_contours, hierarchy = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        print('len(roi_contours):',len(roi_contours))
        if len(roi_contours)<4:continue

        src_img1 = cv2.rectangle(src_img, (x1, y1),(x1+w1,y1+h1), (255,255,255), -1)
        cut_img = src_img[y1:y1+h1,x1:x1+w1]
        cut_img1 = src_img[0:y1,x1:x1+w1]
        cut_img2 = src_img[y1+h1:height,x1:x1+w1]

    #     cv2.imshow('src_img_'+str(i),src_img1)
        cv2.imwrite("bg1.png",cut_img1)
        cv2.imwrite("bg2.png",cut_img2)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
    # cv2.imshow("out",out)
    cv2.imwrite("ocr.png", out)
    # cv2.waitKey()


if __name__ == '__main__':
    preProcess("type2.png")
    input_Path = 'ocr.png'
    cutImg_path = 'C:/Users/Celery/Desktop/'
    cutImg_name = input_Path.split('/')[-1][:-4]
    img = FindContours(input_Path)
    img = cv2.bitwise_not(img)
    # ocr = OCRModel('text.png')
    # result = ocr.get_result()
    # print(result)


    contours = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,3))
    dilated = cv2.dilate(img, kernel, iterations = 1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    contours = contours[1:]
    contours.pop()
    contours_iter = iter(contours)
    
    k=0
    j=0
    wb = xlwt.Workbook()
    ws = wb.add_sheet("qc")
    test = []
    #Loop to draw rectangle around contour, crop, pass to tesseract and save the text
    for i in contours_iter: 
        #Separates the contours into coordinates x,y,start,end
        x ,y ,width ,height = cv2.boundingRect(i)
        test.append([x,y,width,height])
        j=j+1

        if j == 9:
            test = np.array(test)
            test = test[np.lexsort(test[:,::-1].T)]
            for z in range(9):
                x = test[z][0]
                y = test[z][1]
                width = test[z][2]
                height = test[z][3]
                #Draws rectangles around the contours
                img = cv2.rectangle(img, (x,y) ,(x+width, y+height) ,(255,0,0) , 2)
                #Crops the text outlined by the rectangle
                cropped = img[y: y+height, x: x+width]
                cropped = cv2.bitwise_not(cropped)
                # cv2.imshow("temp",cropped)
                # cv2.waitKey()
                cropped = cv2.resize(cropped,(int(cropped.shape[1]*1),int(cropped.shape[0]*1)))
                cv2.imwrite("temp.png",cropped)
                ocr = OCRModel('temp.png')
                result = ocr.get_result()
                print(result)
                if len(result) == 0:
                    ocr1 = PaddleOCR(use_angle_cls = True,lang = "ch")
                    result = ocr1.ocr("temp.png",cls=True)
                    try:
                        print(result[0][1][0])
                        ws.write(k,z,result[0][1][0])
                    except:
                        ws.write(k,z,"")
                else:
                    ws.write(k,z,"".join(result))
                # result = ocr.ocr("temp.png",cls=True)
                # try:
                #     print(result[0][1][0])
                #     ws.write(k,z,result[0][1][0])
                # except:
                #     ws.write(k,z,"")
            test = []
            j=0
            k=k+1

    wb.save("text1.xls")
    img = cv2.bitwise_not(img)
    cv2.imwrite("a.png",img)
        