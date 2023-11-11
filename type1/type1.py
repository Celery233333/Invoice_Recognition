import cv2
import os
import numpy as np
from imutils.perspective import four_point_transform
import paddlehub as hub
import xlwt
from paddleocr import PaddleOCR,draw_ocr


class OCRModel:
    def __init__(self, file_name ):
        self.file_name = file_name
        self.ocr = hub.Module(name="chinese_ocr_db_crnn_server")
        file = cv2.imread(self.file_name)
        w = file.shape[1]
        h = file.shape[0]
        self.image = [cv2.resize(file,(int(w*1.7),int(h*1.7)))] 
        
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

    stats = stats[stats[:,1].argsort()]
    stats=np.delete(stats,0,axis=0)
    stats=np.delete(stats,0,axis=0)
    temp1 = stats[0:4]
    temp2 = stats[4:12]
    temp3 = stats[12:14]
    temp4 = stats[14:]

    temp1 = temp1[np.lexsort(temp1[:,::-1].T)]
    temp2 = temp2[np.lexsort(temp2[:,::-1].T)]
    temp3 = temp3[np.lexsort(temp3[:,::-1].T)]
    temp4 = temp4[np.lexsort(temp4[:,::-1].T)]
    
    stats[0:4] = temp1
    stats[4:12] = temp2
    stats[12:14] = temp3
    stats[14:] = temp4
    output = np.zeros((src_img.shape[0], src_img.shape[1], 3), np.uint8)
    for i in range(1, ret):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 255)
        output[:, :, 1][mask] = np.random.randint(0, 255)
        output[:, :, 2][mask] = np.random.randint(0, 255)

    

    return stats,labels,Net_img,contours

def get_Affine_Location(input_Path,Net_img,contours):
    src_img = cv2.imread(input_Path)
    witdh = src_img.shape[0]
    height = src_img.shape[1]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i in range(len(contours)):
        area0 = cv2.contourArea(contours[i])
        if area0<20:continue

        epsilon = 0.1 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)  
        x1, y1, w1, h1 = cv2.boundingRect(approx)
        roi = Net_img[int(y1):int(y1+h1) ,int(x1):int(x1+w1)]
        roi_contours, hierarchy = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        print('len(roi_contours):',len(roi_contours))
        if len(roi_contours)<4:continue

        src_img1 = cv2.rectangle(src_img, (x1, y1),(x1+w1,y1+h1), (255,255,255), -1)
        cut_img = src_img[y1:y1+h1,x1:x1+w1]
        cut_img1 = src_img[0:y1,x1:x1+w1]
        cut_img2 = src_img[y1+h1:height,x1:x1+w1]


        cv2.imwrite("bg1.png",cut_img1)
        cv2.imwrite("bg2.png",cut_img2)
    cv2.destroyAllWindows()

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
    preProcess("type1.png")
    input_Path = 'ocr.png'
    cutImg_path = 'C:/Users/Celery/Desktop/'
    cutImg_name = input_Path.split('/')[-1][:-4]
    stats,labels,Net_img,contours = FindContours(input_Path)
    get_Affine_Location(input_Path,Net_img,contours)
    wb = xlwt.Workbook()
    ws = wb.add_sheet("qc")
    ws.write(0,0,"购买方")
    ws.write(1,0,"密码区")
    ws.write(2,0,"货物或应税劳务, 服务名称")
    ws.write(3,0,"规格型号")
    ws.write(4,0,"单位")
    ws.write(5,0,"数量")
    ws.write(6,0,"单价")
    ws.write(7,0,"金额")
    ws.write(8,0,"税率")
    ws.write(9,0,"税额")
    ws.write(10,0,"价税合计（大写）")
    ws.write(11,0,"销售方")
    ws.write(12,0,"备注")
    img = cv2.imread("ocr.png")
    # ocr = OCRModel('bg1.png')
    # result = ocr.get_result()
    # print(result)
    # ocr = OCRModel('bg2.png')
    # result = ocr.get_result()
    # print(result)
    ocr = PaddleOCR(use_angle_cls = True,lang = "ch")
    # result = ocr.ocr("bg1.png",cls=True)
    # print(result)
    # result = ocr.ocr("bg2.png",cls=True)
    # print(result)
    for i in range(len(stats)):
        cropped = img[stats[i][1]-5: stats[i][1]+stats[i][3]+5, stats[i][0]-5: stats[i][0]+stats[i][2]+5]
        cv2.imwrite("temp.png",cropped)
        ocr = OCRModel('temp.png')
        result = ocr.get_result()
        print(result)
        if i == 1:
            ws.write(0,1," ".join(result))
        if i == 3:
            ws.write(1,1,"".join(result))
        if i == 4:
            result = result[1:]
            ws.write(2,1,"/".join(result))
        if i == 4:
            result = result[1:]
            ws.write(3,1,"/".join(result))   
        if i == 7:
            result = result[1:]
            ws.write(5,1,"/".join(result))
        if i == 8:
            result = result[1:]
            ws.write(6,1,"/".join(result))
        if i == 9:
            result = result[1:3]
            ws.write(7,1,"/".join(result))
        if i == 10:
            result = result[1:]
            ws.write(8,1,"/".join(result))
        if i == 11:
            result = result[1:]
            ws.write(9,1,"/".join(result))
        if i == 13:
            ws.write(10,1,result[0])
        if i == 15:
            # ws.write(11,1,"名称: "+result[2]+";\n"+"纳税人识别号: "+result[4]+";\n"+"地址，电话: "+result[6]+";\n"+"开户行及账号: "+result[8])
            ws.write(11,1," ".join(result))
        if i == 17:
            ws.write(12,1," ".join(result))
        # cv2.imshow("a",cropped)
        # cv2.waitKey()
    ocr = OCRModel('bg1.png')
    result = ocr.get_result()
    ws.write(13,0," ".join(result))
    print(result)

    ocr = OCRModel('bg2.png')
    result = ocr.get_result()
    ws.write(14,0," ".join(result))
    print(result)
    wb.save("text.xls")
