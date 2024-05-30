import cv2
import numpy as np
import azure.ai.vision as cvsdk

def area(row, col):
    global nn
    if bg[row][col] != 255:
        return
    bg[row][col] = lifearea
    if col > 1:#左
        if bg[row][col-1] == 255:
            nn += 1
            area(row, col-1)
    
    if col < w-1:#右
        if bg[row][col+1] == 255:
            nn += 1
            area(row, col+1)

    if row > 1:#上
        if bg[row-1][col] == 255:
            nn += 1
            area(row-1, col)

    if row < h-1:#下
        if bg[row+1][col] == 255:
            nn += 1
            area(row+1, col)

fileName = "realPlate/resizejpg006.bmp"
image = cv2.imread(fileName)
#車牌偵測
plateCascade =cv2.CascadeClassifier("myPlateDetector.xml")
plates = plateCascade.detectMultiScale(image)

for (x,y,w,h) in plates:
    cropPlate = image[y:y+h,x:x+w]
    cropPlate = cv2.resize(cropPlate,(280,80))
    fName =fileName.split(".")[0]+"_plate.jpg"
    cv2.imwrite(fName,cropPlate)
    
#車牌灰階處理，二值化(黑白)處理
grayImg = cv2.cvtColor(cropPlate, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY_INV)

#除去雜點
for i in range(len(thresh)):
    for j in range(len(thresh[i])):
        if thresh[i][j] == 255:
            count = 0
            for k in range(-2,3):
                for l in range(-2,3):
                    try:
                        if thresh[i+k][j+l] == 255:
                            count += 1
                    except IndexError:
                        pass
            if count <= 6:
                thresh[i][j] = 0


#找輪廓外框
contours1 = cv2.findContours(thresh,
                              cv2.RETR_EXTERNAL,                                    
                              cv2.CHAIN_APPROX_SIMPLE)


contours = contours1[0]

#準備一個紀錄所有外框的清單List，這個內容會包含：文字、雜點、分割符號、崎零地(左右部分不要)
letter_image_regions = []
for contour in contours:
     (x,y,w,h)=cv2.boundingRect(contour)
     letter_image_regions.append((x,y,w,h))
     
#將所有外框依照x軸位置排序(由左至右排序)
letter_image_regions=sorted(letter_image_regions,
                             key=lambda x:x[0])

print(letter_image_regions)
print(len(letter_image_regions))

#for(x,y,w,h) in letter_image_regions:
 #   cv2.rectangle(thresh,(x,y),(x+w,y+h),(255,255,255),1)

#cv2.imshow("thresh",thresh)

count = 0
for box in letter_image_regions:
    (x,y,w,h)=box
    if(x >= 5 and x <= 240) and (w >= 5 and w <=32) and (h >= 60 and h < 70):
        #cv2.rectangle(thresh, (x,y),(x+w,y+h),(255,255,255),1)
        count += 1
        #print(box)
#cv2.imshow("thresh2",thresh)   
#print(count)

if count <= 5:
    wmax = 35
else:
    wmax = 32
    
nChar = 0
#裝文字輪廓的清單
letterlist = []
for box in letter_image_regions:
    (x,y,w,h)=box
    if(x >= 5 and x <= 240) and (w >= 5 and w <=wmax) and (h >= 60 and h < 70):
        nChar += 1
        letterlist.append(box)
        print(box)


#print(letterlist)
#print(nChar)
#cv2.imshow("thresh3", thresh)

real_Shape=[]
for i, box in enumerate(letterlist):
    (x,y,w,h) = box
    bg = thresh[y:y+h,x:x+w]
    #處理畸零地
    if i == 0 or i == nChar:
        lifearea = 0
        nn =0
        life = []
        for row in range(0,h):
            for col in range(0,w):
                if bg[row][col] == 255:
                    nn = 1
                    lifearea += 1
                    area(row,col)
                    life.append(nn)
                    
        maxlife = max(life)
        indexmaxlife = life.index(maxlife)
        
        for row in range(0,h):
            for col in range(0,w):
                if bg[row][col] == indexmaxlife+1:
                    bg[row][col] = 255
                else:
                    bg[row][col] = 0
                    
    real_Shape.append(bg)
    
#print(real_Shape)
#print(len(real_Shape))

newH, newW =thresh.shape
space = 8
offset = 2

bg = np.zeros((newH + space*2, newW + space*2 + (nChar-1) * offset, 1), np.uint8)
bg.fill(0)

for i, letter in enumerate(real_Shape):
    h = letter.shape[0]
    w = letter.shape[1]
    x = letterlist[i][0]
    y = letterlist[i][1]
    for row in range(h):
        for col in range(w):
            bg[space+y+row][space+x+col+i*offset] = letter[row][col]
#cv2.imshow("new bg", bg) 
   
_, bg =cv2.threshold(bg, 127, 255, cv2.THRESH_BINARY_INV)
(bg_h,bg_w)=bg.shape
times=3
bg=cv2.resize(bg,(bg_w*times,bg_h*times))
cv2.imshow("plate2",bg)


ocr_enpoint = "https://wusc-computer-vision.cognitiveservices.azure.com/"
ocr_key="e57e8cebdb124ca9a1f78b4891fc1d73"

service_options =cvsdk.VisionServiceOptions(ocr_enpoint,ocr_key)

#抓到的檔案
#imgFile ="realPlate/resizejpg001.bmp"
vision_source=cvsdk.VisionSource(filename=fileName)

analysis_options=cvsdk.ImageAnalysisOptions()
analysis_options.features=cvsdk.ImageAnalysisFeature.TEXT

image_analyzer=cvsdk.ImageAnalyzer(
    service_options,
    vision_source,
    analysis_options)

result = image_analyzer.analyze()

if(result.reason == cvsdk.ImageAnalysisResultReason.ANALYZED):
    if(result.text is not None):
        print("\n\nPlate is :\n")
        for line in result.text.lines:
            print(line.content)
    
        
cv2.waitKey()
cv2.destroyAllWindows()
   




