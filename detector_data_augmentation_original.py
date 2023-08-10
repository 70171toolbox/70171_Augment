#detector的圖片增強
#重點:augmentation
#詳細使用方法參見if __name__ == "__main__"

import cv2
import numpy as np
import numpy.random as random


class augmentation():
    def __init__(self,imgsize,mean,std):
        self.imgsize = imgsize
        self.mean = mean
        self.std = std
        self.augmentation = compose([
            convertfromints(),
            randomcolor(),
            randomsample(),
            randommirror(),
            resize(self.imgsize),
            normailize(self.mean,self.std)
        ])
    def __call__(self,img,bboxes,labels):
        img,bboxes,labels = self.augmentation(img,bboxes,labels)
        return img,bboxes,labels

class compose():
    def __init__(self,transformers):
        self.transformers = transformers
    def __call__(self,img,bboxes,labels):
        for t in self.transformers:
            img,bboxes,labels = t(img,bboxes,labels)
        return img,bboxes,labels

#隨機顏色變換 randomcolor 而randombrightness~randomnoise為其可選擇的操作
class randomcolor():
    def __init__(self):
        self.brightness = randombrightness(50)
        self.colorstep = [
            randomcontrast(0.5,1.5),
            convertaxissystem("HSV"),
            randomsaturation(0.5,1.5),
            randomhue(18),
            convertaxissystem("BGR"),
            randomcontrast(0.5,1.5)
        ]
        self.noise = randomnoise()
    def __call__(self,img,bboxes,labels):
        img,bboxes,labels = self.brightness(img,bboxes,labels)
        if random.randint(2):
            step = compose(self.colorstep[:-1])
        else:
            step = compose(self.colorstep[1:])
        img,bboxes,labels = step(img,bboxes,labels)
        img,bboxes,labels = self.noise(img,bboxes,labels)
        return img,bboxes,labels

class randombrightness():
    def __init__(self,change):
        self.change = change
    def __call__(self,img,bboxes,labels):
        if random.randint(2):
            a = random.uniform(-self.change,self.change)
            img += a
        return img,bboxes,labels

class randomcontrast():
    def __init__(self,min_multiple,max_multiple):
        self.min_multiple = min_multiple
        self.max_multiple = max_multiple
    def __call__(self,img,bboxes,labels):
        if random.randint(2):
            a = random.uniform(self.min_multiple,self.max_multiple)
            img *= a
        return img,bboxes,labels

class convertaxissystem():
    def __init__(self,goto_axissystem):
        self.goto_axissystem = goto_axissystem
    def __call__(self,img,bboxes,labels):
        if self.goto_axissystem == "HSV":
            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        elif self.goto_axissystem == "BGR":
            img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
        return img,bboxes,labels

class randomsaturation():
    def __init__(self,min_multiple,max_multiple):
        self.min_multiple = min_multiple
        self.max_multiple = max_multiple
    def __call__(self,img,bboxes,labels):
        if random.randint(2):
            a = random.uniform(self.min_multiple,self.max_multiple)
            img[:,:,1] *= a
        return img,bboxes,labels

class randomhue():
    def __init__(self,change):
        self.change = change
    def __call__(self,img,bboxes,labels):
        if random.randint(2):
            a = random.uniform(-self.change,self.change)
            img[:,:,0] += a
            img[:,:,0][img[:,:,0]>360.0] -= 360.0
            img[:,:,0][img[:,:,0]<0.0] += 360.0
        return img,bboxes,labels

class randomnoise():
    def __call__(self,img,bboxes,labels):
        h,w,_ = img.shape
        r = [100,1000,200,2000,500]
        if random.randint(2):
            for _ in range(int(h*w/r[random.randint(len(r))])):
                img[random.randint(h),random.randint(w)] = [random.randint(256),random.randint(256),random.randint(256)]
        return img,bboxes,labels

#將輸入圖片轉為float32 以便對圖片進行隨機顏色變換
class convertfromints():
    def __call__(self,img,bboxes,labels):
        return img.astype(np.float32), bboxes, labels

#隨機剪裁圖片
class randomsample():
    def __call__(self,img,bboxes,labels):
        height,width,_ = img.shape
        while True:
            nheight = int(random.uniform(0.3*height,height))
            nwidth = int(random.randint(0.3*width,width))
            nimgtop = random.randint(height-nheight)
            nimgleft = random.randint(width-nwidth)
            nimgrect = [nimgleft,nimgtop,nimgleft+nwidth,nimgtop+nheight]

            ious,ids = calculateiou(nimgrect,bboxes)
            if ious == []:
                continue

            nimg = img[nimgrect[1]:nimgrect[3],nimgrect[0]:nimgrect[2]]
            nlabels = [labels[ids[i]] for i in range(len(ids))]
            nbboxes = []
            for i in range(len(ids)):
                nbbox = []
                if max(bboxes[ids[i],0],nimgrect[0]) == nimgrect[0]:
                    nbbox.append(0)
                else:
                    nbbox.append(bboxes[ids[i],0]-nimgrect[0])
                if max(bboxes[ids[i],1],nimgrect[1]) == nimgrect[1]:
                    nbbox.append(0)
                else:
                    nbbox.append(bboxes[ids[i],1]-nimgrect[1])
                if min(bboxes[ids[i],2],nimgrect[2]) == nimgrect[2]:
                    nbbox.append(nimgrect[2]-nimgrect[0])
                else:
                    nbbox.append(bboxes[ids[i],2]-nimgrect[0])
                if min(bboxes[ids[i],3],nimgrect[3]) == nimgrect[3]:
                    nbbox.append(nimgrect[3]-nimgrect[1])
                else:
                    nbbox.append(bboxes[ids[i],3]-nimgrect[1])
                nbboxes.append(nbbox)

            return nimg,np.array(nbboxes),np.array(nlabels)

def calculateiou(nimgrect,bboxes):
    ious = []
    ids = []
    for i in range(len(bboxes)):
        bboxarea = (bboxes[i,2]-bboxes[i,0])*(bboxes[i,3]-bboxes[i,1])

        interh = min(nimgrect[2],bboxes[i,2])-max(nimgrect[0],bboxes[i,0])
        interw = min(nimgrect[3],bboxes[i,3])-max(nimgrect[1],bboxes[i,1])
        intrsectionarea = interh*interw

        iou = intrsectionarea/bboxarea
        if 1 >= iou and iou >=  0.5:
            ious.append(iou)
            ids.append(i)
    return ious,ids

#隨機左右翻轉圖片
class randommirror():
    def __call__(self,img,bboxes,labels):
        _, width, _ = img.shape
        if random.randint(2):
            img = img[:,::-1]
            img = img.copy()        #不知道為甚麼要這行 -----> 因為你雞雞小
            bboxes[:,0::2] = width-bboxes[:,0::2]-1
        return img,bboxes,labels

#調整圖片大小
class resize():
    def __init__(self,imgsize):
        self.imgsize = imgsize
    def __call__(self,img,bboxes,labels):
        height,width,_ = img.shape
        if height>width:
            helf = int((height-width)/2)
            nimg = np.ones((height,height,3),np.float32)
            nimg[:,helf:helf+width] = img[:,:]
            nimg[:,:helf] = [127,127,127]
            nimg[:,helf+width:] = [127,127,127]
            bboxes[:,::2] = bboxes[:,::2]+helf+1
        else:
            helf = int((width-height)/2)
            nimg = np.ones((width,width,3),np.float32)
            nimg[helf:helf+height,:] = img[:,:]
            nimg[:helf,:] = [127,127,127]
            nimg[helf+height:,:] = [127,127,127]
            bboxes[:,1::2] = bboxes[:,1::2]+helf+1
        
        r = self.imgsize/len(nimg)
        img = cv2.resize(nimg,(self.imgsize,self.imgsize))
        bboxes = bboxes*r
        bboxes = bboxes.astype(np.int32)
        return img,bboxes,labels

#將圖片RGB標準化
class normailize():
    def __init__(self,mean,std):
        self.mean = np.array(mean,dtype=np.float32)
        self.std = np.array(std,dtype=np.float32)
    def __call__(self,img,bboxes,labels):
        img -= self.mean
        img /= self.std
        return img,bboxes,labels



def show_img():
    if d["randombrightness"] == True:
        img = cv2.imread(line[0])
        img = img.astype(np.float32)
        cv2.imshow("origine",img/255)
        img1 = img+50
        cv2.imshow("brightness up",img1/255)
        img2 = img-50
        cv2.imshow("brightness down",img2/255)
        cv2.waitKey(0)
    if d["randomcontrast"] == True:
        img = cv2.imread(line[0])
        img = img.astype(np.float32)
        cv2.imshow("origine",img/255)
        img1 = img*1.5
        cv2.imshow("contrast up",img1/255)
        img2 = img*0.5
        cv2.imshow("contrast down",img2/255)
        cv2.waitKey(0)
    if d["randomsaturation"] == True:
        import copy
        img = cv2.imread(line[0])
        img = img.astype(np.float32)
        cv2.imshow("origine",img/255)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img1 = copy.deepcopy(img)
        img1[:,:,1] = img1[:,:,1]*1.5
        img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2BGR)
        cv2.imshow("saturation up",img1/255)
        img2 = copy.deepcopy(img)
        img2[:,:,1] = img2[:,:,1]*0.5
        img2 = cv2.cvtColor(img2,cv2.COLOR_HSV2BGR)
        cv2.imshow("saturation down",img2/255)
        cv2.waitKey(0)
    if d["randomhue"] == True:
        import copy
        img = cv2.imread(line[0])
        img = img.astype(np.float32)
        cv2.imshow("origine",img/255)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img1 = copy.deepcopy(img)
        img1[:,:,0] = img1[:,:,0]+360
        img1[:,:,0][img1[:,:,0] > 360.0] -= 360.0
        img1[:,:,0][img1[:,:,0] < 0.0] += 360.0
        img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2BGR)
        cv2.imshow("hue up",img1/255)
        img2 = copy.deepcopy(img)
        img2[:,:,0] = img2[:,:,0]-360
        img2[:,:,0][img2[:,:,0] > 360.0] -= 360.0
        img2[:,:,0][img2[:,:,0] < 0.0] += 360.0
        img2 = cv2.cvtColor(img2,cv2.COLOR_HSV2BGR)
        cv2.imshow("hue down",img2/255)
        cv2.waitKey(0)
    if d["randomnoise"] == True:
        img = cv2.imread(line[0])
        img = img.astype(np.float32)
        cv2.imshow("before",img/255)
        r = [100,1000,200,2000,500]
        h,w,_ = img.shape
        for _ in range(int(h*w/r[random.randint(len(r))])):
            img[random.randint(h),random.randint(w)] = [random.randint(256),random.randint(256),random.randint(256)]
        cv2.imshow("after",img/255)
        cv2.waitKey(0)
    if d["randomsample"] == True:
        img = cv2.imread(line[0])
        img = img.astype(np.float32)
        for i in range(len(bboxes)):
            cv2.rectangle(img, (bboxes[i,0],bboxes[i,1]), (bboxes[i,2],bboxes[i,3]), (0, 255, 0), 1)
        cv2.imshow("before",img/255)
        print("原bboxes",bboxes)
        print("原labels",labels)
        height,width,_ = img.shape
        nheight = int(0.5*height)
        nwidth = int(0.7*width)
        nimgtop = int(0.4*height)
        nimgleft = int(0.2*width)
        nimgrect = [nimgleft,nimgtop,nimgleft+nwidth,nimgtop+nheight]
        ious,ids = calculateiou(nimgrect,bboxes)
        nimg = img[nimgrect[1]:nimgrect[3],nimgrect[0]:nimgrect[2]]
        nlabels = [labels[ids[i]] for i in range(len(ids))]
        nbboxes = []
        for i in range(len(ids)):
            nbbox = []
            if max(bboxes[ids[i],0],nimgrect[0]) == nimgrect[0]:
                nbbox.append(0)
            else:
                nbbox.append(bboxes[ids[i],0]-nimgrect[0])
            if max(bboxes[ids[i],1],nimgrect[1]) == nimgrect[1]:
                nbbox.append(0)
            else:
                nbbox.append(bboxes[ids[i],1]-nimgrect[1])
            if min(bboxes[ids[i],2],nimgrect[2]) == nimgrect[2]:
                nbbox.append(nimgrect[2]-nimgrect[0])
            else:
                nbbox.append(bboxes[ids[i],2]-nimgrect[0])
            if min(bboxes[ids[i],3],nimgrect[3]) == nimgrect[3]:
                nbbox.append(nimgrect[3]-nimgrect[1])
            else:
                nbbox.append(bboxes[ids[i],3]-nimgrect[1])
            nbboxes.append(nbbox)
        nbboxes = np.array(nbboxes)
        nlabels = np.array(nlabels)
        for i in range(len(nbboxes)):
            cv2.rectangle(nimg, (nbboxes[i,0],nbboxes[i,1]), (nbboxes[i,2],nbboxes[i,3]), (0, 0, 255), 1)
        cv2.imshow("after",nimg/255)
        print("調後bboxes",nbboxes)
        print("調後labels",nlabels)
        cv2.waitKey(0)
    if d["randommirror"] == True:
        img = cv2.imread(line[0])
        img = img.astype(np.float32)
        for i in range(len(bboxes)):
            cv2.rectangle(img,(bboxes[i,0],bboxes[i,1]),(bboxes[i,2],bboxes[i,3]),(0,255,0),1)
        cv2.imshow("before",img/255)
        print("原bboxes",bboxes)
        print("原labels",labels)
        _, width, _ = img.shape
        img = img[:,::-1]
        img = img.copy()
        bboxes[:,0::2] = width-bboxes[:,2::-2]-1
        for i in range(len(bboxes)):
            cv2.rectangle(img,(bboxes[i,0],bboxes[i,1]),(bboxes[i,2],bboxes[i,3]),(0,0,255),1)
        cv2.imshow("after",img/255)
        print("調後bboxes",bboxes)
        print("調後labels",labels)
        cv2.waitKey(0)
    if d["resize"] == True:
        img = cv2.imread(line[0])
        img = img.astype(np.float32)
        for i in range(len(bboxes)):
            cv2.rectangle(img,(bboxes[i,0],bboxes[i,1]),(bboxes[i,2],bboxes[i,3]),(0,255,0),1)
        cv2.imshow("before",img/255)
        print("原bboxes",bboxes)
        print("原labels",labels)
        height,width,_ = img.shape
        if height>width:
            helf = int((height-width)/2)
            nimg = np.ones((height,height,3),np.float32)
            nimg[:,helf:helf+width] = img[:,:]
            nimg[:,:helf] = [127,127,127]
            nimg[:,helf+width:] = [127,127,127]
            bboxes[:,::2] = bboxes[:,::2]+helf
        else:
            helf = int((width-height)/2)
            nimg = np.ones((width,width,3),np.float32)
            nimg[helf:helf+height,:] = img[:,:]
            nimg[:helf,:] = [127,127,127]
            nimg[helf+height:,:] = [127,127,127]
            bboxes[:,1::2] = bboxes[:,1::2]+helf
        r = 300/len(nimg)
        img = cv2.resize(nimg,(300,300))
        nbboxes=bboxes.copy()
        nbboxes = nbboxes*r
        nbboxes = nbboxes.astype(np.int32)
        for i in range(len(nbboxes)):
            cv2.rectangle(img,(nbboxes[i,0],nbboxes[i,1]),(nbboxes[i,2],nbboxes[i,3]),(0,0,255),1)
        cv2.imshow("after",img/255)
        print("調後bboxes",bboxes)
        print("調後labels",labels)
        cv2.waitKey(0)
    if d["normailize"] == True:
        img = cv2.imread(line[0])
        img = img.astype(np.float32)
        for i in range(len(bboxes)):
            cv2.rectangle(img,(bboxes[i,0],bboxes[i,1]),(bboxes[i,2],bboxes[i,3]),(0,255,0),1)
        cv2.imshow("before",img/255)
        print("原bboxes",bboxes)
        print("原labels",labels)
        img /= 255.
        img -= np.array((0.406, 0.456, 0.485))
        img /= np.array((0.225, 0.224, 0.229))
        nbboxes=bboxes.copy()
        nbboxes = nbboxes.astype(np.int32)
        for i in range(len(nbboxes)):
            cv2.rectangle(img,(nbboxes[i,0],nbboxes[i,1]),(nbboxes[i,2],nbboxes[i,3]),(0,0,255),1)
        cv2.imshow("after",img)
        print("調後bboxes",bboxes)
        print("調後labels",labels)
        print(img)
        cv2.waitKey(0)



if __name__ == "__main__":
    path = "newdata/train.txt"
    with open(path, encoding='utf-8') as f:
        train_lines = f.readlines()
    line = train_lines[1].split()

    img = cv2.imread(line[0])
    targets = np.array([list(map(int,box.split(","))) for box in line[1:]])
    bboxes = targets[:,0:4]
    labels = targets[:,4]

    #------------------------------augmentation調整的效果顯示------------------------------#
    show = True
    if show:
        a=augmentation(300,(0.406*255,0.456*255,0.485*255),(0.225*255,0.224*255,0.229*255))
        img = img.astype(np.float32)
        for i in range(len(bboxes)):
            cv2.rectangle(img,(bboxes[i,0],bboxes[i,1]),(bboxes[i,2],bboxes[i,3]),(255,255,255),1)
        cv2.imshow("before",img/255)
        print("原bboxes\n",bboxes)
        print("原labels\n",labels)
        img,bboxes,labels = a(img,bboxes,labels)
        for i in range(len(bboxes)):
            cv2.rectangle(img,(bboxes[i,0],bboxes[i,1]),(bboxes[i,2],bboxes[i,3]),(0,0,255),1)
        cv2.imshow("after",img/255)
        print("調後bboxes\n",bboxes)
        print("調後labels\n",labels)
        cv2.waitKey(0)

    #------------------------------各個調整的效果顯示------------------------------#
    d = {"randombrightness":False,
         "randomcontrast":False,
         "randomsaturation":False,
         "randomhue":False,
         "randomnoise":False,
         "randomsample":False,
         "randommirror":False,
         "resize":False,
         "normailize":False}
    show_img()
