import cv2
import os
inpath = "train"
outpath = "train_l1"
for i,fname in enumerate(os.listdir(inpath)):
    fpath = os.path.join(inpath, fname)
    print(i, fpath)
    if fname.endswith(".txt"):
        iname = fname.split(".txt")[0]+".jpg"
        ipath = os.path.join(inpath, iname)
        if not os.path.exists(ipath):
            continue
        print(ipath)
        img = cv2.imread(ipath)
        h,w,_ = img.shape
        with open(fpath, "r") as f:
            c = f.readlines()
        with open(os.path.join(outpath, "label", fname), "w") as f:
            for line in c:
                cls,scx,scy,sw,sh = [float(x) for x in line.rstrip().split(" ")]
                cx = scx*w
                cy = scy*h
                yw = sw*w
                yh = sh*h
                x1 = int(cx-yw/2)
                x2 = int(x1+yw)
                y1 = int(cy-yh/2)
                y2 = int(y1+yh)
                cls = int(cls)
                
                f.write(str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(cls)+"\n")
        
        cv2.imwrite(os.path.join(outpath, "img", iname), img)
        
