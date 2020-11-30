import os, shutil, cv2
from PIL import Image

INP_DIR = '/dataset/test_set_A_full'


# Lọc thư mục data test ra thành 3 thư mục: None, Square (1:1), và phần còn lại (đã được crop ngay chính giữa)
# Trả về path dẫn đến 3 thư mục nói trên
def pre_proc(INP_DIR):
    INP_DIR = INP_DIR + '/'
    NONE_DIR = os.path.dirname(INP_DIR) + '_none'
    SQUARE_DIR = os.path.dirname(INP_DIR) + '_square'
    CROP_DIR = os.path.dirname(INP_DIR) + '_cropped'

    os.makedirs(NONE_DIR, exist_ok=True)
    os.makedirs(SQUARE_DIR, exist_ok=True)
    os.makedirs(CROP_DIR, exist_ok=True)

    dir = os.listdir(INP_DIR)
    dir.sort()

    for fi in dir:
        print(fi)
        inp_path = os.path.join(INP_DIR, fi)
        
        img = Image.open(inp_path)
        if img.format == 'GIF': 
            shutil.move(inp_path, os.path.join(NONE_DIR, fi))
            continue
        width, height = img.size
        if width == height:
            shutil.copyfile(inp_path, os.path.join(SQUARE_DIR, fi))
            continue
        
        img = cv2.imread(inp_path)
        if width > height:
            img = img[:, width//2-height//2:width//2+height//2]
        else:
            img = img[height//2-width//2:height//2+width//2, :]
            
        cv2.imwrite(os.path.join(CROP_DIR, fi), img)
        
    return NONE_DIR, SQUARE_DIR, CROP_DIR
        
if __name__ == '__main__':
    pre_proc(INP_DIR)
        