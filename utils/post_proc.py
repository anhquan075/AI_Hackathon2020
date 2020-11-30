import os
import shutil
from PIL import Image

INP_SUB_PATH = '/storage/timm/'
INP_SUB_FILE = 'submission_best_tf_efficientnet_b8.txt'

OUT_SUB_PATH = '/storage/submissions/'
OUT_SUB_FILE = 'team007_b8tf_datacrop_BEST_postproc_final.txt'

TEST_SET_DIR = '/dataset/test_set_A_full/'

THRESHOLD = {
    '0': 0,
    '1': 0.62,
    '2': 0,
    '3': 0,
    '4': 0.52,
    '5': 0,
    '6': 0.41,
    '7': 0,
}

WIDTH0 = [
    
]

HEIGHT0 = [
    500,
    640,
    672,
    704,
    736,
    768,
    800,
    832,
    864,
    896,
    928,
    960,
]

# Yêu cầu file inp_sub phải theo format "file_name  top1    score1  top2    score2"
# Trả về path đến 2 file: (submission theo format btc, submission kèm theo score để viz)
def post_proc(INP_SUB_PATH, INP_SUB_FILE, OUT_SUB_PATH, OUT_SUB_FILE, TEST_SET_DIR):
    fn, _ = os.path.splitext(OUT_SUB_FILE)
    test_dir = os.path.dirname(TEST_SET_DIR+'/')
    CROP_DIR = test_dir + '_cropped'
    NONE_DIR = test_dir + '_none'
    SQUARE_DIR = test_dir + '_square'

    with open(os.path.join(INP_SUB_PATH, INP_SUB_FILE), 'r') as f:
        lines = f.read().splitlines()

    lines = [line.strip().split() for line in lines]
    lines.sort(key=lambda x: x[0])
    
    def _post_proc(lines):
        result = []
        for line in lines:
            if len(line) == 2:
                result.append(line[:2])
                continue
            fi = line[0]
            top1 = line[1]
            
            score1 = '1.0'
            top2 = '7'
            score2 = '1.0'
            
            score1 = line[2]
            top2 = line[3]
            score2 = line[4]
            
            img_path = os.path.join(CROP_DIR, fi)
            if not os.path.exists(img_path):
                continue
            img_path = os.path.join(TEST_SET_DIR, fi)
            print(img_path)
            img = Image.open(img_path)
            width, height = img.size
            if top1 == '0' and round(width/height, 4) == 1.7778:
                result.append([fi, top2, score2, top1, score1])
            elif width in WIDTH0 or height in HEIGHT0:
                result.append([fi, '0', '1.0', top1, score1])
            elif float(score1) <= THRESHOLD[top1]:
                result.append([fi, top2, score2, top1, score1])
            else:
                result.append([fi, top1, score1, top2, score2])
                
        return result
            
    def _populate_none(result):
        dir = os.listdir(NONE_DIR)
        for fi in dir:
            result.append([fi, '0', '1.0000', '0', '1.0000'])
            
    def _populate_square(result):
        dir = os.listdir(SQUARE_DIR)
        for fi in dir:
            result.append([fi, '0', '1.0000', '0', '1.0000'])

    result = _post_proc(lines)
    
    print(len(result))
    
    _populate_none(result)
    _populate_square(result)

    # print(result)
    print(len(result))

    out_sub_path = os.path.join(OUT_SUB_PATH, OUT_SUB_FILE)
    with open(out_sub_path, 'w') as f:
        for line in result:
            f.write('\t'.join(line[:2]) + '\n')

    out_sub_path_viz = os.path.join(OUT_SUB_PATH, "{}_viz.txt".format(fn))
    with open(out_sub_path_viz, 'w') as f:
        for line in result:
            f.write('\t'.join(line) + '\n')
            
    return out_sub_path, out_sub_path_viz
            
            
if __name__ == '__main__':
    post_proc(INP_SUB_PATH, INP_SUB_FILE, OUT_SUB_PATH, OUT_SUB_FILE, TEST_SET_DIR)