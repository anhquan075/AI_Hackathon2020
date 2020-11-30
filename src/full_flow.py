import os

from src import predict_efficientNetB8
from utils import post_proc, pre_proc

INP_DIR = '/dataset/test_flow/test_set_A_full/'

def full_flow(INP_DIR, fold):

    NONE_DIR, SQUARE_DIR, CROP_DIR = pre_proc(INP_DIR)

    SUB_PATH = predict_efficientNetB8(CROP_DIR, BATCH_SIZE=8, MODEL_PATH='model/checkpoint-4.pth.tar')

    SUB_DIR_PATH = os.path.dirname(SUB_PATH)
    SUB_FILE_PATH = os.path.basename(SUB_PATH)

    OUT_SUB_PATH, OUT_SUB_PATH_VIZ = post_proc(SUB_DIR_PATH, SUB_FILE_PATH, 'submission', 'sub_test_flow_{}.txt'.format(fold), INP_DIR)

    return OUT_SUB_PATH, OUT_SUB_PATH_VIZ

if __name__ == '__main__':
    print(full_flow(INP_DIR))