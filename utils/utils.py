import os 

def load_class_names(filename):
    with open(filename, 'r', encoding='utf8') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

'''
    path data fold: full path 
    path_data_ck: full path
'''
def split_dataService(path_data_fold, path_data_ck, fold, number_service=16):
    fold_path = os.path.join(path_data_ck, "fold_" + str(fold))
    if not os.path.exists(fold_path):
        os.mkdir(fold_path)

    list_filenames = os.listdir(path_data_fold)
    number_path_in_file = (len(list_filenames) // number_service) + 1
    arr_numberfile = [number_path_in_file * (i+1) for i in range(number_service)]
    
    len_list_filenames = len(list_filenames)
    print(len_list_filenames)
    cnt_file = 0
    cnt_service = 1
    log_list_paths_for_service = os.path.join(fold_path, "log_for_fold_{}.txt".format(str(fold)))
    for number in arr_numberfile:
        file_txt_name = os.path.join(fold_path, "service_{}.txt".format(str(cnt_service)))
        with open(log_list_paths_for_service, "a+") as fp:
            fp.write("{}\n".format(file_txt_name))

        while cnt_file < number:
            file_path = os.path.join(path_data_fold, list_filenames[cnt_file])
            cnt_file += 1
            with open(file_txt_name, "a+") as f:
                f.write("{}\n".format(file_path))
            
            if cnt_file >= len_list_filenames:
                return log_list_paths_for_service
        
        cnt_service += 1
    
    return log_list_paths_for_service

# split_dataService('data_ck/ai4vn_2020', 'path_data_ck', 1)