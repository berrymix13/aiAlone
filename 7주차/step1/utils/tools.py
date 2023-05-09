import os 

cifar_mean = [0.49139968, 0.48215827, 0.44653124]
cifar_std = [0.24703233, 0.24348505, 0.26158768]

def get_save_folder_path(args): 
    if not os.path.exists(args.save_folder): 
        os.makedirs(args.save_folder)
        new_folder_name = '1'
    else : 
        current_max_value = max([int(f) for f in os.listdir(args.save_folder) if '.DS_Store' not in f])
        new_folder_name = str(current_max_value + 1)
    
    path = os.path.join(args.save_folder, new_folder_name)
    return path 