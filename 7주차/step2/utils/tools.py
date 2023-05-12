# 필요 패키지 로드 
import os 

cifar_mean = [0.49139968, 0.48215827, 0.44653124]
cifar_std = [0.24703233, 0.24348505, 0.26158768]

# 번호를 통해 폴더별로 best model을 저장
def get_save_folder_path(args): 
    # 폴더 경로 존재하지 않을시 
    if not os.path.exists(args.save_folder): 
        # 폴더 생성
        os.makedirs(args.save_folder)
        # 새로운 폴더명 정의
        new_folder_name = '1'
    else : 
        # 폴더 내의 번호중 가장 큰 번호 찾음
        current_max_value = max([int(f) for f in os.listdir(args.save_folder) if '.DS_Store' not in f])
        # 폴더 번호 업데이트
        new_folder_name = str(current_max_value + 1)
    
    # 폴더명+파일명 경로 지정
    path = os.path.join(args.save_folder, new_folder_name)
    # return
    return path 