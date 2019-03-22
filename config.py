import os
root_dir = '/Users/xingqiangchen/Desktop/2019-02-22/'
data_path = os.path.join(root_dir, 'data')
data_check_path = os.path.join(root_dir,'data_check')
data_prod_path = os.path.join(root_dir,'data_prod')

if not os.path.exists(data_check_path):
    os.makedirs(data_check_path)

if not os.path.exists(data_prod_path):
    os.makedirs(data_prod_path)