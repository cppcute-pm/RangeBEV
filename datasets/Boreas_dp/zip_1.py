import os
import shutil
from multiprocessing import Pool

def process_traversal(traversal):
    subdir_path = os.path.join(source_dir, traversal)
    # 如果是子目录，就压缩它
    if os.path.isdir(subdir_path):
        print(f"start zip {traversal}")
        shutil.make_archive(os.path.join(target_dir, traversal), 'zip', subdir_path)
        print(f"finish zip {traversal}")

if __name__ ==  '__main__':

    data_root = '/media/group2/data/pengjianyi'
    source_dir = os.path.join(data_root, 'Boreas_minuse')
    target_dir = os.path.join(data_root, 'Boreas_minuse_zip')

    # 创建目标目录，如果它不存在
    os.makedirs(target_dir, exist_ok=True)
    traversal_list = ["my_tool"]

    with Pool() as p:
        p.map(process_traversal, traversal_list)