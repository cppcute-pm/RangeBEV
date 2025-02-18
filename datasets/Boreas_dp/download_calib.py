from pyboreas.data.splits import loc_train
import subprocess
from multiprocessing import Pool
import time

def process_traversal(traversal):
    traversal = traversal[0]
    # print(f'current traversal is {traversal}')
    download_cmd = f'aws s3 sync s3://boreas/{traversal}  {data_root}/{traversal} --exclude "*"  --include "applanix/*" --include "calib/*" --no-sign-request'
    print(download_cmd)
    subprocess.run(download_cmd, shell=True)
    print(f'current traversal {traversal} is downloaded')

if __name__ == '__main__':
    # traversal_list = loc_train
    traversal_list = loc_train
    data_root = '/DATA1/Boreas'
    # print(f'as not all the traversal has truth pose, we only choose the loc_train split for now')
    while True:
        with Pool() as p:
            p.map(process_traversal, traversal_list)
        time.sleep(5 * 60)
