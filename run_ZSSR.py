# import GPUtil
import glob
import os
from utils import prepare_result_dir
import configs
from time import sleep
import sys
import run_ZSSR_single_input

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def main(conf_name, gpu):
    # Initialize configs and prepare result dir with date
    print(conf_name)
    if conf_name is None:
        conf = configs.Config()

    else:
        namespace = {'conf': None}
        conf = eval('configs.%s' % conf_name)

    res_dir = prepare_result_dir(conf)
    local_dir = os.path.dirname(__file__)

    # We take all png files that are not ground truth
    files = [file_path for file_path in glob.glob('%s/*.jpg' % conf.input_path)
             if not file_path[-7:-4] == '_gt']

    # Loop over all the files
    for file_ind, input_file in enumerate(files):

        # Ground-truth file needs to be like the input file with _gt (if exists)
        ground_truth_file = input_file[:-4] + '_gt.jpg'
        if not os.path.isfile(ground_truth_file):
            ground_truth_file = '0'

        # Numeric kernel files need to be like the input file with serial number
        kernel_files = ['%s_%d.mat;' % (input_file[:-4], ind) for ind in range(len(conf.scale_factors))]
        kernel_files_str = ''.join(kernel_files)
        for kernel_file in kernel_files:
            if not os.path.isfile(kernel_file[:-1]):
                kernel_files_str = '0'
                print('no kernel loaded')
                break

        print(kernel_files)

        run_ZSSR_single_input.main(input_file, ground_truth_file, kernel_files_str, gpu, conf_name, res_dir)

if __name__ == '__main__':
    conf_str = sys.argv[1] if len(sys.argv) > 1 else None
    print(conf_str)
    gpu_str = sys.argv[2] if len(sys.argv) > 2 else None
    main(conf_str, gpu_str)


