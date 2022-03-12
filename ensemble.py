import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=False,
                        default='ntu/xsub',
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help="epoch1_test_score.pkl")
    parser.add_argument('--bone-dir',
                        help="epoch1_test_score.pkl")


    arg = parser.parse_args()

    dataset = arg.dataset
    if 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/ntu120/NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/'  + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    # with open(arg.joint_dir), 'rb' as r1:
    r1 = open("./work_dir/ntu60/xsub/PAT-Net_joint/epoch1_test_score.pkl", "rb")
    r1 = list(pickle.load(r1).items())

    # with open(arg.bone_dir, 'rb') as r2:
    r2 = open("./work_dir/ntu60/xsub/PAT-Net_bone/epoch1_test_score.pkl", "rb")
    r2 = list(pickle.load(r2).items())

    # with open(arg.motion_dir, 'rb') as r3:
    r3 = open("./work_dir/ntu60/xsub/PAT-Net_Joint_motion/epoch1_test_score.pkl", "rb")
    r3 = list(pickle.load(r3).items())

    r4 = open("./work_dir/ntu60/xsub/PAT-Net_bone_motion/epoch1_test_score.pkl", "rb")
    r4 = list(pickle.load(r4).items())


    right_num = total_num = right_num_5 = 0

    arg.alpha = [0.6, 0.6, 0.4, 0.4]
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]


        r = r11 * arg.alpha[0] + r22 * arg.alpha[1]  + r33 * arg.alpha[2] + r44 * arg.alpha[3]
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
