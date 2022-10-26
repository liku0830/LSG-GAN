import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        data = parser.add_argument_group('Dataset')
        data.add_argument('--split', type=str, default='train',help='train or test')
        data.add_argument('--dataset_root',type=str, default='./datasets/',help='the root of dataset')
        data.add_argument('--image_path', type=str,default='./datasets/image/',help='the root of test_image')
        data.add_argument('--pose_path', type=str,default='./FPGAN/datasets/pose/',help='the root of train_pose')
        data.add_argument('--mask_path', type=str,default='./FPGAN/datasets/mask/',help='the root of train_mask')
        data.add_argument('--texture_path', type=str,default='./FPGAN/datasets/',help='the root of dataset')
        data.add_argument('--shuffle', type=str, default="True",help='shuffle')


        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--template_dir', help='the directory of template',type=str, default='F:/FPGAN/datasets/tem/')

        g_train.add_argument('--save_pic_root', help='root of save picture', type=str,default='E:/lyw/AttentionTransfer/result/0810/whole_net/pic/')
        g_train.add_argument('--save_modelG_root', help='root of save generator model', type=str,default='E:/lyw/AttentionTransfer/result/0810/whole_net/netG/')
        g_train.add_argument('--save_modelD_root', help='root of save discriminator model', type=str,default='E:/lyw/AttentionTransfer/result/0810/whole_net/netD/')

        #g_train.add_argument('--pretrain_dir', help='the directory of template', type=str,default='E:/lyw/AttentionTransfer/result/0810/whole_net/netG/checkpointG_epoch52.pth')
        g_train.add_argument('--pretrain_netG', help='the directory of template', type=str,default='E:/lyw/AttentionTransfer/result/0810/whole_net/netG/checkpointG_epoch52.pth')
        g_train.add_argument('--pretrain_netD', help='the directory of template', type=str,default='E:/lyw/AttentionTransfer/result/0810/whole_net/netD/checkpointG_epoch52.pth')
        g_train.add_argument('--use_gp', help='turn on or turn off the gp function', type=str,default='False')

        g_train.add_argument('--gpu_id', type=int, default=0,help='gpu id for cuda')
        g_train.add_argument('--gpu_ids', type=str, default='0',help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

        g_train.add_argument('--epo_num', type=int, default=9999,help='all epo_num')
        g_train.add_argument('--start_epoch', type=int, default=0,help='start epoch num')
        g_train.add_argument('--batch_size', type=int, default=4,help='input batch size')

        g_train.add_argument('--learning_rateD', type=float, default=1e-4,help='adam learning rate')
        g_train.add_argument('--learning_rateG', type=float, default=1e-4,help='adam learning rate')
        g_train.add_argument('--lossG_gan_weight', type=float, default=3e-1,help='lossG_gan_weight')
        g_train.add_argument('--lossG_style_weight', type=float, default=500,help='lossG_style_weight')
        g_train.add_argument('--lossG_style_temp', type=float, default=1e-1,help='lossG_temp_weight')
        g_train.add_argument('--num_epoch', type=int, default=100,help='num epoch to train')
        g_train.add_argument('--num_workers', type=int, default=1,help='num_workers')


        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--resolution', type=int, default=256,help='# of grid in mesh reconstruction')
        g_test.add_argument('--test_folder_path', type=str, default=None,help='the folder of test image')


        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()


    def parse(self):
        opt = self.gather_options()
        return opt
