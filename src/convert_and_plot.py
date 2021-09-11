# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import os
import nibabel as nib
import pydicom as dicom
import matplotlib.pyplot as plt

# %% [markdown]
# # Convert nii file to txt file and plot

# %%
HOME = 'D:\\results0622\\Results\\'
subdirs = ['ALFF_FunImgWC','DegreeCentrality_FunImgWCF','fALFF_FunImgWC','ReHo_FunImgWCF']

# %%
for subdir in subdirs:
    home = HOME + subdir
    for fname in os.listdir(home):
        if fname.split('.')[-1] == 'nii':
            fp = home + '\\' + fname
            f = nib.load(fp)
            f_data = f.get_fdata()
            dim_0, dim_1, dim_2 = f.header.get_data_shape()
            N = dim_0 * dim_1 * dim_2
            np.savetxt(home + '\\' + fname.split('.')[0] + '.txt', 
                       f_data.reshape(N,))


# # %%
# a = np.array([
    
#               [[1,2,3],
#                [4,5,6]],
    
#               [[7,8,9],
#                [10,11,12]],
               
#               [[13,14,15],
#                [16,17,18]],
    
#               [[19,20,21],
#                [22,23,24]]
    
#               ])

# print(a.shape)
# print(a.reshape(24,))


# # %%
# subjects = ['Sub_001','Sub_002','Sub_003','Sub_004','Sub_005','Sub_006','Sub_007','Sub_008','Sub_009','Sub_010',]
# for i in subjects:
#     p = HOME + '\\FunImgARFCB_ROISignals\\ROISignals_' + i + '.txt'
#     s = np.loadtxt(p)
#     print('ROISignals_' + i + ': ', s.shape)
# for i in subjects:
#     p = HOME + '\\FunImgARFCB_ROISignals\\ROICorrelation_' + i + '.txt'
#     s = np.loadtxt(p)
#     print('ROICorrelation_' + i + ': ', s.shape)
# for i in subjects:
#     p = HOME + '\\FunImgARFCB_ROISignals\\ROICorrelation_FisherZ_' + i + '.txt'
#     s = np.loadtxt(p)
#     print('ROICorrelation_FisherZ_' + i + ': ', s.shape)


# # %%
# for i in subjects:
#     matrix = np.loadtxt(HOME + '\\FunImgARFCB_ROISignals\\ROICorrelation_FisherZ_'+ i +'.txt')
#     plt.imshow(matrix)
#     plt.title('ROICorrelation_FisherZ_' + i)
#     plt.savefig(HOME + '\\FunImgARFCB_ROISignals\\ROICorrelation_FisherZ_'+ i +'.png')


# # %%
# for i in subjects:
#     matrix = np.loadtxt(HOME + '\\FunImgARFCB_ROISignals\\ROICorrelation_'+ i +'.txt')
#     plt.imshow(matrix)
#     plt.title('ROICorrelation_FisherZ_' + i)
#     plt.savefig(HOME + '\\FunImgARFCB_ROISignals\\ROICorrelation_'+ i +'.png')


# # %%
# for i in subjects:
#     matrix = np.loadtxt(HOME + '\\FunImgARFCB_ROISignals\\ROISignals_'+ i +'.txt')
#     plt.imshow(matrix)
#     plt.title('ROICorrelation_FisherZ_' + i)
#     plt.savefig(HOME + '\\FunImgARFCB_ROISignals\\ROISignals_'+ i +'.png')

# # %% [markdown]
# # ----------
# # # Version check

# # %%
# dicom.__version__


# # %%
# nib.__version__

# # %% [markdown]
# # # NII file

# # %%
# # path = r'D:\result0621\data\FunImg\Sub_001'
# # img = nib.load(path + '\\Sub_001_Resting_State_fMRI_20121114151828_501.nii')
# # img_data = img.get_fdata()
# # print(img.header.get_data_shape())


# # # %%
# # def show_slices(slices):
# #     fig,axes = plt.subplots(1,len(slices))
# #     for i,slice in enumerate(slices):
# #         axes[i].imshow(slice.T,cmap="gray",origin="lower")

# # slice_0 = img_data[32, :, :]
# # slice_1 = img_data[:, 32, :]
# # slice_2 = img_data[:, :, 24]
# # show_slices([slice_0, slice_1, slice_2])
# # plt.suptitle("Center slices for EPI image")  

# # # %% [markdown]
# # # # DCM file

# # # %%
# # img_hcy = dicom.dcmread(path + '\\FunRaw\\Sub_001\\000001.dcm')
# # print(img_hcy)


# # %%



# # %%



# # %%


