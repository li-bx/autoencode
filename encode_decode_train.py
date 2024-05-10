import os
import sys
import time
cur_file_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_file_dir)

import cv2
from dataset_loader_two_face import get_data_loader

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from deepfake import Autoencoder

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
def _main() -> None:
    """ The main entry
    -  
    """
    # Hyper Parameters
    EPOCH = 200
    BATCH_SIZE = 16
    LR = 0.0001       # learning rate
    N_TEST_IMG = 5
    SAVE_MODEL = False   #False True
    READ_MODEL = False   #False True
    FLAG = 1
    WEIGHT_PATH=os.path.join(cur_file_dir,'weights\epoch_weight%d.pth' % (FLAG))
    
    autoencoder = Autoencoder().to(device)
    # load the weights
    # 计算参数总数
    # sum(x.numel() for x in autoencoder.parameters())
    if READ_MODEL:
        autoencoder.load_state_dict(torch.load(WEIGHT_PATH))
    autoencoder.train()

    optimizer = torch.optim.Adam(   autoencoder.parameters(), lr=LR,
                                    betas=(0.1, 0.999),# 平滑常数
                                    eps = 1e-08,      # 小常数,用于稳定数值
                                    weight_decay = 0, # L2正则化
                                    amsgrad = False     # 是否保留历史最大值
                                ) 


    loss_func = nn.MSELoss().to(device)

    plt.ion()   # Turn the interactive mode on, continuously plot
    # initialize figure
    f, a = plt.subplots(6, N_TEST_IMG, figsize=(10, 5))    # f是一块画布；a是一个大小为6*5的数组,数组中的每个元素都是一个画图对象
    f.suptitle('编码-解码,训练')
    plt.show()
    for epoch in range(EPOCH):    
        loader_train,dataset_train,loader_eval,dataset_eval = get_data_loader(cur_file_dir+'\\data\\target\\train.bin',cur_file_dir+'\\data\\target\\train.bin',batch_size=BATCH_SIZE)

        view_data1 = dataset_train.imgs1_tensor[:N_TEST_IMG]
        view_data2 = dataset_train.imgs2_tensor[:N_TEST_IMG]
        for i in range(N_TEST_IMG):
            a[0][i].imshow(view_data1[i].permute(1,2,0))
            a[0][i].set_xticks(()); 
            a[0][i].set_yticks(())
            a[3][i].imshow(view_data2[i].permute(1,2,0))
            a[3][i].set_xticks(()); 
            a[3][i].set_yticks(())        
        plt.pause(0.002)
        total_time=0
        current_time_millis = int(round(time.time() * 1000)) 
        for step, (img_tensor_nor1,img_tensor1, img1, \
                   img_tensor_nor2,img_tensor2, img2,) in enumerate(loader_train):

            total_time+= int(round(time.time() * 1000)) - current_time_millis
            print('loop time: %d | total time: %.2f' % (int(round(time.time() * 1000)) - current_time_millis,total_time/1000.0))   
            current_time_millis = int(round(time.time() * 1000))  

            x_in1 = img_tensor_nor1.to(device)   
            x_original1 = img_tensor1.to(device)
            x_in2 = img_tensor_nor2.to(device)   
            x_original2 = img_tensor2.to(device)

            _,decoded = autoencoder(x_in1,'A')
            # print('step: %d | time: %d' % (step,int(round(time.time() * 1000)) - current_time_millis))          
            # print('epoch: %d | step: %d | time: %d ms' % (epoch, step, int(round(time.time() * 1000)-current_time_millis)))
            loss1 = loss_func(decoded, x_original1)      
            # optimizer.zero_grad() 

            # loss.backward() 
            # optimizer.step()    

            # optimizer.zero_grad()                 
            # print('step: %d | time: %d' % (step,int(round(time.time() * 1000)) - current_time_millis))  
            # optimizer.step()  
            # print('step: %d | time: %d | epoch: %d | loss: %f' % (step,int(round(time.time() * 1000)) - current_time_millis,epoch, loss.data.cpu().numpy()))    

            _,decoded = autoencoder(x_in2,'B')
            loss2 = loss_func(decoded, x_original2)      
            optimizer.zero_grad()
            loss = (loss2+loss1)/ 2                    
            loss.backward()   

            optimizer.step()  

            print('step: %d | time: %d | epoch: %d | loss: %f' % (step,int(round(time.time() * 1000)) - current_time_millis,epoch, loss.data.cpu().numpy()))          
    
            if step % 20 == 0:
                print('Epoch: ', epoch, '| train loss: %f' % loss.data.cpu().numpy())                
                # plotting decoded image (second row)
                with torch.no_grad():
                    _,decoded_data = autoencoder(torch.stack(dataset_train.imgs1_tensor_nor[:N_TEST_IMG],dim=0).to(device),'A')
                    decoded_data = decoded_data.cpu()
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    a[1][i].imshow(decoded_data[i].permute(1,2,0).numpy().astype(np.float32),vmin=0, vmax=1)
                    a[1][i].set_xticks(())
                    a[1][i].set_yticks(())
                with torch.no_grad():    
                    _,decoded_data = autoencoder(torch.stack(dataset_train.imgs1_tensor_nor[:N_TEST_IMG],dim=0).to(device),'B')
                    decoded_data = decoded_data.cpu()
                for i in range(N_TEST_IMG):
                    a[2][i].clear()
                    a[2][i].imshow(decoded_data[i].permute(1,2,0).numpy().astype(np.float32),vmin=0, vmax=1)
                    a[2][i].set_xticks(())
                    a[2][i].set_yticks(())

                with torch.no_grad():
                    _,decoded_data = autoencoder(torch.stack(dataset_train.imgs2_tensor_nor[:N_TEST_IMG],dim=0).to(device),'B')
                    decoded_data = decoded_data.cpu()
                for i in range(N_TEST_IMG):
                    a[4][i].clear()
                    a[4][i].imshow(decoded_data[i].permute(1,2,0).numpy().astype(np.float32),vmin=0, vmax=1)
                    a[4][i].set_xticks(())
                    a[4][i].set_yticks(())
                with torch.no_grad():    
                    _,decoded_data = autoencoder(torch.stack(dataset_train.imgs2_tensor_nor[:N_TEST_IMG],dim=0).to(device),'A')
                    decoded_data = decoded_data.cpu()
                for i in range(N_TEST_IMG):
                    a[5][i].clear()
                    a[5][i].imshow(decoded_data[i].permute(1,2,0).numpy().astype(np.float32),vmin=0, vmax=1)
                    a[5][i].set_xticks(())
                    a[5][i].set_yticks(())
                plt.pause(0.1)
    
        if (epoch+1) % 5 == 0 and SAVE_MODEL:
            torch.save(autoencoder.state_dict(),WEIGHT_PATH)
    # save the weights of model
    if EPOCH > 0:
        if SAVE_MODEL:
                torch.save(autoencoder.state_dict(),WEIGHT_PATH)
    else:
        # plotting decoded image (second row)
        with torch.no_grad():
            # view_data2 = torch.from_numpy(view_data.transpose(0,3,1,2))
            _, decoded_data = autoencoder(torch.stack(dataset_train.imgs_tensor_nor[1][:N_TEST_IMG],dim=0).to(device))
        for i in range(N_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(decoded_data[i].cpu().permute(1,2,0))
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.pause(0.02) 
    plt.ioff() 
    plt.show()

    input("请输入任意键结束...")

if __name__ == "__main__":
    _main()