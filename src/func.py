from .utils import *
import numpy as np
from PIL import ImageColor
from einops import rearrange

from icecream import ic
from matplotlib import pyplot as plt
def draw_attention_1d(len_h,border=3):
    
    color=np.array(ImageColor.getcolor("red",'RGBA'))/255
    tmp=np.ones((len_h,dpi,dpi,4))*color.reshape((1,1,1,4))
    alpha=np.random.rand(len_h).reshape(len_h,1,1)
    ic(alpha)
    tmp[:,:,:,-1]=alpha
    ic(tmp.shape)
    tmp=tmp.reshape(-1,dpi,4)
    new_H=len_h*dpi+border*(len_h+1)
    new_W=dpi+border*2
    final=np.ones((new_H,new_W,4))*(0,0,0,1)
    slices_H=tuple([slice((border+dpi)*i+border,(border+dpi)*(i+1)) for i in range(len_h)])
    ic(slices_H)
    ic(final.shape)
    final[np.r_[slices_H],border:border+dpi]=tmp
    '''
    for i in range(len):
        plt.subplot(1,len,i+1)
        plt.imshow(tmp[i])
        plt.axis('off')
    '''
    plt.imshow(final)
    plt.axis('off')
    #plt.show()

def draw_attention_2d(len_h,len_w,border=3):
    
    color=np.array(ImageColor.getcolor("red",'RGBA'))/255
    tmp=np.ones((len_h,len_w,dpi,dpi,4))*color.reshape((1,1,1,1,4))
    alpha=np.random.rand(len_h,len_w).reshape(len_h,len_w,1,1)
    tmp[:,:,:,:,-1]=alpha
    ic(tmp.shape)
    tmp=rearrange(tmp,'H W D1 D2 C -> (H D1) (W D2) C')
    #tmp=tmp.reshape(len_h*dpi,len_w*dpi,4)
    ic(tmp.shape)
    new_H=len_h*dpi+border*(len_h+1)
    new_W=len_w*dpi+border*(len_w+1)
    final=np.ones((new_H,new_W,4))*(0,0,0,1)
    ic(final.shape)
    slices_H=tuple([slice((border+dpi)*i+border,(border+dpi)*(i+1)) for i in range(len_h)])
    slices_W=tuple([slice((border+dpi)*j+border,(border+dpi)*(j+1)) for j in range(len_w)])
    ic(np.r_[slices_H])
    ic(final[np.r_[slices_H].reshape(-1,1),np.r_[slices_W]].shape)
    final[np.r_[slices_H].reshape(-1,1),np.r_[slices_W]]=tmp

    '''
    for i in range(len):
        plt.subplot(1,len,i+1)
        plt.imshow(tmp[i])
        plt.axis('off')
    '''
    plt.imshow(final)
    plt.axis('off')
    plt.show()

if __name__=='__main__':
    #draw_attention_1d(10)
    draw_attention_2d(10,10)