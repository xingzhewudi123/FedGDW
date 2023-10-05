from __future__ import absolute_import, division, print_function
import hashlib
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import time
import dill
import json
import copy
import math
import random
import struct
import binascii
import copy
import torch
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.7
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import io
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys
sys.setrecursionlimit(6000)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#############################  FedGDW  ###########################################

# key = (x0, y0). 
x0 = 0.1876756434213423459
y0 = 0.3564897832154658847

key = [x0, y0]

def gethash(data): 
    hash_object = hashlib.sha256() 
    hash_object.update(data) 
    hq = hash_object.hexdigest()

    hs = float(int(hq[:21], 16))/15**20
    hash_value = hs-math.floor(hs)

    return hash_value 


def TwoD_LICM(hash_value, key, len): 
    a = 0.6; k = 0.8
    T=len
 
    x0 = key[0] + hash_value - math.floor(key[0] + hash_value)
    y0 = key[1] + hash_value - math.floor(key[1] + hash_value)

    x=[];y=[]
    for t in range(T):
        if (y0==1):
            y0==0.342639879728673
        xt = math.sin(21/(a*(y0+3)*k*x0*(1-k*x0)))
        yt = math.sin(21/(a*(k*xt+3)*y0*(1-y0)))

        x0,y0=xt,yt
        x.append(x0)
        y.append(y0)

    return x,y

def SequenceHat(hash_value, key, nv):
    len_hat = 2*nv
    x,y = TwoD_LICM(hash_value, key, nv)

    x = np.array(x)
    y = np.array(y)
    
    x_hat = np.floor(abs(x*10**15%(255)))        # for spread  robust watermark 
    y_hat = np.argsort(y)                        # for diffusion robust watermark

    return x_hat, y_hat

def RowColMatrix(nv): 
    r_len = int(nv**0.5)
    c_len = math.ceil(nv/ r_len)

    return r_len, c_len

def SequenceWave(key, nv_fragile):
    len_wave = math.ceil(nv_fragile/2)
    x,y= TwoD_LICM(0,key, len_wave)

    x_wave = np.array(x)     # for permuting F_m
    y_wave = np.array(y)     # for permuting F_m

    return x_wave, y_wave

def RobustWM(RWM, hash_vec, key):     
    image = Image.open(RWM)
    image_data = np.array(image)
    
    x_hat, y_hat = SequenceHat(hash_vec, key, image_data.size)
    key_stream = np.array(x_hat, dtype=np.uint8)
    # Step2: encrypt RWM_binary via spread and diffusion
    # spread
    RWM_S = np.bitwise_xor(image_data, key_stream.reshape(image_data.shape))
    RWM_S = RWM_S.reshape(image_data.size)
    np.set_printoptions(threshold=np.inf)
    # diffuse
    z_P = np.argsort(y_hat)
    RWM_D = RWM_S[z_P]

    uint8_array = RWM_D.view(np.uint8) 
    RWM_E = np.unpackbits(uint8_array)

    RWME_len = RWM_E.size

    return RWM_E,RWME_len

def DecRobustWM(RWM_E, hash_vec, key,RWM): 

    image = Image.open(RWM)
    image_data = np.array(image)
    # decrypt RWM_E via key
    x_hat, y_hat = SequenceHat(hash_vec, key, image_data.size)
    key_stream = np.array(x_hat, dtype=np.uint8)
    
    binary_string = ''.join(map(str, RWM_E))
    binary_chunks = [binary_string[i:i+8] for i in range(0, (len(binary_string)), 8)]

    RWMs_int8 = np.array([int(chunk, 2) for chunk in binary_chunks])
    np.set_printoptions(threshold=np.inf)

    # de-permutation
    z_P = np.argsort(y_hat)
    RWM_D = np.zeros(image_data.size)
    RWM_D[z_P] = RWMs_int8
    
    # de-diffusion
    RWM_uint8 = np.array(RWM_D, dtype=np.uint8)
    RWM_uint8 = np.bitwise_xor(RWM_uint8, key_stream)
    RWM_dec = RWM_uint8.reshape(image_data.shape)
    decrypted_RWM = Image.fromarray(RWM_dec)

    return decrypted_RWM

def VectoMatrix(V): 
    V_s = V.shape
    len_V = V_s[0]

    r_len = int(len_V**0.5)
    c_len = math.ceil(len_V/ r_len)

    new_c = np.zeros(r_len*c_len)
    new_c[0:len_V] = V

    V_Matrix = new_c.reshape((r_len, c_len))

    return V_Matrix, r_len, c_len

# Phase1 (DWE): embed robust watermark (RWM) into q_m (quantization gradient)
def Embed_RWM_b1(q_m, hash_vec, key, RWM, b1): 
     
    qm_s = q_m.shape
    nv = qm_s[0]
    
    #Step1: encrypt RobustWM
    RWM_E,RWME_len = RobustWM(RWM,hash_vec, key)

    R_mkv = q_m
    #Step2: embed RWM_E via fliping the least significant bit of q_m
    if b1==0:
        R_mkv = q_m
    if b1>0:
        for i in range(0,b1):
           s1 = RWM_E.shape
           s0 = s1[0]
           l = math.floor(nv/s0)
           l1 = int(s0*l)
           RWM_EN = RWM_E*2**i
           
           RWM_EN=np.array(RWM_EN,dtype=np.uint8)
           q_m=np.array(q_m,dtype=np.uint8)
           R_mkv[0:RWME_len] = np.bitwise_xor(RWM_E, q_m[0:RWME_len])
    R_mk, r_len, c_len = VectoMatrix(R_mkv) 

    return R_mk, r_len, c_len,RWME_len

def FragileWM(R_mk, r_len, c_len, b):  # generate fragile watermark
    
    code = 2**b

    # row sum of R_mk
    row_sums = np.sum(R_mk, axis=1) 
    
    # col sum of R_mk
    col_sums = np.sum(R_mk, axis=0)
    
    #print(col_sums)
    # encode fragile watermark
    n_h = math.ceil(math.log(r_len*(code-1), 2**b))
    row_sums_code = np.zeros(r_len*(n_h))

    for t in range(n_h):
       row_sums_code[r_len*t:r_len*(t+1)] = row_sums// (code**(n_h-t-1))  # quotient 
       row_sums = row_sums % (code**(n_h-t-1))  # remainder 
    #print(row_sums_code)
    n_l = math.ceil(math.log(c_len*(code-1), 2**b))
    col_sums_code = np.zeros(c_len*(n_l))

    for t in range(n_l):
       col_sums_code[c_len*t:c_len*(t+1)] = col_sums// (code**(n_l-t-1))  # quotient 
       col_sums = col_sums % (code**(n_l-t-1))  # remainder 

    #print(col_sums_code)

    fwm_vector =np.concatenate((row_sums_code, col_sums_code), axis=0) # row_sums_code + col_sums_code

    fm_s = fwm_vector.shape
    len_fm = fm_s[0]
    row_fm = math.ceil(len_fm/ c_len)
    r_len_new = row_fm + r_len
    
    fwm = np.zeros(row_fm*c_len)
    fwm[0:len_fm] = fwm_vector

    fwm = fwm.reshape(row_fm, c_len)

    return fwm,r_len_new

#Phase1(DWE): embed fWM into R_mk
def Embed_FWM(R_mk, r_len, c_len, x_wave, y_wave,b): 
    
    #Step1: generate Fragile watermark
    fwm,r_len_new = FragileWM(R_mk, r_len, c_len, b)

    #Step2: embed FWM_E via appending the last row of R_m
    F_m = np.concatenate((R_mk, fwm), axis=0)

    #Step3: encrypt F_m via permuting the positions
    x_wave1 = np.concatenate((x_wave, y_wave), axis=0) 

    y_wave1 = np.argsort(x_wave1[0:(r_len_new*c_len)])
 
    F_mk = F_m.reshape(r_len_new*c_len)[y_wave1]

    return F_mk, r_len_new

def Attack(F_mk, Attack_type, r_len_new,c_len,ar=0.5):  #Attack type: 0 remove 25%, remove 50%, 1 Guss 25%, Guss 50%, 
    Ft = F_mk
    if(Attack_type==0):
        dsa_f = F_mk.reshape(r_len_new, c_len)
        arlen=int(r_len_new*ar)
        dsa_f[0:arlen,:]  =  0
        plt.imshow(dsa_f, cmap='viridis') 
        plt.axis('off')
        plt.savefig('/Gradient_lenet_Remove_'+str(ar)+'visualization.png',bbox_inches='tight', dpi=300) 
        #plt.show()
        Ft = dsa_f.reshape(r_len_new*c_len)
    
    if(Attack_type==1):
        #dsa_f = F_mk.reshape(r_len_new, c_len)
        arlen=math.ceil(r_len_new*ar)
        pos = random.sample(range(r_len_new*c_len), arlen*c_len)
        guss_mean = 2
        guss_std = 0 
        random_float_matrix = np.random.normal(guss_mean, guss_std, size=arlen*c_len)
        random_integer = np.clip(np.round(random_float_matrix), 0, 255).astype(np.uint8)
        F_mk[pos]=random_integer
        Ft = copy.deepcopy(F_mk)
        Fmim = F_mk.reshape(r_len_new, c_len)
        plt.imshow(Fmim, cmap='viridis') 
        plt.axis('off')
        plt.savefig('/Gradient_lenet_Guss_'+str(ar)+'visualization.png', dpi=300) 
        #plt.show()

    if(Attack_type==2):
        #dsa_f = F_mk.reshape(r_len_new, c_len)
        arlen=math.ceil(c_len*ar)
        pos = random.sample(range(arlen*arlen), arlen*arlen)
        guss_mean = 2
        guss_std = 0 
        random_float_matrix = np.random.normal(guss_mean, guss_std, size=arlen*arlen)
        random_integer_vec = np.clip(np.round(random_float_matrix), 0, 255).astype(np.uint8)
        F_mk[pos]=random_integer_vec
        Ft = copy.deepcopy(F_mk)
    return Ft

def ExtractFragile(E_dsa,dsa_Rm, x_wave, y_wave, r_len_new, r_len,c_len,b):
    code = 2**b
    n_h = math.ceil(math.log(r_len*(code-1), 2**b))
    n_l = math.ceil(math.log(c_len*(code-1), 2**b))
    
    
    #Step1: decrypt E_dsa by restoring the positions of momentum
    x_wave1 = np.concatenate((x_wave, y_wave), axis=0) 
    y_wave1 = np.argsort(x_wave1[0:(r_len_new*c_len)])
    sort_reverse_y1 = np.argsort(y_wave1) 
    
    ddsa=E_dsa[sort_reverse_y1]
    dsa_f  = E_dsa[sort_reverse_y1]- dsa_Rm
    dsa_f = dsa_f.reshape(r_len_new, c_len)

    #Step2: Extact fragile 
    fmw  = dsa_f[r_len:r_len_new,:]
    fmw = fmw.reshape((r_len_new-r_len)*c_len)
    row_sums_code = fmw[0:r_len*(n_h)] 
    col_sums_code = fmw[r_len*(n_h):(r_len*(n_h)+c_len*(n_l))] 

    dsa1  = dsa_f[0:r_len,:]
    # row sum of R_mk
    hf = np.sum(dsa1, axis=1) 
    # col sum of R_mk 
    lf = np.sum(dsa1, axis=0)
    
    
    dsa  = ddsa[0:r_len*c_len]

    #Step3: decode fragile
    row_sums = np.zeros(r_len)
    col_sums = np.zeros(c_len)

    for t in range(n_h): 
       temp = row_sums_code[r_len*t:r_len*(t+1)] * (code**(n_h-t-1)) 
       #print('temp',temp)
       row_sums = row_sums + temp  

    for t in range(n_l):
       temp = col_sums_code[c_len*t:c_len*(t+1)] * (code**(n_l-t-1))
       col_sums = col_sums + temp   

    WT=(row_sums,col_sums)
    Wf=(hf,lf)
    return dsa,WT,Wf

def TamperLocate(dsa, WT,Wf):
    # generate the differential matrix
    row_len=WT[0].size
    col_len=WT[1].size
    D_matrix = np.zeros((row_len,col_len))
    D_matrix = np.array(D_matrix,dtype=np.uint8)
    Diff_Row=np.array((WT[0]-Wf[0])*10**2).astype(np.int32)
    Diff_col=((WT[1]-Wf[1])*10**2).astype(np.int32)
    
    dsa = dsa.reshape(row_len,col_len)
    #dsa = np.array(dsa,dtype=np.uint8)
    # find non zeros in Diff_Row and Diff_col
    
    Row = np.nonzero(Diff_Row)
    Col = np.nonzero(Diff_col)
    len0=np.array(Row).size
    len1=np.array(Col).size
    # locate 
    locate=(Row[0],Col[0])
    print('Row[0]',Row[0])  
    if np.all(Diff_Row == 0):
        print('there is no tamper value')   
        D_image = Image.fromarray(D_matrix)             
        D_image.save('/NonTamper_lenet.png')
    else:
        print('there are tamper values')
        print('len0',len0)
        if len0*len1==1:                      
           # locate with a cross
           center_x, center_y = Row[0][0],Col[0][0]
           cross_length = 20
           D_matrix[center_y, center_x - cross_length:center_x + cross_length + 1] = 255
           D_matrix[center_y - cross_length:center_y + cross_length + 1, center_x] = 255
           dsa[center_y, center_x - cross_length:center_x + cross_length + 1] = 0
           dsa[center_y - cross_length:center_y + cross_length + 1, center_x] = 0

           plt.imshow(dsa, cmap='viridis')
           plt.savefig('/OneTamper_lenet.png')
           D_image = Image.fromarray(D_matrix)  
           D_image.save('/OneTamperLocate_lenet.png')
        else:
           row_mark = np.random.randint(0, 256, size=(len0,col_len), dtype=np.uint8)
           col_mark = np.random.randint(0, 256, size=(row_len,len1), dtype=np.uint8)
           row_mark1 = np.random.randint(0, 1, size=(len0,col_len), dtype=np.uint8)
           col_mark1 = np.random.randint(0, 1, size=(row_len,len1), dtype=np.uint8)
           D_matrix[Row[0],:]=row_mark
           D_matrix[:,Col[0]]=col_mark

           dsa[Row[0],:]=row_mark1
           dsa[:,Col[0]]=col_mark1

           D_image = Image.fromarray(D_matrix)  
           D_image.save('/TamperLocate'+str(len0+len1)+'_lenet.png')

           plt.imshow(dsa, cmap='viridis')
           plt.savefig('/Tamper'+str(len0+len1)+'_lenet.png')

    return locate

def OwnershipVerify(Ft,qm,hash_vec, key, x_wave, y_wave, b,RWME_len,RWM,r_len,r_len_new,c_len,att,ar):  
    image = Image.open(RWM)
    original = np.array(image)
    len_origin=original.size
    #Step1:  decrypt Ft by restoring the positions
    x_wave1 = np.concatenate((x_wave, y_wave), axis=0) 
    y_wave1 = np.argsort(x_wave1[0:(r_len_new*c_len)])
    sort_reverse_y1 = np.argsort(y_wave1) 
    Rt=Ft[sort_reverse_y1]
    Rt1=np.array(Rt[0:RWME_len],dtype=np.uint8)
    qm1=np.array(qm[0:RWME_len],dtype=np.uint8)
    RWM_E = np.bitwise_xor(Rt1,qm1)

    #Step2: extract robust from b1 bit planes

    RWM_B = RWM_E & 1  # from high to low
    print('RWM_B[0:100]',RWM_B[0:100])
    dec_copyright=DecRobustWM(RWM_B[0:RWME_len], hash_vec, key, RWM)
    diffxor=np.bitwise_xor(dec_copyright,original)
    nonzero_indices = len(np.nonzero(diffxor))
    extract_rate=float(len_origin-nonzero_indices)/float(len_origin)
    print('extract_rate',extract_rate)
    if att==0:
        dec_copyright.save('/DecCopyRightRemove'+str(ar)+'_lenet.png')
    if att==1:
        dec_copyright.save('/DecCopyRightGuss'+str(ar)+'_lenet.png')
    dec_copyright.show()
    file_handle1.write('\n extract_rate '+str(extract_rate))
    return extract_rate 

def gradtovec(grad):
    vec=np.array([])
    le=len(grad)
    for i in range(0,le):
        a=grad[i]
        b = a.numpy()
        if (len(a.shape)==4):   # 
            da=int(a.shape[0])
            db=int(a.shape[1])
            dc=int(a.shape[2])
            dd=int(a.shape[3])
            b=b.reshape(da*db*dc*dd)
        if (len(a.shape)==3):   
            da=int(a.shape[0])
            db=int(a.shape[1])
            dc=int(a.shape[2])
            b=b.reshape(da*db*dc)
        if (len(a.shape)==2):   
            da=int(a.shape[0])
            db=int(a.shape[1])
            b=b.reshape(da*db)
        if (len(a.shape)==1):
            da = int(a.shape[0])
            b=b
        vec=np.concatenate((vec,b),axis=0)
    return vec

def vectograd(vec,grad):
    le=len(grad)
    #print('i',1)
    for i in range(0,le):
        a=grad[i]
        
        if (len(a.shape)==4):   # 
            #print('4 len(a.shape)')
            da=int(a.shape[0])
            db=int(a.shape[1])
            dc=int(a.shape[2])
            dd=int(a.shape[3])
            c=vec[0:da*db*dc*dd]
            c=c.reshape(da,db,dc,dd)
            #print('grad[i]',grad[i])
            #print('c',c)
            lev=len(vec)
            vec=vec[da*db*dc*dd:lev]
        if (len(a.shape)==3):   
            #print('3 len(a.shape)',len(a.shape))
            da=int(a.shape[0])
            db=int(a.shape[1])
            dc=int(a.shape[2])
            c=vec[0:da*db*dc]
            c=c.reshape(da,db,dc)
            lev=len(vec)
            vec=vec[da*db*dc:lev]
        if len(a.shape)==2:
            #print('2 len(a.shape)',len(a.shape))
            da=int(a.shape[0])
            db=int(a.shape[1])
            c=vec[0:da*db]
            c=c.reshape(da,db)
            lev=len(vec)
            vec=vec[da*db:lev]
        if len(a.shape)==1:
            #print('1 len(a.shape)',len(a.shape))
            da=int(a.shape[0])
            c=vec[0:da]
            lev = len(vec)
            vec = vec[da:lev]

        grad[i]=0*grad[i]+c
    return grad

def Get_qm(vec,v2,b): 

    Rm=max(abs(vec-v2))  
    delta=Rm/(np.floor(2**b)-1)  
    q_m = np.floor((vec-v2+Rm+delta)/(2*delta)) 
    
    vec_string = vec.tostring()
    hash_vec = gethash(vec_string)

    return q_m, hash_vec, Rm

def quantd(vec,v2,b):    # LAQ
    n=len(vec)
    Rm=max(abs(vec-v2))  
    delta=Rm/(np.floor(2**b)-1)  
    
    q_m = np.floor((vec-v2+Rm+delta)/(2*delta)) # q_m belongs to [0, 2^b-1]
    quantv=v2+2*delta*q_m-Rm # Eq. 7   # quantv is Q_m^k, v2 is Q_m^(k-1)
    #print('quantv', quantv)
    return quantv

def normalize(X_train, X_test):
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))  
    std = np.std(X_train, axis=(0, 1, 2, 3))  
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

def preprocess(x, y):
    x = tf.image.resize(x, (227, 227))  
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    #y = tf.squeeze(y, axis=1)  
    #y = tf.one_hot(y, depth=10)
    return x, y

def preprocessTE(x, y):
        
    x = tf.image.resize(x, (227, 227))  
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    #y = np.eye(10)[y]
    y = tf.squeeze(y, axis=1)  
    y = tf.one_hot(y, depth=10)
    return x, y

#########################################################################

#al=0:FedGDW;al=1:LAQ; 

#RWM = "CopyRight"
RWM = '/copyRight.png'
tic=time.time()
# 0<b1<b. b1=0, there is no robust watermark
b1=1                   # b1=0 there is no watermark.
b=2                    # 4 6 10 16 24    
M = 10;               #  10 20 30 40 50
OwnershipCheck=0     # 1: OwnershipCheck; 0: no OwnershipCheck  
TamperCheck=0       # 1: TamperCheck;    0: no TamperCheck  

datadistribution = 0  # 0: iid; 1: noniid
noniid_rate = 0.4     # 0.5 0.6 0.7 0.8 0.9 
cl=0.01 

Iter=10 
alpha= 0.08           #0.02
mr = 0.9
batchsize=128        
                  
nalg=2 
acc=np.zeros((Iter,2))

file_handle1=open('fedgdw_lenet_mnist.txt',mode='a')  
file_handle1.write('\n \n Iter: '+ str(Iter)+ ',  batchsize: '+ str(batchsize)+
                   ',  alpha: '+ str(alpha)+', M:'+str(M)+', quant b:'+str(b)+', embed b1:'+str(b1))

print('\n Iter: ', Iter , ',  batchsize: ', batchsize, ',  alpha: ',
       alpha, ',  M', M,', b:',str(b),', embed b1:',str(b1))
if (datadistribution == 0):  # iid
    file_handle1.write('\n data distribution: iid ')
    print('\n data distribution: iid ')
if (datadistribution == 1):  # noniid
    file_handle1.write('\n data distribution: non-iid rate'+str(noniid_rate))
    print('\n data distribution: noniid, noniid rate: ', noniid_rate)

for al in range(0,nalg):
    #al=0

    if(al==0):
       file_handle1.write('\n FedGDW: quantization and dual watermark')
       file_handle1.write('\n mr: '+ str(mr))
       print('FedGDW: quantization and dual watermark') 
       print('mr: ', mr)    
    if(al==1):
       file_handle1.write('\n \n LAQ: quantization but no watermark')
       print('LAQ: quantization but no watermark')
    

    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize and reshape the data
    train_images = train_images.reshape(-1, 28, 28, 1)/ 255.0
    x_test_a = test_images.reshape(-1, 28, 28, 1)/ 255.0
 
    
    Ntr=train_images.__len__()
    Nte=x_test_a.__len__()
    
    if (datadistribution == 0):   # iid
        Mi=int(Ntr/M)   
        Datatr=M*[0]
        NR = 1/M
        for m in range(0,M):
            datr=tf.data.Dataset.from_tensor_slices(
                (tf.cast(train_images[m*Mi:(m+1)*Mi], tf.float32),
                tf.cast(train_labels[m*Mi:(m+1)*Mi], tf.int32)))
                #print('shape mnist_images[m*Mi:(m+1)*Mi,tf.newaxis]', (mnist_images[m*Mi:(m+1)*Mi,tf.newaxis].shape))
            datr=datr.batch(batchsize)
            Datatr[m]=datr
    if (datadistribution == 1):  # noniid
        
        M_1 = int(M*0.1);M_2 = int(M*0.9)
        Mi_1=int((Ntr*noniid_rate)/M_1)   
        Mi_2=int((Ntr*(1-noniid_rate))/M_2)   # 
        A = int(Ntr*noniid_rate)
        NR1=Mi_1/Ntr; NR2=Mi_2/Ntr 
        Datatr=M*[0]

        for m in range(0,M):
            if m< M_1:
                m0=m
                datr=tf.data.Dataset.from_tensor_slices(
                     (tf.cast(train_images[m0*Mi_1:(m0+1)*Mi_1], tf.float32),
                     tf.cast(train_labels[m0*Mi_1:(m0+1)*Mi_1], tf.int32)))
                     #print('shape mnist_images[m*Mi:(m+1)*Mi,tf.newaxis]', (mnist_images[m*Mi:(m+1)*Mi,tf.newaxis].shape))
                datr=datr.batch(batchsize)
            if m>= M_1:
                m1=m-M_1
                datr=tf.data.Dataset.from_tensor_slices(
                     (tf.cast(train_images[A+m1*Mi_2:A+(m1+1)*Mi_2], tf.float32),
                     tf.cast(train_labels[A+m1*Mi_2:A+(m1+1)*Mi_2], tf.int32)))
                     #print('shape mnist_images[m*Mi:(m+1)*Mi,tf.newaxis]', (mnist_images[m*Mi:(m+1)*Mi,tf.newaxis].shape))
                datr=datr.batch(batchsize)
            Datatr[m]=datr

    nl=len(test_labels)
    y_test_a=np.eye(10)[test_labels]   # 

    tf.compat.v1.random.set_random_seed(1234)  # 
    lenet = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=120, activation='relu'),
    tf.keras.layers.Dense(units=84, activation='relu'),
    tf.keras.layers.Dense(units=10)
    ])
    lenet.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(alpha),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    optimizer=tf.compat.v1.train.GradientDescentOptimizer(alpha)

    le=len(lenet.trainable_variables) 
    nv=0
    for i in range (0,le):
        a = lenet.trainable_variables[i]
        if (len(a.shape)==4):   # 
            da=int(a.shape[0])
            db=int(a.shape[1])
            dc=int(a.shape[2])
            dd=int(a.shape[3])
            nv=nv+da*db*dc*dd
        if (len(a.shape)==3):   
            da=int(a.shape[0])
            db=int(a.shape[1])
            dc=int(a.shape[2])
            nv=nv+da*db*dc
        if (len(a.shape)==2):   
            da=int(a.shape[0])
            db=int(a.shape[1])
            nv=nv+da*db
        if (len(a.shape)==1):
            da=int(a.shape[0])
            nv=nv+da 
    
    r_len, c_len= RowColMatrix(nv) 
    code=2*b
    n_h = math.ceil(math.log(r_len*(code-1), 2**b))
    n_l = math.ceil(math.log(c_len*(code-1), 2**b))
    len_fm = c_len*(n_l) + r_len*(n_h)
    row_fm = math.ceil(len_fm/ c_len)
    r_len_new = row_fm + r_len
    print('model size', nv)
    nv_fragile = r_len_new*c_len

    gr=np.zeros((M,nv))  #
    mgr=np.zeros((M,nv))
    dsa=np.zeros(nv)
    mgr_a=np.zeros(nv)

    dL=np.zeros((M,nv))
    dL_M=np.zeros((M,nv))
    W_Rm=np.zeros(M)
    dsa_Rm = 0
    dsa_Rm1 = 0

    E_dL = np.zeros((M,nv_fragile))
    E_mgr = np.zeros((M,nv_fragile))
    E_gr = np.zeros((M,nv_fragile))
    E_dsa = np.zeros(nv_fragile)
    E_dsa1 = np.zeros(nv_fragile)
    
    Tao = 1/(np.floor(2**b)-1)
    if (al==0):
        x_wave, y_wave = SequenceWave(key, nv_fragile)
    for k in range(0,1):  # Iter        
        for m in range(0,M): # M: client number
            Datatr[m] = Datatr[m].shuffle(100)
            for (batch, (images,labels)) in enumerate(Datatr[m].take(50)):
                if batch>0:
                    optimizer.apply_gradients(zip(grads, lenet.trainable_variables),
                                global_step=tf.compat.v1.train.get_or_create_global_step())
                with tf.GradientTape() as tape: 
                    logits=lenet(images, training=True) 
                    loss_value = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits)
                    if (al==1):
                        for i in range(0,len(lenet.trainable_variables)):
                           if i==0:
                               l2_loss=cl*tf.nn.l2_loss(lenet.trainable_variables[i])
                           if i>=1:
                               l2_loss=l2_loss+cl*tf.nn.l2_loss(lenet.trainable_variables[i])
                    
                        loss_value = loss_value +l2_loss
                    
                grads = tape.gradient(loss_value, lenet.trainable_variables)
            vec=gradtovec(grads)
            
            if (al==0 and b1>0): 
                
                s1=time.time()
                q_m, hash_vec, Rm = Get_qm(vec,mgr[m,:],b)
                qm1=copy.deepcopy(q_m)
                delta = Rm * Tao 
                gr[m,:]= 2*delta*q_m - Rm
                mgr[m,:]=mgr[m,:] + gr[m,:]
                
                R_mk, r_len, c_len,RWME_len = Embed_RWM_b1(q_m, hash_vec, key, RWM, b1)
                F_mk, r_len_new = Embed_FWM(R_mk, r_len, c_len, x_wave, y_wave,b)
                
                if (OwnershipCheck==1 and k==0 and m==0):  # k \in (0,Ier)

                    ar=0    #0.1 0.2 0.5    Tamper: 
                    att=1   #0: Remove attack; 1: Guss attack  2: Tamper attack 0.035(111) 0.02(42) 0.012(23) 0.01
                    Ft=Attack(F_mk, att, r_len_new,c_len,ar)  #0.2
                    F_mk=copy.deepcopy(Ft)
                    ER=OwnershipVerify(F_mk,qm1,hash_vec, key, x_wave, y_wave, b,RWME_len,RWM,r_len,r_len_new,c_len,att,ar)
                
                #F_mk[31]=0.001          #one value tamper
                E_gr[m,:] =  2*delta*F_mk-Rm      # de-quantized gradient 
                E_dL[m,:] = E_gr[m,:]-E_mgr[m,:]  # gradient difference
                E_dsa1 = E_dsa1 + E_dL[m,:]       # agg momentum
                E_mgr[m,:] = E_gr[m,:]
                
                Rm_dL = Rm-W_Rm[m] 
                W_Rm[m] = Rm                
                dsa_Rm1 = dsa_Rm1 + Rm_dL

                if m == M-1: 
                    dsa_Rm = mr*dsa_Rm + dsa_Rm1/M 
                    E_dsa = mr*E_dsa + E_dsa1/M    #
                    dsa,WT,Wf = ExtractFragile(E_dsa,-dsa_Rm, x_wave, y_wave, r_len_new, r_len,c_len,b) 
                    if (OwnershipCheck==2 and k==0):
                       #rmk,WT,Wf = ExtractFragile(Ft,0, x_wave, y_wave, r_len_new, r_len,c_len,b) 
                       locate=TamperLocate(dsa,WT,Wf)
                e1=time.time() 
                #print('fedgwd epoch time:', e1-s1)
                                
            if (al==0 and b1==0): 
                s1=time.time()
                gr[m,:] = quantd(vec, mgr[m,:], b) # de-quantized gradient  
                dL[m,:] = gr[m,:]-mgr[m,:]         # gradient difference
                mgr[m,:] = gr[m,:]
                dsa = mr*dsa + dL[m,:]/M       # aggregated            
                e1=time.time() 
                #print('fedgwd epoch time:', e1-s1) 
            
            if (al==1):   
                s2=time.time()              
                gr[m,:] = quantd(vec,mgr[m,:],b)       # LAQ 
                dL[m,:]=gr[m,:]-mgr[m,:]      
                mgr[m,:] = gr[m,:] 
                dsa = dsa + dL[m,:]           # server Aggregation Eq. (4)  
                e2=time.time() 
                #print('LAQ epoch time:', e2-s2)                                    
        ccgrads=vectograd(dsa, grads)   

        optimizer.apply_gradients(zip(ccgrads, lenet.trainable_variables),
                                  global_step=tf.compat.v1.train.get_or_create_global_step())

        
        acc[k]=lenet.evaluate(x_test_a,y_test_a,verbose=0)
        print('iter', k, ',  acc', acc[k])
    
    top_10_elements = np.sort(acc[:,1])[Iter-10:Iter]
    avg=np.mean(top_10_elements)
    if(al==0):
       #file_handle1.write('\n FedGDW best_Accuracy'+str(100 * acc[1].max()))
       file_handle1.write('\n FedGDW epch_Accuracy '+str(acc[:,1]))
       file_handle1.write('\n FedGDW top10_Accuracy '+str(top_10_elements))
       file_handle1.write('\n FedGDW avg of top10_Accuracy '+str(avg))
       print('\n FedGDW avg top10_Accuracy ',str(avg))
    if(al==1):
       file_handle1.write('\n LAQ epch_Accuracy '+str(acc[:,1]))
       file_handle1.write('\n LAQ top10_Accuracy '+str(top_10_elements))
       file_handle1.write('\n LAQ avg of top10_Accuracy '+str(avg))
       print('\n LAQ avg top10_Accuracy ', str(avg))