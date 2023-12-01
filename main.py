import numpy as np
import skimage.io
import skimage.segmentation
import copy
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import os
from PIL import Image
from keras.applications import inception_v3 as inc_net
from keras.applications.imagenet_utils import decode_predictions

np.random.seed(222) 

#make dir for malcc
def mkdir(svdir_pattern):

  os.makedirs(svdir_pattern, exist_ok=True) #make save dir

  # save dir の中身のリスト
  directories = ["colormap", "mask", "merge", "ori", "superpixel","top"]

  # save dir の中身を作成
  for directory in directories:
      os.makedirs(os.path.join(svdir_pattern,directory), exist_ok=True)

#マスク画像生成関数
def perturb_image(img,perturbation,segments):

  mask = np.zeros(img.shape)
  mk=copy.deepcopy(segments[:,:,np.newaxis])
  
  for i in range(len(perturbation)):
    mask=np.where(mk==i,perturbation[i],mask)

  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask

  return perturbed_image

#コサイン距離計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#rgb-limeを定義
def malcc(img, svdir, savename, num_pattern,mode='sp',block=5):
  #学習済みモデル読み込み
  model = inc_net.InceptionV3() #Load pretrained model

  #画像読み込みと前処理
  img_pil = tf.keras.preprocessing.image.load_img(img, target_size=(299, 299))
  Xi=np.array(img_pil)/255
  skimage.io.imsave(svdir+'/ori/'+savename+'.png',Xi*255) #save original img

  #推論
  preds = model.predict(Xi[np.newaxis,:,:,:])
  for x in decode_predictions(preds)[0]:
    print(x)
  #print(preds) #Top 5 classes
  top_pred_classes = preds[0].argsort()[-5:][::-1]
  print(top_pred_classes,preds[0,top_pred_classes[0]])
  #print(preds[0,top_pred_classes[0]])

  #正方形分け
  if mode=='sp':
    #スーパーピクセルとその数
    superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)
  elif mode=='sq':
    width, height = 299,299
    # ゼロ配列の作成
    seg = np.zeros((height, width), dtype=np.uint8)
    re_width=width
    seg_w=round(re_width/block)
    seg_h=round(height/block)
    k=0
    for i in range(block):
        for j in range(block):
            seg[seg_h*i:seg_h*(i+1)+1,seg_w*j+1:seg_w*(j+1)+2]=k
            k=k+1
    
    superpixels=seg
     
  num_superpixels = np.unique(superpixels).shape[0]
  #print(num_superpixels)
  skimage.io.imsave(svdir+'/superpixel/'+savename+'.png',skimage.segmentation.mark_boundaries(Xi*255, superpixels)) #save superpixel img

  #マスク画像を任意のパターンランダムに用意
  num_perturb = num_pattern
  perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels,3))
  skimage.io.imsave(svdir+'/mask/'+savename+'.png',perturb_image(Xi*255,perturbations[0],superpixels)) #save mask img

  #推論を回す
  predictions = []
  for pert in perturbations:
    perturbed_img = perturb_image(Xi,pert,superpixels)
    pred = model.predict(perturbed_img[np.newaxis,:,:,:])
    predictions.append(pred)

  predictions = np.array(predictions)

  #元のマスクされていない画像
  original_image = np.ones((num_superpixels,3)) #Perturbation with all superpixels enabled 

  #nマスク画像と元画像のコサイン距離の計算
  distances=[]
  for i in range(num_perturb):
      distance=cos_sim(perturbations[i].flatten(),original_image.flatten())
      distances.append(distance)
  distances=np.array(distances)
  #print(distances[0:10])
  kernel=np.mean(distances)
  

  #カーネル関数の重み決定
  kernel_width = kernel
  weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function

  #元画像の予測トップスコアとマスク画像の推論スコアの差
  ps=np.resize(perturbations,(num_perturb,num_superpixels*3))
  #print(ps.shape)

  class_to_explain = top_pred_classes[0]
  simpler_model = LinearRegression()
  simpler_model.fit(X=ps, y=predictions[:,:,class_to_explain], sample_weight=weights)
  coeff = simpler_model.coef_[0]
  #回帰係数
  #print(coeff)

  return superpixels,coeff

def coeff2map(sp,coeff,img_shape):
    min = np.min(coeff)
    max = np.max(coeff)
    n_coeff = np.interp(coeff, (min, max), (0, 1))
    #print('normalized_coeff: ',n_coeff)

    map = np.zeros(img_shape)
    base=copy.deepcopy(sp[:,:,np.newaxis])

    for i in range(len(n_coeff)):
        map=np.where(base==i,n_coeff[i],map)
    
    return map

def map2img(map):
    img = Image.fromarray((map * 255).astype(np.uint8))
    #img.save('coeff2imgtest.png')
    return img

def coeff2img(svdir, num_pattern,sp,coeff,savename):
  dir=svdir
  times=str(num_pattern)

  img_shape=[299,299,3]

  #print('sp:',sp)
  coeff=coeff.reshape(int(len(coeff)/3),3)
  #print('coeff:',coeff)

  map=coeff2map(sp,coeff,img_shape)
  #print(map)
  img=map2img(map)

  img.save(os.path.join(dir, times, 'colormap',savename+'.png'))

def th_array(array, n):
    # 配列をフラット化して要素をソートする
    flat_array = array.flatten()
    sorted_array = np.sort(flat_array)

    # 上位n%の閾値を計算する
    threshold_index = int((1 - n) * len(sorted_array))
    threshold_value = sorted_array[threshold_index]

    # 閾値未満の要素を0に設定する
    thresholded_array = np.where(array < threshold_value, 0, array)

    return thresholded_array

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("フォルダが作成されました:", folder_path)
    else:
        print("フォルダは既に存在します:", folder_path)

#スーパーピクセルの領域数を取得
def sp_num(sp_dir):
    array=np.load(sp_dir)
    unique_values, counts = np.unique(array, return_counts=True)
    #print(len(counts))
    return len(counts)

def display_top(th,svdir,num_pattern,savename):
    dir=svdir

    create_folder_if_not_exists(os.path.join(dir,str(num_pattern),'top','cmap_th',str(th)))
    create_folder_if_not_exists(os.path.join(dir,str(num_pattern),'top','cmap_th_max',str(th)))


    image = Image.open(os.path.join(dir,str(num_pattern),'colormap',savename+'.png'))
    array = np.array(image)
    
    th_arr = th_array(array, th)
    th_arr_max = np.where(th_arr != 0, 1, th_arr)
    img = Image.fromarray((th_arr * 255).astype(np.uint8))
    img_max = Image.fromarray((th_arr_max * 255).astype(np.uint8))
    #%のとき
    img.save(os.path.join(dir,str(num_pattern),'top','cmap_th',str(th),savename+".png"))
    img_max.save(os.path.join(dir,str(num_pattern),'top','cmap_th_max',str(th),savename+".png"))

def merge(svdir,num_pattern,th,savename):
    ori=Image.open(os.path.join(svdir,str(num_pattern),'ori',savename+'.png'))
    mask=Image.open(os.path.join(svdir,str(num_pattern),'top/cmap_th_max',str(th),savename+'.png'))
    colormap=Image.open(os.path.join(svdir,str(num_pattern),'colormap',savename+'.png'))

    orimask=Image.blend(ori,mask,0.5)
    save_dir=os.path.join(svdir,str(num_pattern),'merge')
    orimask.save(os.path.join(save_dir,str(th)+'.png'))


    oricmap=Image.blend(ori,colormap,0.5)
    save_dir=os.path.join(svdir,str(num_pattern),'merge')
    oricmap.save(os.path.join(save_dir,'all'+'.png'))

    

def main():
  ####param####
  num_pattern=500  #重回帰分析のパターン数
  name='angora3'
  img=f'images/{name}.jpg' #入力画像名
  svdir=f'results/{name}'
  savename='test_output'
  ths=[0.05,0.03,0.01,0.005,0.001] #上位何％を表示するか
  #############

  svdir_pattern=os.path.join(svdir,str(num_pattern))
  mkdir(svdir_pattern)
  sp,coeff=malcc(img, svdir_pattern, savename, num_pattern,mode='sq',block=7)
  coeff2img(svdir, num_pattern,sp,coeff,savename)
  for th in ths:
    display_top(th,svdir,num_pattern,savename)
    merge(svdir,num_pattern,th,savename)

main()