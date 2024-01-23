import numpy as np
import os
from PIL import Image

np.random.seed(222) 


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

    create_folder_if_not_exists(os.path.join('results',dir,str(num_pattern),'top','cmap_th',str(th)))
    create_folder_if_not_exists(os.path.join('results',dir,str(num_pattern),'top','cmap_th_max',str(th)))


    image = Image.open(os.path.join('results',dir,str(num_pattern),'colormap',savename+'.png'))
    array = np.array(image)
    
    th_arr = th_array(array, th)
    th_arr_max = np.where(th_arr != 0, 1, th_arr)
    img = Image.fromarray((th_arr * 255).astype(np.uint8))
    img_max = Image.fromarray((th_arr_max * 255).astype(np.uint8))
    #%のとき
    img.save(os.path.join('results',dir,str(num_pattern),'top','cmap_th',str(th),savename+".png"))
    img_max.save(os.path.join('results',dir,str(num_pattern),'top','cmap_th_max',str(th),savename+".png"))

def merge(svdir,num_pattern,th,savename):
    ori=Image.open(os.path.join('results',svdir,str(num_pattern),'ori',savename+'.png'))
    mask=Image.open(os.path.join('results',svdir,str(num_pattern),'top/cmap_th_max',str(th),savename+'.png'))
    colormap=Image.open(os.path.join('results',svdir,str(num_pattern),'colormap',savename+'.png'))

    orimask=Image.blend(ori,mask,0.5)
    save_dir=os.path.join('results',svdir,str(num_pattern),'merge')
    orimask.save(os.path.join(save_dir,str(th)+'.png'))


    oricmap=Image.blend(ori,colormap,0.5)
    save_dir=os.path.join('results',svdir,str(num_pattern),'merge')
    oricmap.save(os.path.join(save_dir,'all'+'.png'))

    

def main():
  ####param####
  num_pattern=400  #重回帰分析のパターン数
  name='broccoli'
  img=f'{name}.jpg' #入力画像名
  svdir=f'{name}'
  savename='test_output'
  ths=[0.10,0.20] #上位何％を表示するか
  #############


  for th in ths:
    display_top(th,svdir,num_pattern,savename)
    merge(svdir,num_pattern,th,savename)

main()