import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pickle
import os
import numpy as np
import cv2

def get_frames(filename, n_frames= 1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    frame_dims = np.array([224,224,3])
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frame = cv2.resize(frame,(frame_dims[0],frame_dims[1]))
            frames.append(frame)
    v_cap.release()
    return frames, v_len

video_list = []
class_list = ['Quarrelling','Meeting','Sports','Funeral','Show','Group_Activities','Protest','Casual_family_friends_gathering','Celebration_Party','Fighting']
# train_videos_path = '/home/aarti9/VGAF/TrainAll'
val_videos_path = '/home/aarti9/VGAF/ValAll'
# test_videos_path = '/home/aarti9/VGAF/TestAll'

# train_videos_frames_path = '/scratch/aarti9/Train_16_Frames'
val_videos_frames_path = '/scratch/aarti9/Val_16_Frames'
# test_videos_frames_path = '/scratch/aarti9/Test_16_Frames'

for (root, dirs, files) in os.walk(val_videos_path): #  train_videos_path - CHANGE -
  for file in files:
      fullpath = os.path.join(root, file)
      if ('.mp4' in fullpath):
          video_list.append(fullpath)
video_list = np.asarray(video_list)

for video_id in range(len(video_list)):
  video_name = video_list[video_id].split('/')[-1].split('.')[0]
  video_label = video_list[video_id].split('/')[-2]
  video_frames, len_ = get_frames(video_list[video_id], n_frames = 63)
  video_frames = np.asarray(video_frames)
  video_frames = video_frames/255  
  class_id_loc = class_list.index(video_label)
  label = class_id_loc
  d = torch.as_tensor(np.array(video_frames).astype('float'))
  l = torch.as_tensor(np.array(label).astype('float'))
  with open(val_videos_frames_path+'/'+str(video_name)+'.pkl','wb') as f: # val_videos_frames_path - CHANGE
    pickle.dump([d,l], f, protocol=4)
