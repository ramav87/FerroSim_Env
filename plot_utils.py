import numpy as np
import matplotlib.pyplot as plt
import math
#finally some plot utils
def return_move_arrow(defect_pos,qvals):
  action = np.argmax(qvals)
  if action ==0: 
    idx =0
    mov_dir = 4

  num_moves = 4 #four ways to move, i.e. Up, down, left or right
  mov_dir_text = ['Left', 'Up', 'Right', 'Down', 'Stay']
  
  idx = math.floor((action-1)/((defect_pos.shape[0])+1)) 
  mov_dir = (action-1)%num_moves

  if mov_dir ==0: arrow = [0,0]
  elif mov_dir==1: arrow = [0,0]#[-1,0]
  elif mov_dir==2: arrow = [0,0]
  elif mov_dir==3: arrow = [0,0]
  elif mov_dir==4: arrow = [0,0]
  
  #print('idx {} and mov_dir is {} direction {}'.format(idx, mov_dir, mov_dir_text[mov_dir]))
  defect_position = (defect_pos[idx,0],defect_pos[idx,1]) 
  full_arrow = (defect_pos[idx,1],defect_pos[idx,0],arrow[0], arrow[1])

  return full_arrow

