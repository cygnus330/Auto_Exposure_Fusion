from ep import align_images as img_set
from ep import exposure_fusion as exp_fus
import cv2
import numpy as np

print(
'''

  _____                                        _____          _             
 | ____|_  ___ __   ___  ___ _   _ _ __ ___   |  ___|   _ ___(_) ___  _ __  
 |  _| \ \/ / '_ \ / _ \/ __| | | | '__/ _ \  | |_ | | | / __| |/ _ \| '_ \ 
 | |___ >  <| |_) | (_) \__ \ |_| | | |  __/  |  _|| |_| \__ \ | (_) | | | |
 |_____/_/\_\ .__/ \___/|___/\__,_|_|  \___|  |_|   \__,_|___/_|\___/|_| |_|
            |_|                                                            
                                                                                                                                                         
made by sivcde0405 (AgCl)
now version 1.0.0

version 1.0.0 : can do Exposure Fusion


'''
)

#input 3more img (same res)
images = []
print("input image len")
n = int(input())
print("do you wanna input exposure times?")
print("want : 1 / not want : 0")
a = int(input())
if(a == 1):
    print("input exposure times (format: 10 3 1 0.5 0.1 0.0001 ...)")
    exp_times = np.array(list(map(float, input().split())), dtype=float32)
for i in range(n):
    print(f"\rnow reading img{i+1}/{n}.jpg", end="")
    images.append(cv2.imread(f"input/img{i}.jpg"))
print("\rreading complete\n")

#align images
aligned_images = img_set(images, n)
print("mapping complete\n")

if(a == 1):
    fusion = exp_fus(aligned_images, depth=4, time_decay=exp_times)
else:
    fusion = exp_fus(aligned_images, depth=4)
cv2.imwrite("output/output.jpg", fusion)
print("\rfusion complete")