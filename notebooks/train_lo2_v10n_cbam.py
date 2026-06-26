import os, sys, subprocess, glob, shutil, time, traceback
subprocess.check_call([sys.executable,"-m","pip","install","-q","torch==2.5.1","torchvision==0.20.1","--index-url","https://download.pytorch.org/whl/cu121"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","ultralytics","--no-deps"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","ultralytics-thop","--no-deps"])
import torch
from ultralytics import YOLO
print("torch",torch.__version__,"cuda",torch.cuda.is_available())
if not torch.cuda.is_available(): raise RuntimeError("No CUDA")
print("GPU",torch.cuda.get_device_name(0))

cands=glob.glob('/kaggle/input/**/train/images',recursive=True)
src=os.path.dirname(os.path.dirname(cands[0])); 
DS='/kaggle/working/ds'
if os.path.exists(DS): shutil.rmtree(DS)
shutil.copytree(src,DS)
DATA='/kaggle/working/data.yaml'
open(DATA,'w').write(f"path: {DS}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 3\nnames:\n  0: Healthy\n  1: Trichoderma\n  2: Aspergillus\n")

CBAM_YAML='''# StrawMind-CBAM: YOLOv8 + CBAM attention on each detection scale
nc: 3
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]      # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]     # 1-P2/4
  - [-1, 3, C2f, [128, True]]      # 2
  - [-1, 1, Conv, [256, 3, 2]]     # 3-P3/8
  - [-1, 6, C2f, [256, True]]      # 4
  - [-1, 1, Conv, [512, 3, 2]]     # 5-P4/16
  - [-1, 6, C2f, [512, True]]      # 6
  - [-1, 1, Conv, [1024, 3, 2]]    # 7-P5/32
  - [-1, 3, C2f, [1024, True]]     # 8
  - [-1, 1, SPPF, [1024, 5]]       # 9
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 10
  - [[-1, 6], 1, Concat, [1]]                     # 11
  - [-1, 3, C2f, [512]]                           # 12
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 13
  - [[-1, 4], 1, Concat, [1]]                     # 14
  - [-1, 3, C2f, [256]]                           # 15 (P3)
  - [-1, 1, Conv, [256, 3, 2]]                    # 16
  - [[-1, 12], 1, Concat, [1]]                    # 17
  - [-1, 3, C2f, [512]]                           # 18 (P4)
  - [-1, 1, Conv, [512, 3, 2]]                    # 19
  - [[-1, 9], 1, Concat, [1]]                     # 20
  - [-1, 3, C2f, [1024]]                          # 21 (P5)
  - [15, 1, CBAM, [7]]                            # 22 CBAM-P3
  - [18, 1, CBAM, [7]]                            # 23 CBAM-P4
  - [21, 1, CBAM, [7]]                            # 24 CBAM-P5
  - [[22, 23, 24], 1, Detect, [nc]]              # 25
'''
open('/kaggle/working/yolov8n-cbam.yaml','w').write(CBAM_YAML)
SEED=42
def report(tag,mt):
    print(f">>> {tag} TEST mAP50={mt.box.map50:.4f} mAP50-95={mt.box.map:.4f} P={mt.box.mp:.4f} R={mt.box.mr:.4f}")
    for i,cn in enumerate(['Healthy','Trichoderma','Aspergillus']):
        try: print(f"    {cn}: mAP50={mt.box.ap50[i]:.4f}")
        except Exception: pass

def run_pt(tag,weights):
    print(f"\n>>> TRAIN {tag} seed{SEED}")
    m=YOLO(weights); t0=time.time()
    m.train(data=DATA,epochs=100,imgsz=640,batch=16,seed=SEED,project="runs",name=f"{tag}_seed{SEED}",save=True,plots=True,device=0,copy_paste=0.5)
    print(f">>> {tag} trained {time.time()-t0:.1f}s")
    report(tag, m.val(split='test',project="runs",name=f"{tag}_seed{SEED}_test"))

def run_cbam():
    tag="strawmind_cbam"
    print(f"\n>>> TRAIN {tag} seed{SEED}")
    m=YOLO('/kaggle/working/yolov8n-cbam.yaml')
    try: m.load('yolov8n.pt'); print("warm-started from yolov8n.pt")
    except Exception as e: print("warm-start skipped:",e)
    t0=time.time()
    m.train(data=DATA,epochs=100,imgsz=640,batch=16,seed=SEED,project="runs",name=f"{tag}_seed{SEED}",save=True,plots=True,device=0,copy_paste=0.5)
    print(f">>> {tag} trained {time.time()-t0:.1f}s")
    report(tag, m.val(split='test',project="runs",name=f"{tag}_seed{SEED}_test"))

for fn,a in [(run_pt,("yolov10n","yolov10n.pt")),(run_cbam,None)]:
    try:
        run_pt(*a) if a else run_cbam()
    except Exception as e:
        print(f"Error: {e}"); traceback.print_exc()
print("\n>>> LO 2 COMPLETE")
