import os,sys,subprocess,glob,traceback
subprocess.check_call([sys.executable,"-m","pip","install","-q","torch==2.5.1","torchvision==0.20.1","--index-url","https://download.pytorch.org/whl/cu121"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","ultralytics","--no-deps"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","ultralytics-thop","--no-deps"])
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import CBAM
tasks.CBAM=CBAM
print("cuda",torch.cuda.is_available(),torch.cuda.get_device_name(0))
cands=glob.glob('/kaggle/input/**/train/images',recursive=True)
src=os.path.dirname(os.path.dirname(cands[0]))
DATA='/kaggle/working/data.yaml'
open(DATA,'w').write(f"path: {src}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 3\nnames:\n  0: Healthy\n  1: Trichoderma\n  2: Aspergillus\n")
open('/kaggle/working/cbam.yaml','w').write('''nc: 3
scales:
  n: [0.33, 0.25, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]]
  - [15, 1, CBAM, [64]]
  - [18, 1, CBAM, [128]]
  - [21, 1, CBAM, [256]]
  - [[22, 23, 24], 1, Detect, [nc]]
''')
def report(tag,mt):
    line=f">>> ABL {tag} mAP50={mt.box.map50:.4f} mAP5095={mt.box.map:.4f} P={mt.box.mp:.4f} R={mt.box.mr:.4f}"
    for i,cn in enumerate(['Healthy','Trichoderma','Aspergillus']):
        line+=f" {cn}={mt.box.ap50[i]:.4f}"
    print(line)
def run(tag,use_cbam,cp):
    print(f"\n>>> RUN {tag} cbam={use_cbam} cp={cp}")
    m=YOLO('/kaggle/working/cbam.yaml') if use_cbam else YOLO('yolov8n.pt')
    if use_cbam:
        try: m.load('yolov8n.pt')
        except Exception as e: print("warm skip",e)
    m.train(data=DATA,epochs=100,imgsz=640,batch=16,seed=42,project="runs",name=tag,save=True,plots=False,device=0,copy_paste=cp)
    report(tag,m.val(split='test',project="runs",name=f"{tag}_t"))
for tag,cb,cp in [("abl_yolov8n_cp0",False,0.0),("abl_cbam_cp0",True,0.0)]:
    try: run(tag,cb,cp)
    except Exception as e: print(f"ERROR {tag}: {e}"); traceback.print_exc()
print("\n>>> LO 4 COMPLETE")
