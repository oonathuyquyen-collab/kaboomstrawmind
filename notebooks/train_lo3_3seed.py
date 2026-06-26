import os,sys,subprocess,glob,shutil,time,traceback
subprocess.check_call([sys.executable,"-m","pip","install","-q","torch==2.5.1","torchvision==0.20.1","--index-url","https://download.pytorch.org/whl/cu121"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","ultralytics","--no-deps"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","ultralytics-thop","--no-deps"])
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import CBAM
tasks.CBAM=CBAM
print("torch",torch.__version__,"cuda",torch.cuda.is_available(),torch.cuda.get_device_name(0))
cands=glob.glob('/kaggle/input/**/train/images',recursive=True)
src=os.path.dirname(os.path.dirname(cands[0]))
DS='/kaggle/working/ds'
if os.path.exists(DS): shutil.rmtree(DS)
shutil.copytree(src,DS)
DATA='/kaggle/working/data.yaml'
open(DATA,'w').write(f"path: {DS}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 3\nnames:\n  0: Healthy\n  1: Trichoderma\n  2: Aspergillus\n")
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
def report(tag,seed,mt):
    print(f">>> RESULT {tag} seed{seed} TEST mAP50={mt.box.map50:.4f} mAP50-95={mt.box.map:.4f} P={mt.box.mp:.4f} R={mt.box.mr:.4f}")
    for i,cn in enumerate(['Healthy','Trichoderma','Aspergillus']):
        try: print(f"    {tag} seed{seed} {cn}: mAP50={mt.box.ap50[i]:.4f}")
        except Exception: pass
def train_one(tag,seed):
    print(f"\n>>> TRAIN {tag} seed{seed}")
    if tag=='strawmind_cbam':
        m=YOLO('/kaggle/working/cbam.yaml')
        try: m.load('yolov8n.pt')
        except Exception as e: print("warm-start skip",e)
    else:
        m=YOLO({'yolov8n':'yolov8n.pt','yolov8s':'yolov8s.pt','yolov10n':'yolov10n.pt'}[tag])
    t0=time.time()
    m.train(data=DATA,epochs=100,imgsz=640,batch=16,seed=seed,project="runs",name=f"{tag}_seed{seed}",save=True,plots=False,device=0,copy_paste=0.5)
    print(f">>> {tag} seed{seed} trained {time.time()-t0:.1f}s")
    report(tag,seed,m.val(split='test',project="runs",name=f"{tag}_seed{seed}_test"))
for seed in [123,2026]:
    for tag in ['yolov8n','yolov8s','yolov10n','strawmind_cbam']:
        try: train_one(tag,seed)
        except Exception as e: print(f"ERROR {tag} seed{seed}: {e}"); traceback.print_exc()
print("\n>>> LO 3 COMPLETE")
