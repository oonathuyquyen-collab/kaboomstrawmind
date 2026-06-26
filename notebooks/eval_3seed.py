import os,sys,subprocess,glob,traceback
subprocess.check_call([sys.executable,"-m","pip","install","-q","torch==2.5.1","torchvision==0.20.1","--index-url","https://download.pytorch.org/whl/cu121"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","ultralytics","--no-deps"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","ultralytics-thop","--no-deps"])
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import CBAM
tasks.CBAM=CBAM
cands=glob.glob('/kaggle/input/**/train/images',recursive=True)
src=os.path.dirname(os.path.dirname([c for c in cands if 'lo3' not in c][0]))
DATA='/kaggle/working/data.yaml'
open(DATA,'w').write(f"path: {src}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 3\nnames:\n  0: Healthy\n  1: Trichoderma\n  2: Aspergillus\n")
weights=sorted(glob.glob('/kaggle/input/**/weights/best.pt',recursive=True))
print("found weights:",len(weights))
seen=set()
for w in weights:
    name=os.path.basename(os.path.dirname(os.path.dirname(w)))  # folder above /weights
    if name in seen: continue
    seen.add(name)
    try:
        m=YOLO(w)
        mt=m.val(data=DATA,split='test',project='/kaggle/working/ev',name=name,verbose=False)
        line=f">>> EVAL2 {name} mAP50={mt.box.map50:.4f} mAP5095={mt.box.map:.4f} P={mt.box.mp:.4f} R={mt.box.mr:.4f}"
        for i,cn in enumerate(['Healthy','Trichoderma','Aspergillus']):
            line+=f" {cn}={mt.box.ap50[i]:.4f}"
        print(line)
    except Exception as e:
        print(f">>> EVAL2 {name} ERROR {e}")
print(">>> EVAL2 ALL DONE")
