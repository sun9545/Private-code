'''
用于修改image的size，便于后期Unet的训练
'''
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=512,height=512):
    img=Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        if new_img.mode == 'P':
            new_img = new_img.convert("RGB")
        if new_img.mode == 'RGBA':
            new_img = new_img.convert("RGB")
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

for jpgfile in glob.glob("/home/sjq/img_data/*.jpg"):
    convertjpg(jpgfile,"/home/sjq/img/query/")

