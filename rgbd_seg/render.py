import base64
import io
import numpy as np
from PIL import Image,ImageDraw

PALETTE=[(50,211,173),(255,182,77),(107,153,255),(240,111,167),(172,138,255),(92,211,242)]

def rgb_image(sample):
    return Image.fromarray((sample['image'].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8))

def png_uri(image):
    stream=io.BytesIO(); image.save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()

def depth_image(sample):
    d=sample['depth'][0].cpu().numpy(); v=sample['depth_valid'][0].cpu().numpy()>0
    z=np.zeros_like(d)
    if v.any():
        lo,hi=np.quantile(d[v],[.01,.99]); z[v]=np.clip((d[v]-lo)/max(hi-lo,.001),0,1)
    a=np.stack([255*(1-z),180*(1-abs(z-.5)*2),255*z],axis=-1).astype(np.uint8)
    a[~v]=(18,24,34)
    return Image.fromarray(a)

def overlay(image,masks,boxes=None,labels=None,scores=None,classes=None):
    a=np.asarray(image).copy().astype(float)
    for i,mask in enumerate(masks):
        m=np.asarray(mask)>0.5; color=np.array(PALETTE[i%len(PALETTE)])
        a[m]=.45*a[m]+.55*color
        interior=m.copy()
        interior[1:]&=m[:-1]; interior[:-1]&=m[1:]; interior[:,1:]&=m[:,:-1]; interior[:,:-1]&=m[:,1:]
        a[m&~interior]=color
    out=Image.fromarray(a.clip(0,255).astype(np.uint8)); draw=ImageDraw.Draw(out)
    if boxes is not None:
        for i,box in enumerate(boxes):
            color=PALETTE[i%len(PALETTE)]; draw.rectangle(tuple(float(x) for x in box),outline=color,width=1)
            name=str(int(labels[i])) if labels is not None else str(i+1)
            if classes and labels is not None: name=classes[int(labels[i])-1]
            text=name+(f' {float(scores[i]):.2f}' if scores is not None else '')
            if min(image.size)<256:
                # Tiny demo inputs need short IDs to keep labels off nearby masks.
                text=str(i+1)
            # Pillow's default font is deliberately ASCII for portable server rendering.
            text=text.encode('ascii','replace').decode()
            draw.text((float(box[0]),max(0,float(box[1])-10)),text,fill=color)
    return out
