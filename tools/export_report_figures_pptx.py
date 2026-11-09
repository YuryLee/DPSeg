#!/usr/bin/env python3
"""Export real report figure assets as editable PowerPoint objects.

All text is native DrawingML with SimSun East Asian and Times New Roman Latin
typefaces. Images contain source RGB/depth/masks only, never flattened labels.
"""
from __future__ import annotations
import io
import json
import os
import re
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR','/tmp/dpseg-report-matplotlib')
sys.path.insert(0,str(ROOT))
if Path('/tmp/dpseg_ppt_deps').exists():sys.path.insert(0,'/tmp/dpseg_ppt_deps')
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn
from matplotlib import patches
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from tools import build_report_figures_v2 as source

OUT=ROOT/'outputs/report_figures_pptx_v1'
OUT.mkdir(exist_ok=True)
LATIN='Times New Roman';CJK='宋体'
SW,SH=18.,10.125
EMU=914400
SLIDE_COUNTS=[]

def font_spec(properties):
    properties.set('lang','zh-CN');properties.set('altLang','en-US')
    for tag,name in [('latin',LATIN),('ea',CJK),('cs',LATIN)]:
        child=properties.find(qn('a:'+tag))
        if child is None:child=OxmlElement('a:'+tag);properties.append(child)
        child.set('typeface',name)

def solid_color(rgb):
    from matplotlib.colors import to_rgba
    rgba=to_rgba(rgb)
    return RGBColor(*[int(round(v*255)) for v in rgba[:3]]),rgba[3]

def opacity(solid,alpha):
    if alpha>=.999:return
    c=solid.find(qn('a:srgbClr'))
    if c is not None:
        a=OxmlElement('a:alpha');a.set('val',str(int(round(alpha*100000))));c.append(a)

def style_shape(shape,face=None,edge=None,width=.7,dashed=False):
    if face is None or solid_color(face)[1]<=0:shape.fill.background()
    else:
        c,a=solid_color(face);shape.fill.solid();shape.fill.fore_color.rgb=c
        opacity(shape._element.spPr.find(qn('a:solidFill')),a)
    if edge is None or solid_color(edge)[1]<=0:shape.line.fill.background()
    else:
        c,a=solid_color(edge);shape.line.color.rgb=c;shape.line.width=Pt(max(.25,width))
        if dashed:shape.line.dash_style=MSO_LINE_DASH_STYLE.DASH

def add_text(slide,x,y,w,h,text,size=14,bold=False,color='#202020',align='center',rotation=0,italic=False):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(max(w,.02)),Inches(max(h,.02)))
    box.name='文字｜'+text.replace('\n',' / ')[:70]
    tf=box.text_frame;tf.clear();tf.word_wrap=False;tf.auto_size=MSO_AUTO_SIZE.NONE
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
    tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    for i,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}.get(align,PP_ALIGN.CENTER)
        p.space_before=p.space_after=Pt(0);p.line_spacing=1.0
        default=p._p.get_or_add_pPr();dr=OxmlElement('a:defRPr');font_spec(dr);default.append(dr)
        math=re.fullmatch(r'\$f_\{?(\d)\}?\$',line)
        parts=[('f',False),(math.group(1),True)] if math else [(line,False)]
        for value,sub in parts:
            run=p.add_run();run.text=value
            run.font.name=LATIN;run.font.size=Pt(size*(.8 if sub else 1));run.font.bold=bold;run.font.italic=italic or bool(math)
            run.font.color.rgb=solid_color(color)[0]
            rp=run._r.get_or_add_rPr();font_spec(rp)
            if sub:rp.set('baseline','-25000')
    box.rotation=rotation%360
    return box

class Converter:
    def __init__(self,slide,fig,region=None,title=None):
        fig.canvas.draw();self.fig=fig;self.slide=slide;self.renderer=fig.canvas.get_renderer()
        width,height=fig.canvas.get_width_height()
        self.region=region or Bbox.from_bounds(0,0,width,height)
        self.cropped=region is not None
        header=.72 if title else .08
        self.scale=min((SW-.36)/self.region.width,(SH-header-.20)/self.region.height)
        self.left=(SW-self.region.width*self.scale)/2
        self.top=header+(SH-header-.20-self.region.height*self.scale)/2
        self.stats={'native_text':0,'pictures':0,'vector_shapes':0,'lines':0}
        if title:add_text(slide,.25,.08,17.5,.52,title,23,bold=True,align='left')

    def point(self,p):
        return (self.left+(p[0]-self.region.x0)*self.scale,
                self.top+(self.region.y1-p[1])*self.scale)

    def rect(self,b):
        x,y=self.point((b.x0,b.y1));return x,y,b.width*self.scale,b.height*self.scale

    def inside(self,b):
        if not self.cropped:return True
        return b.x0>=self.region.x0-2 and b.x1<=self.region.x1+2 and b.y0>=self.region.y0-2 and b.y1<=self.region.y1+2

    def rectangle(self,b,face,edge,width=.6,rounded=False,dashed=False,oval=False):
        if not self.inside(b):return
        x,y,w,h=self.rect(b)
        if w<=0 or h<=0:return
        shape=self.slide.shapes.add_shape(MSO_SHAPE.OVAL if oval else MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                         Inches(x),Inches(y),Inches(w),Inches(h))
        if rounded:shape.adjustments[0]=min(.12,4*self.scale/min(w,h))
        style_shape(shape,face,edge,width*self.fig.dpi*self.scale,dashed)
        shape.name='可编辑图形'
        self.stats['vector_shapes']+=1
        return shape

    def connector(self,points,color,width,dashed=False,arrow=False):
        for i,(p,q) in enumerate(zip(points[:-1],points[1:])):
            if not self.inside(Bbox.from_extents(min(p[0],q[0]),min(p[1],q[1]),max(p[0],q[0]),max(p[1],q[1]))):continue
            x1,y1=self.point(p);x2,y2=self.point(q)
            shape=self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
            shape.name='可编辑箭头' if arrow and i==len(points)-2 else '可编辑连线'
            shape.line.color.rgb=solid_color(color)[0];shape.line.width=Pt(max(.25,width*self.fig.dpi*self.scale))
            if dashed:shape.line.dash_style=MSO_LINE_DASH_STYLE.DASH
            if arrow and i==len(points)-2:
                end=OxmlElement('a:tailEnd');end.set('type','triangle');end.set('w','sm');end.set('len','sm')
                shape._element.spPr.get_or_add_ln().append(end)
            self.stats['lines']+=1

    def text(self,artist):
        value=artist.get_text()
        if not value.strip():return
        b=artist.get_window_extent(self.renderer)
        if not self.inside(b):return
        if artist.get_bbox_patch() is not None:self.patch(artist.get_bbox_patch())
        x,y,w,h=self.rect(b);angle=artist.get_rotation()%360
        if angle in (90,270):
            cx,cy=x+w/2,y+h/2;w,h=h,w;x,y=cx-w/2,cy-h/2
        # Font metrics differ slightly from the source preview. Keep centres,
        # disable wrapping, and allow a small native textbox safety margin.
        pad=.035
        size=artist.get_fontsize()*self.fig.dpi*self.scale
        add_text(self.slide,x-pad,y-.018,w+2*pad,h+.036,value,size,
                 bold=artist.get_fontweight() in ('bold','heavy',700,800,900),color=artist.get_color(),
                 align=artist.get_ha(),rotation=-angle,italic=artist.get_fontstyle()=='italic')
        self.stats['native_text']+=1

    def patch(self,artist):
        if isinstance(artist,patches.FancyArrowPatch):
            pos=artist._posA_posB
            if pos is not None:
                points=artist.get_transform().transform(pos)
                self.connector(points,artist.get_edgecolor(),artist.get_linewidth(),artist.get_linestyle() in ('--','dashed'),True)
            return
        b=artist.get_window_extent(self.renderer)
        dashed=artist.get_linestyle() not in ('-','solid',None)
        self.rectangle(b,artist.get_facecolor(),artist.get_edgecolor(),artist.get_linewidth(),
                       rounded=isinstance(artist,patches.FancyBboxPatch),dashed=dashed,
                       oval=isinstance(artist,(patches.Circle,patches.Ellipse)))

    def picture(self,artist):
        b=artist.get_window_extent(self.renderer)
        if not self.inside(b):return
        arr=np.asarray(artist.get_array())
        if arr.dtype!=np.uint8:arr=np.uint8(np.clip(arr,0,1)*255)
        # Preserve the source figure's nearest-neighbour display of tiny masks.
        # Integer replication changes display resolution, not the mask geometry.
        factor=max(1,int(np.ceil(640/max(arr.shape[:2]))))
        if factor>1:arr=np.repeat(np.repeat(arr,factor,axis=0),factor,axis=1)
        stream=io.BytesIO();Image.fromarray(arr).save(stream,format='PNG');stream.seek(0)
        x,y,w,h=self.rect(b)
        pic=self.slide.shapes.add_picture(stream,Inches(x),Inches(y),Inches(w),Inches(h))
        pic.name='例图｜RGB / 深度 / 实际预测（独立图片）'
        self.stats['pictures']+=1

    def axes(self,axis):
        if axis.axison:
            self.rectangle(axis.get_window_extent(self.renderer),axis.get_facecolor(),None)
        for artist in sorted(axis.get_children(),key=lambda x:x.get_zorder()):
            if not artist.get_visible() or artist is axis.patch:continue
            if isinstance(artist,AxesImage):self.picture(artist)
            elif isinstance(artist,Text):self.text(artist)
            elif isinstance(artist,Line2D):
                self.connector(artist.get_transform().transform(artist.get_xydata()),artist.get_color(),artist.get_linewidth(),artist.get_linestyle()=='--')
            elif isinstance(artist,patches.Patch) and not artist.__class__.__name__=='Spine':self.patch(artist)
        if axis.axison and any(sp.get_visible() for sp in axis.spines.values()):
            self.rectangle(axis.get_window_extent(self.renderer),None,'#C3C3C3',.6)

    def draw(self):
        for artist in sorted(self.fig.get_children(),key=lambda x:x.get_zorder()):
            if artist is self.fig.patch or not artist.get_visible():continue
            if isinstance(artist,Axes):self.axes(artist)
            elif isinstance(artist,Text):self.text(artist)
            elif isinstance(artist,patches.Patch):self.patch(artist)
        return self.stats

def set_theme_fonts(path):
    """Apply the requested fonts also to masters, layouts and future new text."""
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with ZipFile(path) as z:contents={n:z.read(n) for n in z.namelist()}
    from lxml import etree
    for name,value in contents.items():
        if not name.startswith('ppt/') or not name.endswith('.xml'):continue
        tree=etree.fromstring(value)
        # Native Office presets otherwise inherit a drop shadow. The paper's
        # thin, flat shapes and lines should keep the reference visual style.
        for effect in tree.xpath('//a:effectLst',namespaces=ns):
            for child in list(effect):effect.remove(child)
        for ref in tree.xpath('//a:effectRef',namespaces=ns):ref.set('idx','0')
        for el in tree.xpath('//a:rPr | //a:defRPr | //a:endParaRPr',namespaces=ns):font_spec(el)
        if name.startswith('ppt/theme/'):
            for el in tree.xpath('//a:majorFont | //a:minorFont',namespaces=ns):
                for tag,font in [('latin',LATIN),('ea',CJK),('cs',LATIN)]:
                    child=el.find(qn('a:'+tag))
                    if child is not None:child.set('typeface',font)
                for child in el.findall(qn('a:font')):
                    if child.get('script') in ('Hans','Hant'):child.set('typeface',CJK)
        contents[name]=etree.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        for name,value in contents.items():z.writestr(name,value)

def main():
    figures={}
    def capture(fig,stem):figures[stem]=fig
    source.save=capture
    cases=source.cached()
    for fn in (source.teaser,source.challenges,source.framework,source.ablation,source.improvement,source.failures):fn(cases)
    prs=Presentation();prs.slide_width=Inches(SW);prs.slide_height=Inches(SH)
    prs.core_properties.title='DPSeg 报告插图（可编辑）'
    prs.core_properties.subject='宋体中文；Times New Roman 英文和数字'
    prs.core_properties.author='DPSeg'
    captions=(source.OUT/'图注与使用说明.md').read_text()
    for stem,title in source.TITLES:
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        cv=Converter(slide,figures[stem],title=None if stem in ('fig00_teaser','fig02_framework') else title.replace('（RGB 基线待补）',''))
        stats=cv.draw();SLIDE_COUNTS.append(dict(slide=len(prs.slides),title=title,**stats))
        slide.notes_slide.notes_text_frame.text=(title+'\n\n中文：宋体；英文、数字：Times New Roman。文字、模块和箭头均为可编辑对象；例图为独立图片。\n'+
            '选例来自实际课程数据与冻结模型预测；未匹配表示该目标最大整图 IoU < 0.5；显示阈值 score ≥ 0.50。\n'+
            ('此页 RGB Swin-Large 最终预测待补；保留原图状态。\n' if 'ablation' in stem else '')+
            '完整来源、图注和匹配规则：outputs/report_figures_swinl_v2/图注与使用说明.md')
    # The complete figure is intentionally dense. Three additional slides make
    # each part easy to edit or show at a normal presentation viewing distance.
    framework=figures['fig02_framework'];height=framework.canvas.get_width_height()[1]
    for title,(l,t,r,b) in [
        ('总体框架：完整推理流程',(5,48,1795,462)),
        ('模块展开：多尺度 RGB-D 特征融合',(15,473,1159,992)),
        ('模块展开：深度感知掩码细化',(1170,473,1788,992))]:
        region=Bbox.from_extents(l,height-b,r,height-t)
        slide=prs.slides.add_slide(prs.slide_layouts[6]);cv=Converter(slide,framework,region=region,title=title)
        stats=cv.draw();SLIDE_COUNTS.append(dict(slide=len(prs.slides),title=title,**stats))
        slide.notes_slide.notes_text_frame.text='由第 3 页对应区域放大；保持同一真实例图、预测及网络结构。中文宋体，英文和数字 Times New Roman。'
    path=OUT/'DPSeg_报告插图_可编辑_宋体_TimesNewRoman.pptx'
    prs.save(path);set_theme_fonts(path)
    for fig in figures.values():plt.close(fig)
    (OUT/'slides_manifest.json').write_text(json.dumps(dict(slides=SLIDE_COUNTS,
        fonts={'east_asian':CJK,'latin':LATIN,'embedded':False},
        source=str(source.OUT),powerpoint=str(path),notes='Server preview uses font substitution only; PPTX specifies the requested original typefaces.'),ensure_ascii=False,indent=2)+'\n')
    (OUT/'使用说明.md').write_text('''# DPSeg 报告插图 PPT

- 共 9 页：6 张原报告插图，以及完整流程、RGB-D 融合、掩码细化的 3 页放大图。
- 中文设置为宋体，英文、数字设置为 Times New Roman；主题、母版及逐项文字均已指定。
- 所有说明文字、模块框、箭头、连线和区域框可在 PowerPoint 中直接编辑；RGB、深度及预测掩码保留为独立图片。
- 字体未嵌入。打开文件的电脑需已安装宋体和 Times New Roman。当前服务器不含这两款原字体，因此服务器预览用宋体风格和衬线字体替代；PPT 文件中仍为指定原字体。
- 保留本次原图内容和预测：第 4 页的 RGB Swin-Large 结果尚未补入；可替换该占位框，其他列不受影响。
- 图片 ID 仅保存在来源记录，不在幻灯片中显示。逐页图注放在备注区。
''')
    print(json.dumps({'path':str(path),'MiB':round(path.stat().st_size/1024**2,2),'slides':SLIDE_COUNTS},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
