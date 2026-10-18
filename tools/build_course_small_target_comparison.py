#!/usr/bin/env python3
"""Audit a completed sampling experiment and compare it to original DA-Fusion.

Read existing predictions only. Never train, infer, rescore, or alter old runs.
"""
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_course_dafusion_comparison as shared
from tools.summarize_course_finetune import read, sha256, check_hash, require
from tools.run_course_dafusion import lr_factor
from rgbd_seg.course_archive import CourseArchive

NAMES = {'mask2former':'Mask2Former · RGB', 'original':'DA-Fusion 原版 · RGB-D',
         'optimized':'DA-Fusion + 小目标采样 · RGB-D'}


def add_synchronized_zoom(html):
    """Magnify the existing native canvases; saved predictions and metrics stay intact."""
    controls='''<div class="controls" aria-label="同步放大查看">
<label>放大查看 <select id="zoom" aria-label="放大倍数"><option value="1">全图</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select></label>
<label class="zoom-pan" hidden>水平位置 <input id="pan-x" type="range" min="0" max="1000" value="500" aria-label="观察区域水平位置"></label>
<label class="zoom-pan" hidden>垂直位置 <input id="pan-y" type="range" min="0" max="1000" value="500" aria-label="观察区域垂直位置"></label>
<button id="view-reset" type="button">恢复全图</button></div>
<p class="muted">放大后，点击任一图像或调整位置滑杆，四幅图会同步显示同一区域。放大保留原始掩码像素，不改变预测和指标。</p>
'''
    require('<p id="case-info"' in html and '</style>' in html and '</html>' in html,
            'Gallery template no longer supports zoom insertion')
    html=html.replace('<p id="case-info"',controls+'<p id="case-info"',1)
    html=html.replace('</style>', '.zoom-viewport{width:100%;aspect-ratio:1;overflow:hidden;position:relative}'
        '.zoom-viewport canvas{transform-origin:0 0}.zoom-pan[hidden]{display:none}'
        '.zoom-viewport[data-zoomed="true"]{cursor:crosshair}</style>',1)
    script='''<script>
(()=>{const get=id=>document.getElementById(id),names=['rgb','gt','mask2former','dafusion'];
for(const name of names){const canvas=get(name),viewport=document.createElement('div');viewport.className='zoom-viewport';canvas.before(viewport);viewport.append(canvas);viewport.addEventListener('click',event=>{const view=window.__galleryView;if(view.zoom===1)return;const box=viewport.getBoundingClientRect(),extent=1-1/view.zoom;const left=view.x+(event.clientX-box.left)/box.width/view.zoom-.5/view.zoom,top=view.y+(event.clientY-box.top)/box.height/view.zoom-.5/view.zoom;get('pan-x').value=Math.round(Math.max(0,Math.min(extent,left))/extent*1000);get('pan-y').value=Math.round(Math.max(0,Math.min(extent,top))/extent*1000);apply()})}
function apply(){const zoom=Number(get('zoom').value),extent=1-1/zoom,x=Number(get('pan-x').value)/1000*extent,y=Number(get('pan-y').value)/1000*extent;for(const name of names){const canvas=get(name);canvas.style.transform=`translate(${-x*zoom*100}%,${-y*zoom*100}%) scale(${zoom})`;canvas.style.imageRendering=zoom>1?'pixelated':'auto';canvas.parentElement.dataset.zoomed=String(zoom>1)}for(const label of document.querySelectorAll('.zoom-pan'))label.hidden=zoom===1;window.__galleryView={zoom,x,y,size:1/zoom}}
get('zoom').addEventListener('change',apply);get('pan-x').addEventListener('input',apply);get('pan-y').addEventListener('input',apply);get('view-reset').addEventListener('click',()=>{get('zoom').value='1';get('pan-x').value='500';get('pan-y').value='500';apply()});apply();
})();</script>'''
    return html.replace('</html>',script+'</html>',1)


def check_synchronized_zoom(page):
    """Exercise shared coordinates, pan controls, pixel preservation and reset."""
    page.wait_for_function('window.__galleryReady===true')
    before=page.locator('#metrics').inner_text()
    pixels=page.locator('canvas').evaluate_all(
        'els=>els.map(el=>el.toDataURL())')
    states=[]
    for zoom in ('2','4','8'):
        page.locator('#zoom').select_option(zoom)
        for x,y in ((0,0),(500,500),(1000,1000)):
            for axis,value in (('x',x),('y',y)):
                page.locator('#pan-'+axis).evaluate(
                    '(el,value)=>{el.value=value;el.dispatchEvent(new Event("input"))}',str(value))
            state=page.evaluate('window.__galleryView')
            require(state=={'zoom':int(zoom),'x':x/1000*(1-1/int(zoom)),
                    'y':y/1000*(1-1/int(zoom)),'size':1/int(zoom)},'Crop coordinates differ')
            transforms=page.locator('canvas').evaluate_all('els=>els.map(el=>el.style.transform)')
            require(len(transforms)==4 and len(set(transforms))==1,'Panels are not synchronized')
            require(page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'Zoom causes page overflow')
            states.append(state)
    page.locator('#pan-x').evaluate('el=>{el.value=500;el.dispatchEvent(new Event("input"))}')
    page.locator('#pan-y').evaluate('el=>{el.value=500;el.dispatchEvent(new Event("input"))}')
    page.locator('.zoom-viewport').first.scroll_into_view_if_needed()
    box=page.locator('.zoom-viewport').first.bounding_box()
    page.mouse.click(box['x']+.75*box['width'],box['y']+.25*box['height'])
    view=page.evaluate('window.__galleryView')
    require(view['x']>.4375 and view['y']<.4375,'Click did not move the shared crop')
    require(page.locator('canvas').evaluate_all('els=>els.map(el=>el.toDataURL())')==pixels,
            'Zoom changed native canvas pixels')
    require(page.locator('#metrics').inner_text()==before,'Zoom changed metrics')
    page.locator('#view-reset').click()
    require(page.evaluate('window.__galleryView')=={'zoom':1,'x':0,'y':0,'size':1},'View reset differs')
    require(page.locator('.zoom-pan:visible').count()==0,'Pan controls should collapse at full view')
    return states


def browser_check(output):
    shared.browser_check(output)
    from playwright.sync_api import sync_playwright
    errors=[];external=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1500,'height':1100},offline=True)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda req:external.append(req.url) if req.url.startswith(('http:','https:')) else None)
        page.goto((output/'gallery/index.html').as_uri())
        states=check_synchronized_zoom(page)
        page.locator('#zoom').select_option('8')
        page.screenshot(path=str(output/'gallery/browser/zoom-8x.png'),full_page=True)
        for option in page.locator('#case option').evaluate_all('els=>els.map(el=>el.value)'):
            page.locator('#case').select_option(option)
            page.wait_for_function('window.__galleryReady===true')
            require(page.evaluate('window.__galleryCase')==option,'Zoomed case selection differs')
            require(page.evaluate('window.__galleryView.zoom')==8,'Case selection reset the zoom')
            require(len(set(page.locator('canvas').evaluate_all('els=>els.map(el=>el.style.transform)')))==1,
                    'Case selection broke synchronized zoom')
        for width in (390,320):
            page.set_viewport_size({'width':width,'height':1100})
            check_synchronized_zoom(page)
        require(not errors and not external,'Zoom produced browser errors or network requests')
        browser.close()
    (output/'gallery/browser/zoom_test.json').write_text(json.dumps({
        'status':'passed','states':states,'viewport_widths':[1500,390,320],
        'native_canvas_pixels_unchanged':True,'metrics_unchanged':True,
        'js_errors':errors,'external_requests':external},ensure_ascii=False,indent=2)+'\n')


def native_gallery_cases(dataset, annotations, saved, fixed_cases):
    """Keep native 1024px mask sprites so tiny instances survive display preparation."""
    cases=[]
    for index,previous in zip(shared.INDICES,fixed_cases):
        record=dataset.images[index]
        require(previous['index']==index and previous['image_id']==record['id']
            and previous['file_name']==record['file_name'],'Fixed cases differ')
        image=dataset.load_rgb(index).convert('RGB')
        require(image.size==(1024,1024),'Native case image dimensions differ')
        picture=io.BytesIO();image.save(picture,format='WEBP',quality=90,method=6)
        case={'key':f'case_{index:04d}','index':index,'image_id':record['id'],
            'file_name':record['file_name'],'original_size':[1024,1024],
            'display_size':[1024,1024],'rgb':'data:image/webp;base64,'
                +base64.b64encode(picture.getvalue()).decode('ascii'),'gt':[],'predictions':{}}
        for ordinal,ann in enumerate(annotations[record['id']]):
            if ann['area']>0:
                case['gt'].append({'annotation_id':ann['id'],**shared.mask_layer(
                    shared.decode(ann['segmentation'],1024,1024),ordinal,(1024,1024))})
        for key in ('mask2former','dafusion'):
            case['predictions'][key]=[{'score':item['score'],**shared.mask_layer(
                shared.decode(item['segmentation'],1024,1024),ordinal,(1024,1024))}
                for ordinal,item in enumerate(saved[key][index]['predictions'])]
        cases.append(case)
    return cases


def training_audit(folder, baseline_folder):
    run=read(folder/'run.json'); protocol=read(Path(run['arguments']['protocol']))
    rows=[json.loads(x) for x in (folder/'training_log.jsonl').read_text().splitlines()]
    require(len(rows)==9020 and [r['global_step'] for r in rows]==list(range(1,9021)), 'Training coverage differs')
    for epoch in range(1,6):
        part=rows[(epoch-1)*1804:epoch*1804]
        require(all(r['epoch']==epoch for r in part)
            and [r['images_in_epoch'] for r in part]==list(range(8,14433,8)), 'Epoch coverage differs')
        require(read(folder/f'train_order_epoch_{epoch:02d}.json')==
                read(baseline_folder/f'train_order_epoch_{epoch:02d}.json'), 'Original DA image order differs')
        entry=read(folder/'selection.json')['epochs'][epoch-1]
        require(abs(sum(r['loss'] for r in part)/1804-entry['training_loss_mean'])<1e-9,'Training loss receipt differs')
        val=Path(entry['output']); receipt=read(val/'evaluation_run.json')
        check_hash(val/'predictions.jsonl',receipt['predictions_sha256'])
    for step,r in enumerate(rows):
        require(all(math.isfinite(r[k]) for k in ('loss','grad_norm_before_clip','head_lr','step_seconds_with_data')),
                'Nonfinite training diagnostics')
        require(r['total_steps']==9020 and r['expected_train_images']==14432
            and r['grad_norm_before_clip']>=0 and r['step_seconds_with_data']>0,
            'Training budget or numeric diagnostics differ')
        require(abs(r['head_lr']-.0002*lr_factor(step,9020))<1e-15,'Learning-rate schedule differs')
    samples=[json.loads(x) for x in (folder/'sampling_log.jsonl').read_text().splitlines()]
    require(len(samples)==18040,'Expected two microbatches per update')
    exposures=[]
    for r in samples:
        require(len(r['image_ids'])==4 and r['loss_mask_calls']==10 and r['point_budget_per_pair']==12544,
                'Sampling audit budget differs')
        pairs=r['matched_small_pairs_across_layers']
        require(r['foreground_points_forced']==r['boundary_points_forced']==128*pairs,
                'Actual quota application differs')
        require(0<=pairs<=10*(r['small_targets']-r['small_empty_after_resize']), 'Small match counts differ')
        exposures.extend(r['image_ids'])
    expected=[]
    for epoch in range(1,6):expected.extend(read(folder/f'train_order_epoch_{epoch:02d}.json')['image_ids'])
    require(exposures==expected,'Actual sampling logs do not cover frozen training order')
    check_hash(Path(protocol['data_audit']['path']),protocol['data_audit']['sha256'])
    data_audit=read(Path(protocol['data_audit']['path']))
    require(sum(r['small_targets'] for r in samples)==5*data_audit['splits']['train']['small_gt_area_le_1024'],
            'Original-area small-instance exposure count differs')
    require(sum(r['matched_small_pairs_across_layers'] for r in samples)>0,'Optimization never applied')
    require(run['model_metadata']['parameters']==87517981, 'Architecture parameter count changed')
    require(run['model_metadata']['small_target_policy']==protocol['method']['small_target_policy'], 'Policy differs')
    return {'status':'passed','updates':9020,'image_exposures':72160,'microbatches':len(samples),
        'loss_and_gradient_finite':True,'exact_original_epoch_orders':True,
        'small_targets':sum(r['small_targets'] for r in samples),
        'resize_empty_small_targets':sum(r['small_empty_after_resize'] for r in samples),
        'matched_small_pairs_across_layers':sum(r['matched_small_pairs_across_layers'] for r in samples),
        'training_log_sha256':sha256(folder/'training_log.jsonl'),
        'sampling_log_sha256':sha256(folder/'sampling_log.jsonl')}


def learning_curve(models, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for key,color,marker in [('original','#1764ab','o'),('optimized','#c65b22','s')]:
        history=models[key]['validation'];chosen=models[key]['best_epoch']
        for ax,metric,title in zip(axes,('mask_ap_small','mask_ap'),('Validation mask APsmall','Validation mask AP')):
            values=[r['metrics'][metric] for r in history]
            ax.plot(range(1,6),values,marker=marker,color=color,label=key,linewidth=2)
            ax.scatter(chosen,values[chosen-1],marker='*',s=160,color=color,edgecolor='black',zorder=5)
            ax.set(title=title,xlabel='Training epoch',ylabel='AP (0-1)',ylim=(0,1),xticks=range(1,6))
            ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,['Original DA-Fusion','DA-Fusion + small-target quota'],loc='lower center',ncol=2)
    fig.suptitle('Five epochs, full validation (2,584 images); single seed')
    fig.tight_layout(rect=(0,.1,1,.94));fig.savefig(output,dpi=200);plt.close(fig)


def build(args):
    folder=args.run.resolve(); output=args.output.resolve()
    run=read(folder/'run.json')
    require(run.get('status')=='completed' and run.get('scope')=='full_train_val_test',
            'Full formal experiment is incomplete; no comparison generated')
    for name in ('comparison_summary.json','comparison_report.md','learning_curves.png','gallery'):
        require(not (output/name).exists(), 'Refusing to overwrite final comparison')
    protocol=read(Path(run['arguments']['protocol']))
    require(protocol['schema']=='course_small_target_protocol_v1','Wrong experiment protocol')
    original_folder=Path(protocol['baseline_comparison']['baseline_run'])
    baseline_summary=Path(protocol['baseline_comparison']['baseline_summary_path'])
    check_hash(baseline_summary,protocol['baseline_comparison']['baseline_summary_sha256'])
    previous=read(baseline_summary)
    require(previous['status']=='complete_and_verified','Original result was not verified')
    rgb_path=Path(previous['source_binding']['baseline_summary_path'])
    check_hash(rgb_path,previous['source_binding']['baseline_summary_sha256'])
    rgb=read(rgb_path)
    models={}; saved={}
    with CourseArchive(protocol['archive'],'test') as dataset:
        for key,source in [('original',original_folder),('optimized',folder)]:
            model,preds=shared.audit_dafusion(source,dataset,rgb)
            model['name']=NAMES[key];models[key]=model;saved[key]=preds
        require(models['original']['gt_instances']==models['optimized']['gt_instances']
            and models['original']['gt_area_groups_including_crowd']==models['optimized']['gt_area_groups_including_crowd'],
            'Test GT coverage differs')
        models['mask2former']={'name':NAMES['mask2former'],
            'metrics':rgb['models']['mask2former']['test']['metrics']}
        annotations=defaultdict(list)
        for ann in dataset.coco_ground_truth()['annotations']:annotations[ann['image_id']].append(ann)
        fixed=previous['fixed_cases']
        cases=native_gallery_cases(dataset,annotations,
            {'mask2former':saved['original'],'dafusion':saved['optimized']},fixed)
    audit=training_audit(folder,original_folder)
    best_original=models['original']['validation'][models['original']['best_epoch']-1]['metrics']
    best_new=models['optimized']['validation'][models['optimized']['best_epoch']-1]['metrics']
    validation_delta={k:best_new[k]-best_original[k] for k in shared.METRICS}
    test_delta={k:models['optimized']['metrics'][k]-models['original']['metrics'][k] for k in shared.METRICS}
    gate=protocol['improvement_gate']
    adopted=(validation_delta['mask_ap_small']>=gate['minimum_absolute_ap_small_gain']
        and validation_delta['mask_ap']>=-gate['maximum_absolute_overall_ap_drop'])
    control=output/'launch_v1'; exit_receipt=read(control/'exit.json')
    require(exit_receipt['exit_code']==0 and exit_receipt['status']=='completed','Supervisor did not record successful exit')
    rules_path=ROOT/'reports/kaggle_score_alignment/20261015_rules_clarification.json'
    competition_context=read(rules_path)
    limitations=['单种子、五轮训练，不代表充分收敛或统计显著。',
        'GT只用于训练采样；推理不使用标注，网络参数量与原版相同。',
        '改法改变匹配后掩码损失的采样分布和后续随机流；没有新增边界损失或细化头。',
        '预先固定验证门槛；历史测试分数已知，本轮未根据新测试结果调整方案。',
        '固定案例沿用原版六个索引；展示阈值0.5，评估阈值0.01，滑杆不改变指标。',
        '按用户提供的比赛说明，Kaggle公开榜使用约30%的测试图，最终榜使用其余70%；本页为全部2112张图的本地评分。',
        '下载的课程归档实际含测试标注；这些标注与Kaggle后台标签的一致性尚未核实。']
    summary={'status':'complete_and_verified','schema':'course_small_target_comparison_v1',
        'verification_status':'ANALYZED','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'counts':{'train_per_epoch':14432,'epochs':5,'val':2584,'test':2112},'models':models,
        'training_audit':audit,'validation_delta':validation_delta,'test_delta':test_delta,
        'validation_adoption_gate_passed':adopted,'improvement_gate':gate,'limitations':limitations,
        'fixed_cases':fixed,'mask_sprite_resolution':[1024,1024],
        'competition_context':competition_context,
        'source_binding':{'generator_sha256':sha256(__file__),
            'competition_rules_path':str(rules_path),'competition_rules_sha256':sha256(rules_path),
            'original_summary_path':str(baseline_summary),'original_summary_sha256':sha256(baseline_summary)},
        'audit_scope':'Saved-artifact consistency audit; no new model execution or COCO scoring'}
    output.mkdir(parents=True,exist_ok=True);(output/'gallery').mkdir()
    (output/'comparison_summary.json').write_text(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    lines=['# 小目标采样优化：正式实验结果','','全部数值为0–1，以下表格覆盖全部2,112张测试图。',
        'Kaggle公开榜使用约30%的测试图，最终榜使用其余70%；本地全量成绩与榜单成绩应分开报告。','',
        '| 模型 | APsmall | AP | AP50 | AP75 | ARsmall |','|---|---:|---:|---:|---:|---:|']
    for key in ('mask2former','original','optimized'):
        lines.append('| '+NAMES[key]+' | '+' | '.join(f'{models[key]["metrics"][k]:.6f}' for k in shared.METRICS)+' |')
    lines += ['',f'相对原版测试 APsmall 差值：{test_delta["mask_ap_small"]:+.6f}；整体 AP 差值：{test_delta["mask_ap"]:+.6f}。',
        f'验证 APsmall 差值：{validation_delta["mask_ap_small"]:+.6f}；预设验证采用门槛：'+('通过。' if adopted else '未通过。'),
        '', '本轮只改变小实例匹配后的监督点分配，总点数12,544、网络及参数量保持相同。',
        '[完整方法与设置](../../docs/small_target_experiment.md) · [固定案例对比](gallery/index.html) · [验证曲线](learning_curves.png) · [提交文件](formal_v1/test/submission.csv)',
        '', '## 限制','']+['- '+s for s in limitations]
    (output/'comparison_report.md').write_text('\n'.join(lines)+'\n')
    payload={'cases':cases,'metric_keys':shared.METRICS,'metric_rows':[
        {'name':NAMES[k],'metrics':models[k]['metrics']} for k in ('mask2former','original','optimized')],
        'limitations':limitations,'summary_sha256':sha256(output/'comparison_summary.json')}
    html=shared.HTML.replace('深度融合，实际改变了什么？','小目标采样，实际改变了什么？')
    html=html.replace('Mask2Former 与 DA-Fusion 课程适配的已保存预测','DA-Fusion 原版与小目标采样版的已保存预测')
    html=html.replace('DA-Fusion 使用 RGB-D；两条 RGB 基线仅用 RGB。新模型增加骨干和参数，差异不能仅归因于深度。',
        '两种 DA-Fusion 使用相同 RGB-D 输入与网络参数量；本轮只改变训练中的小目标监督点分配。')
    html=html.replace('全量测试指标来自 2112 张图。',
        '本地全量测试指标来自 2112 张图。Kaggle 公开榜使用约30%，最终榜使用其余70%。')
    verdict=('达到预设验证改进门槛。' if adopted else '未达到预设验证改进门槛，保留原版作为当前推荐模型。')
    html=html.replace('<div class="table-wrap">',
        '<p class="notice">'+verdict+'</p><div class="table-wrap">',1)
    html=html.replace('Mask2Former · RGB<span','DA-Fusion 原版<span').replace('aria-label="Mask2Former掩码"','aria-label="原版DA-Fusion掩码"')
    html=html.replace('DA-Fusion · RGB-D<span','DA-Fusion + 小目标采样<span').replace('aria-label="DA-Fusion掩码"','aria-label="采样优化版DA-Fusion掩码"')
    html=html.replace('页面缩略图用于浏览，评估始终使用原始 1024×1024 掩码。',
        '掩码图层保留原始1024×1024像素，浏览时随面板缩放；评估使用原始RLE。')
    html=html.replace('__DATA__',shared.safe_json(payload))
    html=add_synchronized_zoom(html)
    (output/'gallery/index.html').write_text(html)
    receipt={'status':'completed','fixed_cases':fixed,'new_inference':False,
        'summary_sha256':sha256(output/'comparison_summary.json'),'html_sha256':sha256(output/'gallery/index.html'),
        'panel_aliases':{'mask2former':'original DA-Fusion','dafusion':'sampling optimized DA-Fusion'},
        'generator_sha256':sha256(__file__)}
    (output/'gallery/gallery.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    learning_curve(models,output/'learning_curves.png')
    (output/'comparison_artifacts.json').write_text(json.dumps({
        'status':'completed','generator_sha256':sha256(__file__),
        'files':{name:{'path':str(output/name),'sha256':sha256(output/name)} for name in
            ('comparison_summary.json','comparison_report.md','learning_curves.png','gallery/index.html')},
        'source':'Existing completed evaluations and predictions; no new inference or scoring'},
        ensure_ascii=False,indent=2)+'\n')
    if args.browser_check:browser_check(output)
    print(json.dumps({'status':'completed','validation_gate_passed':adopted,'test_ap_small_delta':test_delta['mask_ap_small']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'reports/course_small_target/formal_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/course_small_target')
    parser.add_argument('--browser-check',action='store_true')
    parser.add_argument('--check-existing',action='store_true')
    args=parser.parse_args()
    if args.check_existing:browser_check(args.output.resolve())
    else:build(args)
