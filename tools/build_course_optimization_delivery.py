#!/usr/bin/env python3
"""Build a validation comparison and offline demo from completed saved artifacts."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rgbd_seg.course_archive import CourseArchive
from tools import build_course_dafusion_comparison as shared
from tools.build_course_small_target_comparison import native_gallery_cases, add_synchronized_zoom, browser_check
from tools.select_course_resolution import read, sha, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--browser-check',action='store_true')
    args = parser.parse_args()
    d = read(args.decision)
    require(d['status'] == 'completed' and not d['test_used'],'Final selection missing')
    for key in ('protocol','run'):
        require(sha(d[key]['path']) == d[key]['sha256'],'Selection evidence changed')
    p = read(d['protocol']['path'])
    crop = read(p['decision']['path'])
    require(sha(p['decision']['path']) == p['decision']['sha256'],'Crop/Focal decision changed')
    baseline800 = ROOT/'reports/course_resolution_ddp/formal_v1/800'
    s800 = read(baseline800/'selection.json')
    v800 = Path(s800['epochs'][s800['best_epoch']-1]['output'])
    r800 = read(v800/'evaluation_run.json')
    e800 = read(v800/'evaluation.json')
    require(sha(v800/'evaluation.json') == r800['evaluation_sha256'],'800 metrics changed')
    final_ref = p['baseline_predictions'] if d['selected_method']=='native' else d['variants'][d['selected_method']]['predictions']
    names = {'native':'1024 原始预测','spatial':'1024 + 普通平滑','depth':'1024 + 深度边界细化'}
    rows = [{'name':'800 输入（配对基线）','metrics':e800['metrics']},
            {'name':'1024 输入','metrics':d['baseline']['metrics']},
            {'name':'1024 + 局部裁剪训练','metrics':crop['trials']['crop']['metrics']},
            {'name':'1024 + Focal 训练','metrics':crop['trials']['focal']['metrics']},
            {'name':names['spatial'],'metrics':d['variants']['spatial']['metrics']},
            {'name':names['depth'],'metrics':d['variants']['depth']['metrics']}]
    output = args.output.resolve()
    require(not (output/'comparison_summary.json').exists() and not (output/'gallery').exists(),'Delivery already exists')
    output.mkdir(parents=True,exist_ok=True)
    (output/'gallery').mkdir()
    with CourseArchive(p['archive'],'val') as dataset:
        saved800,binding800 = shared.selected_predictions(v800,dataset,r800['predictions_sha256'])
        saved_final,binding_final = shared.selected_predictions(Path(final_ref['path']).parent,dataset,final_ref['sha256'])
        annotations = defaultdict(list)
        for a in dataset.coco_ground_truth()['annotations']:
            annotations[a['image_id']].append(a)
        fixed = [{'key':f'case_{i:04d}','index':i,'image_id':dataset.images[i]['id'],
                  'file_name':dataset.images[i]['file_name']} for i in shared.INDICES]
        cases = native_gallery_cases(dataset,annotations,{'mask2former':saved800,'dafusion':saved_final},fixed)
    limits = ['全部指标来自完整 2,584 张验证图，范围 0–1；不是 Kaggle 分数。',
        '800/1024、裁剪、Focal 均使用七卡、全局 batch 33、相同初始化和 5 epoch；每组 2,190 次更新、72,160 次图像曝光。',
        '深度边界实验复用同一个 1024 检查点的预测；普通平滑作为对照，置信度与实例数量保持不变。',
        '只有一个训练种子和固定训练预算；结果不能证明充分收敛或统计显著。',
        '案例按验证集固定索引选择，未按效果筛选。默认显示分数≥0.5；评测使用≥0.01，显示滑杆不改变指标。',
        '当前推荐方案由验证集确定。测试集导出不使用测试标注或分数选择模型，Kaggle 公开榜约为测试集的30%。',
        '网页展示已保存预测，支持同步放大；不提供上传图片后实时推理。']
    summary = {'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'material_passport':{'origin_skill':'academic-research-suite/experiment-agent','mode':'run',
            'scope':'Completed three-stage validation comparison','verification_status':'BOUND_TO_AUDITED_DECISION'},
        'decision':{'path':str(args.decision.resolve()),'sha256':sha(args.decision)},
        'metric_rows':rows,'selected_method':d['selected_method'],'selected_metrics':d['selected_metrics'],
        'fixed_cases':fixed,'prediction_bindings':{'800':binding800,'selected':binding_final},
        'limitations':limits,'script_sha256':sha(__file__),'new_inference':False}
    (output/'comparison_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# 小目标优化：三阶段实验结果','','## Material Passport','',
        '- Origin: academic-research-suite / experiment-agent','- Mode: run',
        '- Scope: 分辨率、独立裁剪/Focal、深度边界细化；完整验证集比较。',
        '- Verification: 绑定已完成的实验选择与验证预测，不重新推理或评分。','',
        '## 结论','',f'当前推荐：**{names[d["selected_method"]]}**。',
        f'验证集 Mask APₛ = **{d["selected_metrics"]["mask_ap_small"]:.6f}**，整体 Mask AP = **{d["selected_metrics"]["mask_ap"]:.6f}**。','',
        '| 方法 | Mask APₛ | Mask AP | AP50 | AP75 | ARₛ |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['name']+' | '+' | '.join(f'{row["metrics"][k]:.6f}' for k in shared.METRICS)+' |')
    gain=d['baseline']['metrics']['mask_ap_small']-e800['metrics']['mask_ap_small']
    lines += ['',f'提高分辨率带来 APₛ 绝对增量 {gain:+.6f}；裁剪和 Focal 的独立试验均未超过 1024 基线。',
        f'深度细化相对普通平滑的 APₛ 差值：{d["depth_minus_spatial_APsmall"]:+.6f}。',
        '后处理采用门槛预先固定为：APₛ 至少增加 0.005，整体 AP 下降不超过 0.005。','',
        '## 执行与解释','',
        '使用 GPU 0/1/2/3/5/6/7；四张 80GB 卡各 batch 6，三张 40GB 卡各 batch 3，总 batch 33。GPU 4 留给已有演示。',
        '训练每轮覆盖全部 14,432 张图。七卡验证覆盖全部 2,584 张图，最终掩码始终为 1024×1024。',
        'Focal 仅替换匹配后的掩码 BCE，gamma=2、不增加 alpha 权重，保留 Dice、类别损失和原始采样。',
        '深度细化采用一次固定联合双边平滑；5×5 邻域，空间 sigma=1.5 像素，深度 sigma=1 mm。无效中心保留原值，变空实例回退。',
        '训练抽样的目标边界深度差中位数约 9.2 mm、内部约 0.1 mm，支持测试该假设；是否采用仍由完整验证成绩决定。',
        '历史单卡 DA-Fusion 验证 APₛ 为 0.392844；它的 batch、更新次数和归一化设置不同，仅作为历史参考，不能把差异全部归因于输入分辨率。','',
        '## 交付入口','',
        '- [离线对比 Demo](gallery/index.html)：双击打开，切换六个固定验证案例，调整显示阈值，2×/4×/8×同步放大。',
        '- [分辨率实验](../course_resolution_ddp/comparison_report.md)',
        '- [独立裁剪与 Focal 实验](../course_crop_focal/comparison_report.md)',
        '- [深度边界实验方案](../course_depth_boundary/experiment_plan.md)',
        '- [最终选择及完整结果](../course_depth_boundary/decision.json)',
        '- 测试 CSV 的导出状态见同目录 `test_v1/run.json`；完成并校验后才作为提交文件。','',
        '## 限制','']+['- '+x for x in limits]
    (output/'comparison_report.md').write_text('\n'.join(lines)+'\n')
    payload={'cases':cases,'metric_keys':shared.METRICS,'metric_rows':rows,'limitations':limits,
             'summary_sha256':sha(output/'comparison_summary.json')}
    html=shared.HTML
    changes={'FIXED COURSE TEST CASES':'FIXED VALIDATION CASES',
        '深度融合，实际改变了什么？':'小目标优化：从 800 到最终方案',
        'Mask2Former 与 DA-Fusion 课程适配的已保存预测':'配对 800 基线与最终推荐方案的已保存预测',
        '全量测试指标来自 2112 张图。':'全量指标来自 2,584 张验证图；不代表 Kaggle 分数。',
        'DA-Fusion 使用 RGB-D；两条 RGB 基线仅用 RGB。新模型增加骨干和参数，差异不能仅归因于深度。':'两幅预测均使用 DA-Fusion 课程适配模型，训练预算一致。当前推荐：'+names[d['selected_method']]+'。',
        'Mask2Former · RGB<span':'800 输入<span','DA-Fusion · RGB-D<span':names[d['selected_method']]+'<span',
        'aria-label="Mask2Former掩码"':'aria-label="800基线掩码"','aria-label="DA-Fusion掩码"':'aria-label="最终方案掩码"',
        '页面缩略图用于浏览，评估始终使用原始 1024×1024 掩码。':'掩码保留原始 1024×1024 像素，支持同步放大。',
        '结果审计 SHA256：${data.summary_sha256}。仅展示已保存结果；页面无需联网。':'展示已保存的验证集预测；页面无需联网。'}
    for old,new in changes.items():
        require(old in html,'Gallery template changed: '+old)
        html=html.replace(old,new)
    html=add_synchronized_zoom(html.replace('__DATA__',shared.safe_json(payload)))
    (output/'gallery/index.html').write_text(html)
    (output/'gallery/gallery.json').write_text(json.dumps({'status':'completed','fixed_cases':fixed,
        'summary_sha256':sha(output/'comparison_summary.json'),'html_sha256':sha(output/'gallery/index.html'),
        'split':'val','new_inference':False,'panel_aliases':{'mask2former':'800','dafusion':names[d['selected_method']]}},
        ensure_ascii=False,indent=2)+'\n')
    if args.browser_check:
        browser_check(output)
    print(json.dumps({'status':'completed','output':str(output),'selected_method':d['selected_method']}))


if __name__=='__main__':
    main()
