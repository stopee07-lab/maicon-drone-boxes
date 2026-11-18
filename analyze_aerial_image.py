"""Inspect a local aerial image without connecting to the robot or game API."""
import argparse
import hashlib
from pathlib import Path

from capture_boxes import analyze_image, decode_image, output_report, render_overlay, render_ugv_overlay
from ugv_detector import detect_ugvs


def main():
    parser = argparse.ArgumentParser(description='Detect UGVs and boxes in a local aerial photo')
    parser.add_argument('image',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--roi',type=int,nargs=4)
    parser.add_argument('--ugv-only',action='store_true',help='Skip the separate box detector')
    args=parser.parse_args()
    raw=args.image.read_bytes()
    image=decode_image(raw)
    roi=tuple(args.roi) if args.roi else None
    if args.ugv_only:
        ugvs=detect_ugvs(image,roi=roi)
        result={'image_size_px':[image.shape[1],image.shape[0]],'ugv_count':len(ugvs),
                'ugvs':ugvs,'boxes_analyzed':False}
    else:
        result=analyze_image(image,roi=roi)
        result['boxes_analyzed']=True
    result.update(source_name=args.image.name,source_sha256=hashlib.sha256(raw).hexdigest(),
                  source_kind='local_photo',tracking=False,roi=roi)
    args.output.mkdir(parents=True,exist_ok=False)
    render_ugv_overlay(image,result['ugvs'],args.output/'ugvs.png',roi)
    if not args.ugv_only:
        render_overlay(image,result['observations'],args.output/'boxes.png',roi)
    output_report(result,args.output/'analysis.json')
    output_report({'output':str(args.output.resolve()),'ugv_count':result['ugv_count']})


if __name__=='__main__':
    main()
