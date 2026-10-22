"""Run a YOLOv8 detection .kmodel with nncase Simulator on a local video.

This is a PC functional test, NOT a K230 FPS benchmark and NOT a tracker.
Requires Python environment with nncase, nncase-kpu, numpy, opencv-python.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import nncase


def find_video(path_or_prefix: str) -> Path:
    path = Path(path_or_prefix)
    if path.is_file():
        return path
    if path.suffix:
        raise FileNotFoundError(f"Video not found: {path}")
    matches = [p for p in path.parent.glob(path.name + '.*')
               if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one video named '{path.name}.*'; found: {matches}")
    return matches[0]


def letterbox(frame: np.ndarray, size: int, fill: int = 128):
    height, width = frame.shape[:2]
    ratio = min(size / width, size / height)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    canvas[top:top+new_height, left:left+new_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb, ratio, left, top


def decode_yolov8(outputs, conf: float, nms: float, ratio: float,
                  left: int, top: int, width: int, height: int):
    if len(outputs) != 1:
        raise RuntimeError(f"Expected 1 YOLOv8 detection output, got {len(outputs)}: "
                           f"{[x.shape for x in outputs]}")
    arr = np.asarray(outputs[0], dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected detection output shape: {arr.shape}")
    # YOLOv8 commonly exports [1, 4+classes, boxes]; convert to [boxes, 4+classes].
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] < 5:
        raise RuntimeError(f"Unexpected YOLOv8 channel count: {arr.shape}")
    scores_by_class = arr[:, 4:]
    class_ids = scores_by_class.argmax(axis=1)
    scores = scores_by_class.max(axis=1)
    inds = np.where(scores >= conf)[0]
    if not len(inds):
        return []

    coords = arr[inds, :4]
    cx = (coords[:, 0] - left) / ratio
    cy = (coords[:, 1] - top) / ratio
    bw = coords[:, 2] / ratio
    bh = coords[:, 3] / ratio
    x1 = np.clip(cx - bw / 2, 0, width - 1)
    y1 = np.clip(cy - bh / 2, 0, height - 1)
    x2 = np.clip(cx + bw / 2, 0, width - 1)
    y2 = np.clip(cy + bh / 2, 0, height - 1)
    boxes_xywh = np.stack([x1, y1, x2-x1, y2-y1], axis=1)
    valid = np.where((boxes_xywh[:, 2] > 1) & (boxes_xywh[:, 3] > 1))[0]
    boxes_xywh = boxes_xywh[valid]
    chosen_scores = scores[inds][valid]
    chosen_classes = class_ids[inds][valid]

    detections = []
    for class_id in np.unique(chosen_classes):
        class_positions = np.where(chosen_classes == class_id)[0]
        selected = cv2.dnn.NMSBoxes(
            boxes_xywh[class_positions].tolist(),
            chosen_scores[class_positions].tolist(),
            conf, nms)
        if len(selected) == 0:
            continue
        for index in np.asarray(selected).reshape(-1):
            pos = class_positions[int(index)]
            x, y, w, h = boxes_xywh[pos]
            detections.append((int(x), int(y), int(x+w), int(y+h),
                               float(chosen_scores[pos]), int(class_id)))
    return detections


def main():
    parser = argparse.ArgumentParser(description='Test K230 YOLOv8 KModel using nncase Simulator')
    parser.add_argument('--model', default='runs/detect/drone_only_v2.kmodel')
    parser.add_argument('--video', default='tests/test1_720p_30fps')
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--conf', type=float, default=0.2)
    parser.add_argument('--nms', type=float, default=0.45)
    parser.add_argument('--save', default='', help='Output video path (optional)')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f'KModel not found: {model_path}')
    video_path = find_video(args.video)

    print(f'Loading KModel: {model_path.resolve()}')
    simulator = nncase.Simulator()
    simulator.load_model(model_path.read_bytes())
    print('Inputs:', simulator.inputs_size, 'Outputs:', simulator.outputs_size)
    if simulator.inputs_size != 1:
        raise RuntimeError('This script expects a single-input YOLOv8 model.')

    # Use model input tensor to inspect actual shape/type; K230 model normally needs uint8 NCHW.
    try:
        input_template = simulator.get_input_tensor(0).to_numpy()
        expected_shape = input_template.shape
        input_dtype = input_template.dtype
    except Exception as exc:
        print('Could not inspect input tensor; using expected uint8 NCHW:', exc)
        expected_shape = (1, 3, args.size, args.size)
        input_dtype = np.dtype(np.uint8)
    print('KModel input:', expected_shape, input_dtype)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    output = None
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb, scale, left, top = letterbox(frame, args.size)
            if expected_shape == (1, 3, args.size, args.size):
                input_array = np.transpose(rgb, (2, 0, 1))[None, ...]
            elif expected_shape == (1, args.size, args.size, 3):
                input_array = rgb[None, ...]
            else:
                raise RuntimeError(f'Unexpected KModel input shape {expected_shape}. '
                                   'Check size/layout against your conversion settings.')
            # For models compiled with uint8 input_type and input_range [0,1],
            # normalization happens INSIDE the kmodel; do not divide by 255 here.
            if input_dtype == np.dtype(np.uint8):
                input_array = input_array.astype(np.uint8, copy=False)
            else:
                raise RuntimeError(f'Unsupported input dtype {input_dtype}; verify to_kmodel.py')
            input_array = np.ascontiguousarray(input_array)
            simulator.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input_array))
            start = time.perf_counter()
            simulator.run()
            infer_ms = (time.perf_counter() - start) * 1000
            outs = [simulator.get_output_tensor(i).to_numpy().copy()
                    for i in range(simulator.outputs_size)]
            if count == 0:
                print('KModel outputs:', [(x.shape, str(x.dtype)) for x in outs])
            dets = decode_yolov8(outs, args.conf, args.nms,
                                scale, left, top, w, h)
            for x1, y1, x2, y2, score, class_id in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = 'drone' if class_id == 0 else f'class_{class_id}'
                cv2.putText(frame, f'{label} {score:.2f}', (x1, max(22, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            cv2.putText(frame, f'Simulator {infer_ms:.1f} ms | detections: {len(dets)}',
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 2)
            if args.save:
                if output is None:
                    target = Path(args.save)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'),
                                             source_fps, (w, h))
                    if not output.isOpened():
                        raise RuntimeError(f'Cannot create output video: {target}')
                output.write(frame)
            if not args.no_display:
                preview = cv2.resize(frame, (int(w * min(1., 960/w)),
                                                   int(h * min(1., 960/w))))
                cv2.imshow('KModel PC Simulator (press Q to quit)', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            count += 1
            if count % 30 == 0:
                print(f'Frame {count}, simulator inference {infer_ms:.1f} ms, detections {len(dets)}')
            if args.max_frames and count >= args.max_frames:
                break
    finally:
        cap.release()
        if output is not None:
            output.release()
        cv2.destroyAllWindows()
    print(f'Completed {count} frames (simulator timings are NOT K230 FPS).')


if __name__ == '__main__':
    main()
