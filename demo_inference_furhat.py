"""AlphaPose Furhat live camera inference.
Keeps the same AlphaPose detection -> crop transform -> pose model -> heatmap decoding
path as the demo pipeline, but saves results into the custom per-person JSON format.
Each run gets a unique timestamped prefix, and a new JSON file is created whenever a
new person is detected.
"""
import argparse
import asyncio
import base64
import json
import os
import platform
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm
import natsort  # kept to mirror demo_inference imports
import websockets

# Make AlphaPose root importable when this file lives inside /scripts
# This script lives inside AlphaPose/scripts, but a lot of the imports we need
# sit one folder higher up in the AlphaPose root. These two lines make sure
# Python can still find the main AlphaPose modules when you run:
#   python scripts/demo_inference_furhat.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import flip, flip_heatmap, get_func_heatmap_to_coord
from alphapose.utils.vis import getTime
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL


"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Furhat Demo (custom JSON output)')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default='yolo')
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default='')
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default='')
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default='')
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default='')
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default='examples/res/')
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='kept for compatibility with demo_inference', default='custom')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=1,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='kept for compatibility')
parser.add_argument('--gpus', type=str, dest='gpus', default='0',
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=32,
                    help='kept for compatibility')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default='')
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='kept for compatibility', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

"""----------------------------- Furhat options -----------------------------"""
parser.add_argument('--furhat_ip', type=str, required=True,
                    help='Furhat IP address')
parser.add_argument('--furhat_port', type=int, default=9000,
                    help='Furhat websocket port')
parser.add_argument('--furhat_key', type=str, required=True,
                    help='Furhat realtime API auth key')
parser.add_argument('--furhat_fps', type=float, default=5.0,
                    help='How often to request frames from Furhat')
parser.add_argument('--max_frames', type=int, default=-1,
                    help='Optional limit for testing; -1 means run forever')
parser.add_argument('--session_name', type=str, default='furhat_live',
                    help='base name for output json files')
parser.add_argument('--person_iou_threshold', type=float, default=0.3,
                    help='IoU threshold for matching a detection to an existing person when tracking is off')
parser.add_argument('--person_max_missing', type=int, default=15,
                    help='how many frames a person can be missing before a later detection becomes a new JSON')
parser.add_argument('--person_min_frames', type=int, default=10,
                    help='minimum number of frames before a person gets saved to its own JSON; lower this to keep shorter tracks')

# Read all command-line settings and then load the AlphaPose config file.
# The config file tells AlphaPose which model layout, image size and heatmap
# size to use.
args = parser.parse_args()
cfg = update_config(args.cfg)

# AlphaPose's normal demo script can use multiple worker processes.
# For this Furhat version we force single-process mode because we are already
# handling live frames ourselves and this avoids a lot of multiprocessing issues.
if platform.system() == 'Windows':
    args.sp = True

# Furhat mode is more stable in single-process mode.
args.sp = True

if args.gpus.strip() == '-1' or torch.cuda.device_count() == 0:
    args.gpus = [-1]
else:
    args.gpus = [int(i) for i in args.gpus.split(',')]

args.device = torch.device('cuda:' + str(args.gpus[0]) if args.gpus[0] >= 0 else 'cpu')
args.detbatch = max(1, args.detbatch * max(1, len(args.gpus)))
args.posebatch = max(1, args.posebatch * max(1, len(args.gpus)))
args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'


def loop():
    n = 0
    while True:
        yield n
        n += 1


class FurhatCameraClient:
    # Small helper around the Furhat websocket API.
    # Its only job is:
    # 1) connect and authenticate
    # 2) ask Furhat for a camera frame
    # 3) decode the returned JPEG into an OpenCV image
    def __init__(self, ip: str, port: int, auth_key: str):
        self.url = f'ws://{ip}:{port}/v1/events'
        self.auth_key = auth_key
        self.ws = None

    async def connect(self):
        # Open the websocket and send the auth request once at startup.
        self.ws = await websockets.connect(self.url, max_size=None)
        await self.ws.send(json.dumps({'type': 'request.auth', 'key': self.auth_key}))
        auth_resp = json.loads(await self.ws.recv())
        if not auth_resp.get('access'):
            raise RuntimeError(f'Furhat auth failed: {auth_resp}')
        print('Authenticated with Furhat.')
        sys.stdout.flush()

    async def get_frame(self):
        # request.camera.once asks Furhat for one still image.
        # Furhat sends it back as a base64 JPEG, so we decode that back into
        # a normal image array that OpenCV / AlphaPose can use.
        await self.ws.send(json.dumps({'type': 'request.camera.once'}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get('type') == 'response.camera.data' and 'image' in msg:
                img_bytes = base64.b64decode(msg['image'])
                frame = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError('Failed to decode Furhat frame')
                return frame

    async def close(self):
        if self.ws is not None:
            await self.ws.close()


# AlphaPose boxes are sometimes tensors, sometimes lists.
# This helper just normalises them into plain Python floats in the form:
# [x, y, width, height]
def _to_float_box_xywh(box):
    if isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy().tolist()
    return [float(v) for v in box]


# Intersection-over-Union is a common way to measure how much two boxes overlap.
# We use it here to decide whether a person in the current frame is probably
# the same person we already saw in earlier frames.
def _iou_xywh(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class MultiPersonJSONWriter:
    # This class is only about organising output files.
    # It keeps track of who is who across frames and writes one JSON per person.
    #
    # Important points:
    # - each run gets a timestamped prefix, so runs do not overwrite each other
    # - each detected person gets their own p1 / p2 / p3 JSON file
    # - very short false detections can be ignored using min_frames_to_save
    def __init__(self, output_dir, session_name, iou_threshold=0.3, max_missing=15, min_frames_to_save=10):
        self.output_dir = output_dir
        self.session_name = session_name
        self.run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f'{self.session_name}_{self.run_stamp}'
        self.iou_threshold = float(iou_threshold)
        self.max_missing = int(max_missing)
        self.min_frames_to_save = max(1, int(min_frames_to_save))
        os.makedirs(self.output_dir, exist_ok=True)

        self.frame_size = [0, 0]
        self.total_frames = 0
        self.people = {}  # person_id -> dict(data/json_path/last_box/last_frame/track_key/is_confirmed)
        self.track_key_to_person_id = {}
        self.next_person_id = 1

    def set_frame_size_if_needed(self, width, height):
        # The JSON stores frame size once. We set it when the first frame arrives.
        if self.frame_size == [0, 0]:
            self.frame_size = [int(width), int(height)]
            for info in self.people.values():
                info['data']['frame_size'] = self.frame_size.copy()

    def increment_total_frames(self):
        # video_frames_number in the JSON means total frames seen in this run,
        # not just frames where a person was detected.
        self.total_frames += 1
        for info in self.people.values():
            info['data']['video_frames_number'] = self.total_frames

    def _new_person_record(self):
        # Create the bookkeeping for a brand-new person.
        # Nothing is written yet unless the person lasts long enough.
        person_id = self.next_person_id
        self.next_person_id += 1
        json_path = os.path.join(self.output_dir, f'{self.run_name}_p{person_id}.json')
        data = {
            'video_name': f'{self.run_name}.jpg',
            'frame_size': self.frame_size.copy(),
            'video_frames_number': self.total_frames,
            'detected_frames_number': 0,
            'person_id': person_id,
            'intention_class': 0,
            'attitude_class': 0,
            'action_class': 0,
            'frames': []
        }
        self.people[person_id] = {
            'json_path': json_path,
            'data': data,
            'last_box': None,
            'last_frame': -10**9,
            'track_key': None,
            'is_confirmed': self.min_frames_to_save <= 1,
        }
        if self.min_frames_to_save <= 1:
            print(f'Created new person JSON: {json_path}')
        else:
            print(f'Started person candidate p{person_id}: {json_path}')
        return person_id

    def _extract_track_key(self, person):
        # If pose tracking is enabled, AlphaPose can give us a track id.
        # That is the cleanest way to keep the same person matched over time.
        # If tracking is off, this usually returns None and we fall back to IoU.
        idx = person.get('idx', None)
        if idx is None:
            return None
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
        if isinstance(idx, np.ndarray):
            if idx.size == 0:
                return None
            idx = idx.reshape(-1)[0]
        try:
            idx = int(idx)
        except Exception:
            return None
        if not self.track_key_to_person_id and idx == 0:
            # Ambiguous when tracking is off; let IoU decide until explicit tracking ids exist.
            return None
        return idx

    def _match_existing_person(self, frame_idx, box_xywh, track_key=None, reserved_person_ids=None):
        # Decide whether this detection belongs to:
        # - an existing person JSON, or
        # - a brand-new person JSON
        #
        # Matching order:
        # 1) use tracker id if one exists
        # 2) otherwise compare box overlap (IoU) with recent people
        # 3) if no good match is found, start a new person
        if reserved_person_ids is None:
            reserved_person_ids = set()

        if track_key is not None and track_key in self.track_key_to_person_id:
            person_id = self.track_key_to_person_id[track_key]
            if person_id not in reserved_person_ids:
                return person_id

        best_person_id = None
        best_iou = -1.0
        for person_id, info in self.people.items():
            if person_id in reserved_person_ids:
                continue
            if info['last_box'] is None:
                continue
            if frame_idx - info['last_frame'] > self.max_missing:
                continue
            overlap = _iou_xywh(box_xywh, info['last_box'])
            if overlap > best_iou:
                best_iou = overlap
                best_person_id = person_id

        if best_person_id is not None and best_iou >= self.iou_threshold:
            return best_person_id

        return self._new_person_record()

    def add_people(self, frame_idx, people):
        # Save every person detected in the current frame into the right JSON.
        # This does not change the keypoints at all — it only repackages them
        # into the custom structure you wanted.
        used_person_ids = set()
        for person in people:
            box_xywh = _to_float_box_xywh(person['box'])
            track_key = self._extract_track_key(person)
            person_id = self._match_existing_person(frame_idx, box_xywh, track_key, used_person_ids)
            used_person_ids.add(person_id)

            info = self.people[person_id]
            if track_key is not None:
                info['track_key'] = track_key
                self.track_key_to_person_id[track_key] = person_id

            # AlphaPose already gave us the final keypoint coordinates and
            # confidence values. We keep them the same and only convert them
            # into a JSON-friendly list of [x, y, score].
            kpts_xy = person['keypoints']
            kpts_score = person['kp_score']
            if isinstance(kpts_xy, torch.Tensor):
                kpts_xy = kpts_xy.detach().cpu().numpy()
            if isinstance(kpts_score, torch.Tensor):
                kpts_score = kpts_score.detach().cpu().numpy()

            keypoints = []
            for j in range(kpts_xy.shape[0]):
                x = float(kpts_xy[j][0])
                y = float(kpts_xy[j][1])
                s = float(kpts_score[j][0] if np.ndim(kpts_score[j]) > 0 else kpts_score[j])
                keypoints.append([x, y, s])

            proposal_score = person['proposal_score']
            if isinstance(proposal_score, torch.Tensor):
                proposal_score = float(proposal_score.detach().cpu().item())
            else:
                proposal_score = float(proposal_score)

            frame_entry = {
                'frame_id': int(frame_idx),
                'keypoints': keypoints,
                'score': proposal_score,
                'box': box_xywh,
                'gaze': [0, 0, 0, 0]
            }
            info['data']['frames'].append(frame_entry)
            info['data']['detected_frames_number'] = len(info['data']['frames'])
            info['data']['video_frames_number'] = self.total_frames
            info['last_box'] = box_xywh
            info['last_frame'] = frame_idx

            if (not info['is_confirmed']) and len(info['data']['frames']) >= self.min_frames_to_save:
                info['is_confirmed'] = True
                print(f"Created new person JSON: {info['json_path']} (reached {len(info['data']['frames'])} frames)")

    def save(self):
        # Write confirmed people to disk.
        # People below the minimum frame threshold are skipped on purpose
        # so very brief false detections do not clutter the folder.
        for info in self.people.values():
            if not info['is_confirmed']:
                continue
            info['data']['video_frames_number'] = self.total_frames
            with open(info['json_path'], 'w') as f:
                json.dump(info['data'], f, indent=4)

    def list_paths(self):
        return [info['json_path'] for _, info in sorted(self.people.items()) if info['is_confirmed']]


class FurhatProcessor:
    # This is the main processing block.
    # It owns the detector, the pose model, the crop transform and the JSON writer.
    # One frame comes in, and this class turns it into:
    # detection boxes -> person crops -> keypoints -> custom JSON rows
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        # Load the object detector first.
        # This finds where people are in the image before AlphaPose estimates joints.
        self.detector = get_detector(args)

        # Build the pose model using the same config/checkpoint style as the
        # normal AlphaPose demo script.
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        print(f'Loading pose model from {args.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        # Tracking is optional. If enabled, AlphaPose will try to keep a
        # consistent id for each person over time.
        if args.pose_track:
            self.tracker = Tracker(tcfg, args)
        else:
            self.tracker = None

        if len(args.gpus) > 1 and args.gpus[0] >= 0:
            self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=args.gpus).to(args.device)
        else:
            self.pose_model.to(args.device)
        self.pose_model.eval()

        # These values define how person crops are resized before they are fed
        # into the pose model, and how big the output heatmaps are.
        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = cfg.DATA_PRESET.SIGMA
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.eval_joints = [0, 1, 2, 3, 4, 5, 6,
                            7, 8, 9, 10, 11, 12, 13,
                            14, 15, 16]
        self.use_heatmap_loss = (self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')

        # The transform handles the crop from a full image to a single-person
        # pose input. This is the same idea used in the regular AlphaPose demo.
        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                self.pose_dataset,
                scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0,
                sigma=self._sigma,
                train=False,
                add_dpg=False,
                gpu_device=args.device,
            )
        elif cfg.DATA_PRESET.TYPE == 'simple_smpl':
            from easydict import EasyDict as edict
            dummy_set = edict({
                'joint_pairs_17': None,
                'joint_pairs_24': None,
                'joint_pairs_29': None,
                'bbox_3d_shape': (2.2, 2.2, 2.2),
            })
            self.transformation = SimpleTransform3DSMPL(
                dummy_set,
                scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2.2, 2.2),
                rot=cfg.DATASET.ROT_FACTOR,
                sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False,
                add_dpg=False,
                loss_type=cfg.LOSS['TYPE'],
            )
        else:
            raise NotImplementedError(f'Unsupported DATA_PRESET.TYPE: {cfg.DATA_PRESET.TYPE}')

        if not os.path.exists(args.outputpath):
            os.makedirs(args.outputpath)

        # This writer is only responsible for the custom output format.
        # It does not affect the actual pose inference.
        self.json_writer = MultiPersonJSONWriter(
            args.outputpath,
            args.session_name,
            iou_threshold=args.person_iou_threshold,
            max_missing=args.person_max_missing,
            min_frames_to_save=args.person_min_frames,
        )
        self.runtime_profile = {'dt': [], 'pt': [], 'pn': []}

        loss_type = self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss')
        num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
        if loss_type == 'MSELoss':
            self.vis_thres = [0.4] * num_joints
        elif 'JointRegression' in loss_type:
            self.vis_thres = [0.05] * num_joints
        elif loss_type == 'Combined':
            hand_face_num = 42 if num_joints == 68 else 110
            self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num
        else:
            self.vis_thres = [0.4] * num_joints

    def _decode_pose(self, boxes, scores, ids, hm_data, cropped_boxes):
        # The pose model does not directly output final joint coordinates.
        # It outputs heatmaps. This function converts those heatmaps into
        # real image-space keypoints and then runs pose NMS to clean duplicates.
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        assert hm_data.dim() == 4
        face_hand_num = 110
        if hm_data.size()[1] == 136:
            self.eval_joints = [*range(0, 136)]
        elif hm_data.size()[1] == 26:
            self.eval_joints = [*range(0, 26)]
        elif hm_data.size()[1] == 133:
            self.eval_joints = [*range(0, 133)]
        elif hm_data.size()[1] == 68:
            face_hand_num = 42
            self.eval_joints = [*range(0, 68)]
        elif hm_data.size()[1] == 21:
            self.eval_joints = [*range(0, 21)]

        # Convert each person's heatmaps into x/y coordinates and confidence scores.
        pose_coords = []
        pose_scores = []
        for i in range(hm_data.shape[0]):
            bbox = cropped_boxes[i].tolist()
            if isinstance(self.heatmap_to_coord, list):
                pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                    hm_data[i][self.eval_joints[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                    hm_data[i][self.eval_joints[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                pose_coord, pose_score = self.heatmap_to_coord(
                    hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)

            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)

        # pose_nms removes duplicate overlapping pose predictions when tracking
        # is not being used. This is part of AlphaPose's normal post-processing.
        if not self.args.pose_track:
            boxes, scores, ids, preds_img, preds_scores, pick_ids = pose_nms(
                boxes, scores, ids, preds_img, preds_scores,
                self.args.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

        return boxes, scores, ids, preds_img, preds_scores

    def _result_for_visualization(self, boxes, scores, ids, preds_img, preds_scores, im_name):
        # Build a result dictionary in the same kind of structure AlphaPose's
        # visualisation code expects. We also reuse it for the custom JSON step.
        result_list = []
        for k in range(len(scores)):
            result_list.append({
                'keypoints': preds_img[k],
                'kp_score': preds_scores[k],
                'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                'idx': ids[k],
                'box': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
            })
        return {
            'imgname': im_name,
            'result': result_list
        }

    def _save_all_people_to_custom_json(self, frame_idx, result):
        # Push the current frame's decoded people into the JSON writer.
        if len(result['result']) == 0:
            return
        self.json_writer.add_people(frame_idx, result['result'])

    def _visualize_or_save(self, orig_img_rgb, result, im_name):
        # Optional visual output. This is separate from the JSON files.
        # Turn on --vis to show a live window or --save_img to save drawn frames.
        if not (self.args.save_img or self.args.vis):
            return

        orig_img_bgr = np.array(orig_img_rgb, dtype=np.uint8)[:, :, ::-1]

        if len(result['result']) == 0:
            img = orig_img_bgr
        else:
            if result['result'][0]['keypoints'].shape[0] == 49:
                from alphapose.utils.vis import vis_frame_dense as vis_frame
            elif self.args.vis_fast:
                from alphapose.utils.vis import vis_frame_fast as vis_frame
            else:
                from alphapose.utils.vis import vis_frame
            img = vis_frame(orig_img_bgr, result, self.args, self.vis_thres)

        if self.args.vis:
            cv2.imshow('AlphaPose Demo', img)
            cv2.waitKey(30)

        if self.args.save_img:
            vis_dir = os.path.join(self.args.outputpath, 'vis')
            os.makedirs(vis_dir, exist_ok=True)
            cv2.imwrite(os.path.join(vis_dir, im_name), img)

    def process_frame(self, frame_bgr, frame_idx: int):
        # This is the main frame pipeline:
        # 1) preprocess image for detector
        # 2) detect person boxes
        # 3) crop each person
        # 4) run pose model on each crop
        # 5) decode heatmaps into joints
        # 6) save results into the custom JSON format
        # Actual processing starts below.
        start_time = getTime()
        im_name = f'furhat_{frame_idx:08d}.jpg'

        self.json_writer.increment_total_frames()
        self.json_writer.set_frame_size_if_needed(frame_bgr.shape[1], frame_bgr.shape[0])

        # Keep this close to AlphaPose's own loader logic:
        # - detector sees the original BGR frame
        # - orig_img is stored as RGB for the crop transform / visualisation path
        # Matching the normal pipeline here helps keep the output consistent.
        # Match DetectionLoader.frame_preprocess exactly: detector sees BGR frame,
        # orig_img stored as RGB.
        img = self.detector.image_preprocess(frame_bgr)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        orig_img = frame_bgr[:, :, ::-1]
        im_dim_list = torch.FloatTensor([(frame_bgr.shape[1], frame_bgr.shape[0])]).repeat(1, 2)

        with torch.no_grad():
            # Run person detection first. If no person is found, we still keep
            # the total frame count moving, but there is nothing to add to a person JSON.
            dets = self.detector.images_detection(img, im_dim_list)

            if isinstance(dets, int) or getattr(dets, 'shape', [0])[0] == 0:
                self.json_writer.save()
                return

            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()

            # AlphaPose detector output contains class id, box coordinates,
            # score, and sometimes a tracking id depending on the mode.
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            if self.args.tracking:
                ids = dets[:, 6:7]
            else:
                ids = torch.zeros(scores.shape)

            boxes = boxes[dets[:, 0] == 0]
            scores = scores[dets[:, 0] == 0]
            ids = ids[dets[:, 0] == 0]

            if boxes is None or boxes.nelement() == 0:
                self.json_writer.save()
                return

            if self.args.profile:
                ckpt_time, det_time = getTime(start_time)
                self.runtime_profile['dt'].append(det_time)

            # Build the pose-model input for each detected person.
            # Each full-frame detection box is turned into a fixed-size crop.
            inps = torch.zeros(boxes.size(0), 3, *self._input_size)
            cropped_boxes = torch.zeros(boxes.size(0), 4)
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            inps = inps.to(self.args.device)
            batchSize = self.args.posebatch
            if self.args.flip:
                batchSize = int(batchSize / 2)
            batchSize = max(1, batchSize)

            datalen = inps.size(0)
            leftover = 1 if (datalen % batchSize) else 0
            num_batches = datalen // batchSize + leftover

            # Run the pose model in batches. The output here is still heatmaps,
            # not final x/y keypoints yet.
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                if self.args.flip:
                    inps_j = torch.cat((inps_j, flip(inps_j)))
                hm_j = self.pose_model(inps_j)
                if self.args.flip:
                    hm_j_flip = flip_heatmap(
                        hm_j[int(len(hm_j) / 2):],
                        self.pose_dataset.joint_pairs,
                        shift=True,
                    )
                    hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                hm.append(hm_j)
            hm = torch.cat(hm)

            if self.args.profile:
                ckpt_time, pose_time = getTime(ckpt_time)
                self.runtime_profile['pt'].append(pose_time)

            if self.args.pose_track:
                boxes, scores, ids, hm, cropped_boxes = track(
                    self.tracker,
                    self.args,
                    orig_img,
                    inps,
                    boxes,
                    hm,
                    cropped_boxes,
                    im_name,
                    scores,
                )

            hm = hm.cpu()

            # Turn raw model heatmaps into final keypoint coordinates.
            boxes, scores, ids, preds_img, preds_scores = self._decode_pose(
                boxes, scores, ids, hm, cropped_boxes)

            result = self._result_for_visualization(
                boxes, scores, ids, preds_img, preds_scores, im_name)

            self._save_all_people_to_custom_json(frame_idx, result)
            self.json_writer.save()
            self._visualize_or_save(orig_img, result, im_name)

            if self.args.profile:
                ckpt_time, post_time = getTime(ckpt_time)
                self.runtime_profile['pn'].append(post_time)


async def main():
    # Set everything up once, then keep pulling frames from Furhat in a loop.
    processor = FurhatProcessor(cfg, args)
    camera = FurhatCameraClient(args.furhat_ip, args.furhat_port, args.furhat_key)
    await camera.connect()

    print('Starting Furhat demo, press Ctrl + C to terminate...')
    print(f'Run name: {processor.json_writer.run_name}')
    sys.stdout.flush()

    interval = 1.0 / max(args.furhat_fps, 0.001)
    frame_counter = 0
    im_names_desc = tqdm(loop())

    try:
        for _ in im_names_desc:
            # Pull one frame from Furhat, process it, then wait just enough
            # to roughly hit the target FPS.
            frame_start = time.time()
            frame = await camera.get_frame()
            processor.process_frame(frame, frame_counter)
            frame_counter += 1

            if args.profile and processor.runtime_profile['dt']:
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(processor.runtime_profile['dt']),
                        pt=np.mean(processor.runtime_profile['pt']) if processor.runtime_profile['pt'] else 0.0,
                        pn=np.mean(processor.runtime_profile['pn']) if processor.runtime_profile['pn'] else 0.0,
                    )
                )

            if args.max_frames > 0 and frame_counter >= args.max_frames:
                break

            # This sleep keeps the loop from running faster than the requested FPS.
            # If processing already took longer than the interval, sleep is skipped.
            elapsed = time.time() - frame_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    except KeyboardInterrupt:
        print('\nStopping Furhat demo...')
    finally:
        await camera.close()
        processor.json_writer.save()
        if args.vis:
            cv2.destroyAllWindows()
        paths = processor.json_writer.list_paths()
        if paths:
            print('Saved JSON files:')
            for p in paths:
                print(p)
        else:
            print('No person JSON files were created.')


if __name__ == '__main__':
    asyncio.run(main())
