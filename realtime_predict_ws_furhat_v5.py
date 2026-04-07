"""Live SocialEgoNet prediction sender.

This file watches the per-person JSON files produced by the AlphaPose/Furhat capture
script, turns the newest pose window into a SocialEgoNet prediction, and forwards the
result to the robot-side receiver.

High-level flow:
1. Look through the folder where AlphaPose is writing person JSON files.
2. For each person, take the newest sequence_length frames.
3. Convert those frames into the graph structure expected by SocialEgoNet.
4. Predict intention, attitude, and action.
5. Decide on a simpler Yes/No "interacting" label for robot control.
6. Estimate a sensible pixel target for Furhat to look at.
7. Send one compact message to the robot receiver over WebSocket.

This script does not control Furhat directly. Its job is only to interpret the pose data
and publish live predictions.
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch, Data

from Models import SocialEgoNet
from DataLoader import WholebodyPoseData, body_edge_index, face_edge_index, hands_edge_index
from constants import (
    action_classes,
    attitude_classes,
    coco_body_point_num,
    device,
    face_point_num,
    hands_point_num,
    intention_classes,
)

try:
    import websockets
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'websockets'. Install it with: pip install websockets"
    ) from exc


# Keypoint indexes used for picking a sensible head target.
NOSE_IDX = 0
LEFT_EYE_IDX = 1
RIGHT_EYE_IDX = 2
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6


# Small helper for sending prediction messages to the robot-side receiver.
# This keeps the networking code in one place so the prediction loop stays readable.
class PredictionSender:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.websocket = None

    async def connect(self):
        if self.websocket is None or self.websocket.closed:
            self.websocket = await websockets.connect(self.ws_url)

    async def send(self, payload: dict):
        try:
            await self.connect()
            await self.websocket.send(json.dumps(payload))
        except Exception:
            # Reconnect once if the receiver restarted or the socket dropped.
            self.websocket = None
            await self.connect()
            await self.websocket.send(json.dumps(payload))

    async def close(self):
        if self.websocket is not None and not self.websocket.closed:
            await self.websocket.close()


# Read the YAML config file used to build the SocialEgoNet model and
# to recover settings such as sequence_length.
def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# Read one AlphaPose person file. Returning None instead of crashing is useful
# because these files can briefly be half-written while another process updates them.
def safe_load_json(file_path: Path) -> Optional[dict]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Ignore files that are mid-write or temporarily unreadable.
        return None


# Convert raw pixel coordinates into the normalised coordinate system the model
# was trained with. This step needs to match training-time preprocessing closely.
def normalize_frame(frame: dict, frame_width: float, frame_height: float) -> torch.Tensor:
    # Match the same coordinate scaling expected by the model.
    frame_feature = np.array(frame["keypoints"], dtype=np.float32)
    frame_feature[:, :2] = 2 * (frame_feature[:, :2] / [frame_width, frame_height] - 0.5)
    return torch.tensor(frame_feature, dtype=torch.float32)


# Pull out the newest chunk of frames for one person. The model expects a fixed
# number of frames, so until we have enough frames we simply wait.
def extract_live_window(feature_json: dict, sequence_length: int) -> Tuple[Optional[torch.Tensor], Optional[int], Optional[dict]]:
    frames: List[dict] = feature_json.get("frames", [])
    if not frames:
        return None, None, None

    frame_width, frame_height = feature_json["frame_size"][0], feature_json["frame_size"][1]
    frames = sorted(frames, key=lambda x: x["frame_id"])

    if len(frames) < sequence_length:
        return None, frames[-1]["frame_id"], frames[-1]

    # Only use the newest sequence_length frames for live prediction.
    selected_frames = frames[-sequence_length:]
    x_tensor = torch.empty(
        (sequence_length, coco_body_point_num + face_point_num + hands_point_num, 3),
        dtype=torch.float32,
    )

    for i, frame in enumerate(selected_frames):
        x_tensor[i] = normalize_frame(frame, frame_width, frame_height)

    return x_tensor, selected_frames[-1]["frame_id"], selected_frames[-1]


# Split the full whole-body tensor into body, face, and hands graphs because
# that is how SocialEgoNet expects its input to be packaged.
def build_graph_input(x_tensor: torch.Tensor, sequence_length: int) -> WholebodyPoseData:
    body_pose_graph_data = []
    face_pose_graph_data = []
    hand_pose_graph_data = []

    for i in range(sequence_length):
        frame = x_tensor[i]
        body_pose_graph_data.append(Data(x=frame[:coco_body_point_num], edge_index=body_edge_index))
        face_pose_graph_data.append(
            Data(
                x=frame[coco_body_point_num:coco_body_point_num + face_point_num],
                edge_index=face_edge_index,
            )
        )
        hand_pose_graph_data.append(
            Data(
                x=frame[coco_body_point_num + face_point_num:],
                edge_index=hands_edge_index,
            )
        )

    # SocialEgoNet expects separate graph batches for body, face and hands.
    whole_body_pose_data = WholebodyPoseData()
    whole_body_pose_data.body = Batch.from_data_list(body_pose_graph_data)
    whole_body_pose_data.face = Batch.from_data_list(face_pose_graph_data)
    whole_body_pose_data.hands = Batch.from_data_list(hand_pose_graph_data)
    return whole_body_pose_data


# Run one live prediction for one person window and package the result in a
# simple dictionary that the rest of the script can use.
def predict_single(
    model: SocialEgoNet,
    x_tensor: torch.Tensor,
    sequence_length: int,
    no_intention_prob_threshold: float,
    no_attitude_prob_threshold: float,
) -> dict:
    inputs = build_graph_input(x_tensor, sequence_length)

    with torch.no_grad():
        int_outputs, att_outputs, act_outputs = model(inputs)

        int_outputs = torch.softmax(int_outputs, dim=1)
        att_outputs = torch.softmax(att_outputs, dim=1)
        act_outputs = torch.softmax(act_outputs, dim=1)

        _, int_preds = torch.max(int_outputs, dim=1)
        _, att_preds = torch.max(att_outputs, dim=1)
        _, act_preds = torch.max(act_outputs, dim=1)

        int_idx = int_preds[0].item()
        att_idx = att_preds[0].item()
        act_idx = act_preds[0].item()

        intention_label = intention_classes[int_idx]
        attitude_label = attitude_classes[att_idx]
        action_label = action_classes[act_idx]

        # Keep full class probabilities in case you want to inspect behaviour later.
        intention_probabilities = {
            intention_classes[i]: float(int_outputs[0, i].item())
            for i in range(int_outputs.shape[1])
        }
        attitude_probabilities = {
            attitude_classes[i]: float(att_outputs[0, i].item())
            for i in range(att_outputs.shape[1])
        }

        p_not_interested = intention_probabilities.get("Not_Interested", 0.0)
        p_not_interacting = attitude_probabilities.get("Not_Interacting", 0.0)

        # Only call it a clear "No" when both non-interaction signals are strong enough.
        overall_interacting = (
            "No"
            if (p_not_interested >= no_intention_prob_threshold and p_not_interacting >= no_attitude_prob_threshold)
            else "Yes"
        )

        return {
            "intention": intention_label,
            "attitude": attitude_label,
            "action": action_label,
            "interacting": overall_interacting,
            "intention_confidence": float(int_outputs[0, int_idx].item()),
            "attitude_confidence": float(att_outputs[0, att_idx].item()),
            "action_confidence": float(act_outputs[0, act_idx].item()),
            "p_not_interested": p_not_interested,
            "p_not_interacting": p_not_interacting,
            "no_intention_prob_threshold": float(no_intention_prob_threshold),
            "no_attitude_prob_threshold": float(no_attitude_prob_threshold),
            "intention_probabilities": intention_probabilities,
            "attitude_probabilities": attitude_probabilities,
        }


# Convenience check: only treat a keypoint as usable if it has x, y, confidence
# and that confidence is above the threshold we chose.
def _valid_xyc(point: np.ndarray, min_conf: float) -> bool:
    if point.shape[0] < 3:
        return False
    x, y, c = float(point[0]), float(point[1]), float(point[2])
    return c >= min_conf and x > 0 and y > 0


# Average two keypoints. Useful when the best target is somewhere between
# left and right body landmarks.
def _midpoint(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    return (float(a[0] + b[0]) / 2.0, float(a[1] + b[1]) / 2.0)


# Choose a point Furhat should look at for this person.
# Preference order is: shoulder midpoint first, then nose, then eye midpoint.
# That usually gives a steadier target than jumping straight to a single face point.
def extract_attention_target(
    latest_frame: dict,
    frame_width: float,
    frame_height: float,
    min_body_conf: float,
    min_face_conf: float,
) -> Optional[dict]:
    keypoints = np.array(latest_frame["keypoints"], dtype=np.float32)

    nose = keypoints[NOSE_IDX]
    left_eye = keypoints[LEFT_EYE_IDX]
    right_eye = keypoints[RIGHT_EYE_IDX]
    left_shoulder = keypoints[LEFT_SHOULDER_IDX]
    right_shoulder = keypoints[RIGHT_SHOULDER_IDX]

    source = None
    target_x = target_y = None
    shoulder_width_px = None

    # Best case: estimate a point around head level from the shoulders.
    if _valid_xyc(left_shoulder, min_body_conf) and _valid_xyc(right_shoulder, min_body_conf):
        shoulder_x, shoulder_y = _midpoint(left_shoulder, right_shoulder)
        shoulder_width_px = abs(float(right_shoulder[0]) - float(left_shoulder[0]))
        upward_offset_px = float(np.clip(0.9 * shoulder_width_px, 20.0, 90.0))
        target_x = shoulder_x
        target_y = shoulder_y - upward_offset_px
        source = "shoulder_midpoint"
    elif _valid_xyc(nose, min_face_conf):
        target_x, target_y = float(nose[0]), float(nose[1])
        source = "nose"
    elif _valid_xyc(left_eye, min_face_conf) and _valid_xyc(right_eye, min_face_conf):
        target_x, target_y = _midpoint(left_eye, right_eye)
        source = "eye_midpoint"
    else:
        return None

    # Rough distance estimate from shoulder width. Falls back to a default range.
    z_m = 1.2
    if shoulder_width_px is not None and shoulder_width_px > 1.0:
        approx = 0.42 * 900.0 / shoulder_width_px
        z_m = float(np.clip(approx, 0.8, 2.5))

    return {
        "target_pixel_x": float(target_x),
        "target_pixel_y": float(target_y),
        "frame_width": float(frame_width),
        "frame_height": float(frame_height),
        "target_source": source,
        "estimated_z_m": z_m,
    }


# Tell the robot receiver that a person file has gone stale.
# This is how the downstream script learns that someone has effectively left.
async def send_person_absent(sender: PredictionSender, file_path: Path, person_id: str, last_frame_id: Optional[int]):
    # This is the message consumed by the robot-side controller.
    payload = {
        "person_id": person_id,
        "source_file": file_path.name,
        "timestamp": time.time(),
        "last_frame_id": last_frame_id,
        "present": False,
        "interacting": "No",
        "stable_interacting": "No",
        "intention": None,
        "stable_intention": None,
        "attitude": None,
        "action": None,
        "intention_confidence": 0.0,
        "attitude_confidence": 0.0,
        "action_confidence": 0.0,
        "intention_probabilities": {},
        "attitude_probabilities": {},
        "reason": "stale_person_file",
    }
    await sender.send(payload)
    print(json.dumps(payload, indent=2))


# Main live loop. It keeps scanning the JSON folder, skips old frames,
# runs predictions on new ones, and forwards the result to the robot side.
async def main():
    parser = argparse.ArgumentParser(description="Real-time SocialEgoNet prediction sender for Furhat")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--check_point", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Folder where AlphaPose writes JSONs")
    parser.add_argument("--ws_url", type=str, default="ws://127.0.0.1:8765")
    parser.add_argument("--poll_interval", type=float, default=0.03)
    parser.add_argument("--stability", type=int, default=1)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--min_body_conf", type=float, default=0.25)
    parser.add_argument("--min_face_conf", type=float, default=0.35)
    parser.add_argument("--person_file_stale_time", type=float, default=0.75,
                        help="If a person JSON file has not been updated for this long, send present=False and interacting=No")
    parser.add_argument("--require_target_for_yes", action="store_true",
                        help="Only allow interacting=Yes when a valid head target is available")
    parser.add_argument("--no_intention_prob_threshold", type=float, default=0.90,
                        help="Only predict interacting=No when P(Not_Interested) is at least this high")
    parser.add_argument("--no_attitude_prob_threshold", type=float, default=0.90,
                        help="Only predict interacting=No when P(Not_Interacting) is at least this high")
    args = parser.parse_args()

    config = load_config(args.cfg)
    sequence_length = config["data"]["sequence_length"]

    # Build the exact model described by the config file, then load the trained weights.
    model = SocialEgoNet(sequence_length=sequence_length, **config["model"])
    model.load_checkpoint(args.check_point)
    model.to(device)
    model.eval()

    sender = PredictionSender(args.ws_url)
    data_dir = Path(args.data_dir)

    # Per-person state so multiple JSON files can be handled at once.
    file_states: Dict[str, dict] = defaultdict(lambda: {
        "last_frame_id": None,
        "intention_history": deque(maxlen=max(1, args.stability)),
        "interacting_history": deque(maxlen=max(1, args.stability)),
        "sent_absent": False,
        "last_file_mtime": 0.0,
    })

    print(f"Watching {data_dir} and sending predictions to {args.ws_url}")
    print(f"Sequence length: {sequence_length}")

    try:
        while True:
            now = time.time()
            # Each JSON file represents one tracked person from the AlphaPose stage.
            for file_path in sorted(data_dir.glob("*.json")):
                feature_json = safe_load_json(file_path)
                if feature_json is None:
                    continue

                person_id = str(feature_json.get("person_id", file_path.stem))
                state = file_states[person_id]

                try:
                    file_mtime = file_path.stat().st_mtime
                except OSError:
                    file_mtime = now
                state["last_file_mtime"] = file_mtime

                # If AlphaPose stopped updating a person's file, mark them gone once.
                if (now - file_mtime) > args.person_file_stale_time:
                    if not state["sent_absent"]:
                        await send_person_absent(sender, file_path, person_id, state["last_frame_id"])
                        state["sent_absent"] = True
                        state["intention_history"].clear()
                        state["interacting_history"].clear()
                    continue

                x_tensor, last_frame_id, latest_frame = extract_live_window(feature_json, sequence_length)
                if last_frame_id is None or x_tensor is None or latest_frame is None:
                    continue

                # Skip files that have not advanced to a new frame yet.
                if state["last_frame_id"] == last_frame_id:
                    continue

                state["last_frame_id"] = last_frame_id
                state["sent_absent"] = False

                prediction = predict_single(
                    model,
                    x_tensor,
                    sequence_length,
                    no_intention_prob_threshold=args.no_intention_prob_threshold,
                    no_attitude_prob_threshold=args.no_attitude_prob_threshold,
                )

                if prediction["intention_confidence"] < args.min_confidence:
                    continue

                attention_target = extract_attention_target(
                    latest_frame,
                    frame_width=float(feature_json["frame_size"][0]),
                    frame_height=float(feature_json["frame_size"][1]),
                    min_body_conf=args.min_body_conf,
                    min_face_conf=args.min_face_conf,
                )

                # Optional extra guard so "Yes" is only sent when we can also point at the person.
                if args.require_target_for_yes and attention_target is None:
                    prediction["interacting"] = "No"

                state["intention_history"].append(prediction["intention"])
                state["interacting_history"].append(prediction["interacting"])

                # Stable fields only appear when the same output repeats enough times.
                stable_intention = None
                if len(state["intention_history"]) == state["intention_history"].maxlen and len(set(state["intention_history"])) == 1:
                    stable_intention = state["intention_history"][-1]

                stable_interacting = None
                if len(state["interacting_history"]) == state["interacting_history"].maxlen and len(set(state["interacting_history"])) == 1:
                    stable_interacting = state["interacting_history"][-1]

                payload = {
                    "person_id": person_id,
                    "source_file": file_path.name,
                    "timestamp": now,
                    "last_frame_id": last_frame_id,
                    "present": True,
                    "intention": prediction["intention"],
                    "intention_confidence": prediction["intention_confidence"],
                    "stable_intention": stable_intention,
                    "attitude": prediction["attitude"],
                    "action": prediction["action"],
                    "attitude_confidence": prediction["attitude_confidence"],
                    "action_confidence": prediction["action_confidence"],
                    "interacting": prediction["interacting"],
                    "stable_interacting": stable_interacting,
                    "intention_probabilities": prediction["intention_probabilities"],
                    "attitude_probabilities": prediction["attitude_probabilities"],
                }
                if attention_target is not None:
                    payload.update(attention_target)

                await sender.send(payload)
                print(json.dumps(payload, indent=2))

            await asyncio.sleep(args.poll_interval)
    finally:
        await sender.close()


if __name__ == "__main__":
    asyncio.run(main())
