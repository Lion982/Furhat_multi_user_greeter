"""Robot-side receiver for live interaction predictions.

This file sits between the prediction script and Furhat itself.
It listens for prediction messages, decides who Furhat should pay attention to,
and turns those decisions into Furhat commands such as head movement and speech.

High-level flow:
1. Receive one JSON message per person from the prediction script.
2. Keep the newest message for each visible person in memory.
3. Decide who Furhat should lock onto.
4. Convert the chosen screen position into Furhat x/y/z coordinates.
5. Send attend/speak commands to Furhat through the Realtime API.

The logic is written to feel stable rather than twitchy:
- stale people are removed after a timeout
- tiny head target changes are ignored
- one person is treated as the current "primary" person
- a second person can be briefly acknowledged without fully stealing the interaction
"""

import argparse
import asyncio
import json
import time
from typing import Dict, Optional, Set, Tuple

try:
    import websockets
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'websockets'. Install it with: pip install websockets"
    ) from exc


# Latest prediction message we have for each tracked person.
latest_by_person: Dict[str, dict] = {}

# Person Furhat is currently focused on.
locked_person_id: Optional[str] = None
locked_last_seen: float = 0.0
locked_last_interacting: float = 0.0

# Smoothed head target so the robot does not jitter around.
smoothed_xyz: Optional[Tuple[float, float, float]] = None
last_commanded_xyz: Optional[Tuple[float, float, float]] = None
last_attend_sent: float = 0.0

# Keeps track of people Furhat has already glanced at and acknowledged.
acknowledged_waiting_people: Set[str] = set()
ack_task: Optional[asyncio.Task] = None
ack_active: bool = False

# Simple cooldown so "hello" is not repeated too often.
last_primary_greet_at: float = 0.0


# Thin wrapper around Furhat's Realtime API.
# The rest of the file can just call friendly methods like attend_location()
# without worrying about the low-level WebSocket details.
class FurhatClient:
    def __init__(self, furhat_ip: str, furhat_key: str, furhat_port: int = 9000):
        self.furhat_ip = furhat_ip
        self.furhat_key = furhat_key
        self.furhat_port = furhat_port
        self.ws_url = f"ws://{furhat_ip}:{furhat_port}/v1/events"
        self.websocket = None
        self.authenticated = False
        self.send_lock = asyncio.Lock()

    async def connect(self):
        if self.websocket is not None:
            try:
                if not self.websocket.closed:
                    return
            except AttributeError:
                pass

        self.websocket = await websockets.connect(self.ws_url)
        self.authenticated = False
        await self.authenticate()

    async def authenticate(self):
        auth_payload = {
            "type": "request.auth",
            "key": self.furhat_key,
        }
        await self.websocket.send(json.dumps(auth_payload))

        raw = await self.websocket.recv()
        response = json.loads(raw)

        if response.get("type") != "response.auth" or not response.get("access", False):
            raise RuntimeError(f"Furhat authentication failed: {response}")

        self.authenticated = True
        print(f"[FURHAT] authenticated to {self.ws_url} (scope={response.get('scope')})")

    async def send_event(self, payload: dict):
        # One send at a time so messages do not collide on the socket.
        async with self.send_lock:
            try:
                await self.connect()
                await self.websocket.send(json.dumps(payload))
            except Exception:
                # If the socket dropped, reconnect once and retry.
                self.websocket = None
                self.authenticated = False
                await self.connect()
                await self.websocket.send(json.dumps(payload))

    async def attend_location(
        self,
        x: float,
        y: float,
        z: float,
        speed: str = "medium",
        slack_pitch: float = 15.0,
        slack_yaw: float = 5.0,
        slack_timeout: int = 3000,
    ):
        payload = {
            "type": "request.attend.location",
            "x": x,
            "y": y,
            "z": z,
            "speed": speed,
            "slack_pitch": slack_pitch,
            "slack_yaw": slack_yaw,
            "slack_timeout": slack_timeout,
        }
        await self.send_event(payload)
        print(
            f"[FURHAT] attend.location x={x:.3f} y={y:.3f} z={z:.3f} "
            f"speed={speed}"
        )

    async def attend_nobody(self):
        await self.send_event({"type": "request.attend.nobody"})
        print("[FURHAT] attend.nobody")

    async def speak_text(self, text: str, abort: bool = False):
        payload = {
            "type": "request.speak.text",
            "text": text,
            "abort": abort,
        }
        await self.send_event(payload)
        print(f"[FURHAT] speak.text text={text!r} abort={abort}")

    async def close(self):
        if self.websocket is not None:
            try:
                await self.websocket.close()
            except Exception:
                pass


# Convert a point in image space into the rough 3D coordinate system
# Furhat expects for attend.location. This is only an approximation, but
# it is usually good enough for head tracking.
def image_to_robot_coords(
    target_px_x: float,
    target_px_y: float,
    frame_width: float,
    frame_height: float,
    z_m: float,
) -> Tuple[float, float, float]:
    # Convert image position into a rough 3D target in front of Furhat.
    x_norm = (target_px_x - frame_width / 2.0) / (frame_width / 2.0)
    y_norm = (target_px_y - frame_height / 2.0) / (frame_height / 2.0)

    # These are just practical scaling factors for this setup.
    max_x_at_1m = 0.55
    max_y_at_1m = 0.35

    x_m = float(x_norm * max_x_at_1m * z_m)
    y_m = float(-y_norm * max_y_at_1m * z_m)
    return x_m, y_m, z_m


# Blend the newest target with the previous one so the head does not
# jump sharply every time the detected point moves by a few pixels.
def smooth_xyz(new_xyz: Tuple[float, float, float], alpha: float) -> Tuple[float, float, float]:
    global smoothed_xyz
    if smoothed_xyz is None:
        smoothed_xyz = new_xyz
    else:
        # Blend new target with the previous one so head motion stays calmer.
        smoothed_xyz = tuple(
            alpha * old + (1.0 - alpha) * new
            for old, new in zip(smoothed_xyz, new_xyz)
        )
    return smoothed_xyz


# Pick the interaction label the robot should trust from a prediction message.
# If a person is marked absent we force this to "No" immediately.
def choose_interacting(message: dict) -> Optional[str]:
    if message.get("present") is False:
        return "No"
    return message.get("interacting") or message.get("stable_interacting")


# Decide whether a message is strong enough to become the robot's current focus.
# We only lock if the person is present, counted as interacting, and
# has a usable screen target to look at.
def should_lock(message: dict) -> bool:
    # We only lock onto a person if they are present, interacting,
    # and we know where to point the head.
    return (
        message.get("present", True) is True
        and choose_interacting(message) == "Yes"
        and "target_pixel_x" in message
        and "target_pixel_y" in message
    )


# Clear everything tied to the current primary person.
# This gets called when the person disappears, stops interacting for too long,
# or when the server decides to switch to someone else.
def release_lock(reason: str):
    global locked_person_id, locked_last_seen, locked_last_interacting, smoothed_xyz, last_commanded_xyz
    if locked_person_id is not None:
        print(f"[LOCK] released {locked_person_id} because {reason}")
    locked_person_id = None
    locked_last_seen = 0.0
    locked_last_interacting = 0.0
    smoothed_xyz = None
    last_commanded_xyz = None


# Fresh means we have heard about this person recently enough that their
# data is still safe to use for robot decisions.
def person_record_is_fresh(record: dict, now: float, stale_after: float) -> bool:
    return (now - float(record.get("received_at", 0.0))) <= stale_after


# Remove old person entries so the robot does not keep considering people
# who have actually left the frame.
def prune_stale_people(now: float, stale_after: float):
    # Drop people whose prediction files stopped updating.
    stale_ids = [
        pid for pid, record in latest_by_person.items()
        if not person_record_is_fresh(record, now, stale_after)
    ]
    for pid in stale_ids:
        latest_by_person.pop(pid, None)
        acknowledged_waiting_people.discard(pid)


# Forget people we already acknowledged once they are gone, stale, or no longer
# interacting. That lets them be acknowledged again later if they reappear.
def cleanup_acknowledged_waiting(now: float, stale_after: float):
    # Clear out anyone who disappeared or is no longer waiting.
    removable = set()
    for pid in acknowledged_waiting_people:
        record = latest_by_person.get(pid)
        if record is None:
            removable.add(pid)
            continue
        if not person_record_is_fresh(record, now, stale_after):
            removable.add(pid)
            continue
        if choose_interacting(record.get("message", {})) != "Yes":
            removable.add(pid)
    for pid in removable:
        acknowledged_waiting_people.discard(pid)


# From everyone currently visible, pick the strongest candidate to become
# the main person Furhat should focus on. The sort order favours higher
# confidence first, then newer messages.
def get_best_interacting_candidate(now: float, args) -> Optional[dict]:
    candidates = []
    for pid, record in latest_by_person.items():
        if not person_record_is_fresh(record, now, args.person_stale_time):
            continue
        msg = record.get("message", {})
        if not should_lock(msg):
            continue
        # Prefer the strongest interacting candidate, then the freshest one.
        candidates.append((
            float(msg.get("attitude_confidence", 0.0)),
            float(record.get("received_at", 0.0)),
            pid,
            msg,
        ))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _, _, pid, msg = candidates[0]
    return {"person_id": pid, "message": msg}


# Pull back the newest valid message for the currently locked person.
# If they have gone stale, this returns None and the lock will be released.
def get_locked_message(now: float, args) -> Optional[dict]:
    if locked_person_id is None:
        return None
    record = latest_by_person.get(locked_person_id)
    if record is None:
        return None
    if not person_record_is_fresh(record, now, args.person_stale_time):
        return None
    return record.get("message")


# Ignore tiny movement requests. Small changes usually come from pose noise
# and make the head look shaky rather than helpful.
def xyz_change_big_enough(new_xyz: Tuple[float, float, float], args) -> bool:
    global last_commanded_xyz
    if last_commanded_xyz is None:
        return True
    dx = abs(new_xyz[0] - last_commanded_xyz[0])
    dy = abs(new_xyz[1] - last_commanded_xyz[1])
    dz = abs(new_xyz[2] - last_commanded_xyz[2])
    return (
        dx >= args.deadband_x_m
        or dy >= args.deadband_y_m
        or dz >= args.deadband_z_m
    )


# Read the target pixel fields from a prediction message and turn them into
# the x/y/z values used by Furhat attend.location.
def message_to_xyz(message: dict, default_z: float) -> Optional[Tuple[float, float, float]]:
    target_px_x = message.get("target_pixel_x")
    target_px_y = message.get("target_pixel_y")
    frame_width = message.get("frame_width")
    frame_height = message.get("frame_height")
    z_m = float(message.get("estimated_z_m", default_z))

    if None in (target_px_x, target_px_y, frame_width, frame_height):
        return None

    return image_to_robot_coords(
        float(target_px_x),
        float(target_px_y),
        float(frame_width),
        float(frame_height),
        z_m,
    )


# Look for a second interacting person while Furhat is already busy with
# someone else. This powers the quick glance + "I'll be with you in a moment"
# behaviour without fully dropping the first person.
def maybe_choose_waiting_candidate(now: float, args) -> Optional[dict]:
    global ack_task, ack_active
    if locked_person_id is None or ack_active:
        return None
    if ack_task is not None and not ack_task.done():
        return None

    locked_message = get_locked_message(now, args)
    if locked_message is None or choose_interacting(locked_message) != "Yes":
        return None

    candidates = []
    for pid, record in latest_by_person.items():
        if pid == locked_person_id:
            continue
        if pid in acknowledged_waiting_people:
            continue
        if not person_record_is_fresh(record, now, args.person_stale_time):
            continue
        msg = record.get("message", {})
        if not should_lock(msg):
            continue
        # Pick the best waiting person we have not acknowledged yet.
        candidates.append((
            float(record.get("received_at", 0.0)),
            float(msg.get("attitude_confidence", 0.0)),
            pid,
            msg,
        ))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _, _, pid, msg = candidates[0]
    return {"person_id": pid, "message": msg}


# Start the temporary acknowledgement task for a waiting second person
# if the current situation allows it.
async def maybe_schedule_waiting_ack(furhat: FurhatClient, args):
    global ack_task
    now = time.time()
    cleanup_acknowledged_waiting(now, args.person_stale_time)
    candidate = maybe_choose_waiting_candidate(now, args)
    if candidate is None:
        return
    waiting_person_id = candidate["person_id"]
    original_person_id = locked_person_id
    if original_person_id is None:
        return
    acknowledged_waiting_people.add(waiting_person_id)
    print(f"[ACK] acknowledging waiting person {waiting_person_id} while staying with {original_person_id}")
    ack_task = asyncio.create_task(
        acknowledge_waiting_person(
            waiting_person_id=waiting_person_id,
            original_person_id=original_person_id,
            furhat=furhat,
            args=args,
        )
    )


# Say the main greeting when a new primary person is chosen, but respect
# a cooldown so Furhat does not repeat hello too often.
async def greet_primary_person(furhat: FurhatClient, args):
    global last_primary_greet_at
    now = time.time()
    if (now - last_primary_greet_at) < args.hello_cooldown:
        return
    last_primary_greet_at = now
    try:
        await furhat.speak_text(args.primary_greeting, abort=True)
    except Exception as exc:
        print(f"[SPEECH] primary greeting failed: {exc}")


# Briefly look at a second person, say the waiting message, then return
# to the original primary person. This runs as its own task so the main
# message loop does not freeze while the acknowledgement is happening.
async def acknowledge_waiting_person(waiting_person_id: str, original_person_id: str, furhat: FurhatClient, args):
    global ack_active, smoothed_xyz, last_attend_sent, last_commanded_xyz, ack_task
    ack_active = True
    try:
        waiting_record = latest_by_person.get(waiting_person_id)
        waiting_message = waiting_record.get("message") if waiting_record else None
        waiting_xyz = message_to_xyz(waiting_message, args.default_z) if waiting_message else None

        if waiting_xyz is not None:
            # Reset smoothing so the quick glance feels direct.
            smoothed_xyz = None
            last_commanded_xyz = None
            last_attend_sent = 0.0
            await furhat.attend_location(
                waiting_xyz[0],
                waiting_xyz[1],
                waiting_xyz[2],
                speed=args.ack_speed,
                slack_pitch=args.ack_slack_pitch,
                slack_yaw=args.ack_slack_yaw,
                slack_timeout=args.slack_timeout,
            )
            await asyncio.sleep(args.ack_pre_speech_delay)

        await furhat.speak_text(args.waiting_greeting, abort=False)
        await asyncio.sleep(args.ack_hold_time)

        original_record = latest_by_person.get(original_person_id)
        original_message = original_record.get("message") if original_record else None
        original_xyz = message_to_xyz(original_message, args.default_z) if original_message else None
        if original_xyz is not None and locked_person_id == original_person_id:
            # Reset again so the return movement is clean too.
            smoothed_xyz = None
            last_commanded_xyz = None
            last_attend_sent = 0.0
            await furhat.attend_location(
                original_xyz[0],
                original_xyz[1],
                original_xyz[2],
                speed=args.return_speed,
                slack_pitch=args.slack_pitch,
                slack_yaw=args.slack_yaw,
                slack_timeout=args.slack_timeout,
            )
            await asyncio.sleep(args.return_settle_time)
    except Exception as exc:
        print(f"[ACK] waiting acknowledgement failed: {exc}")
    finally:
        ack_active = False
        ack_task = None


# Main decision function. Every incoming message eventually passes through here.
# This is where the script decides who to lock onto, when to let go, when to
# greet, and when to send a fresh head target to Furhat.
async def handle_robot_behaviour(message: dict, furhat: FurhatClient, args):
    global locked_person_id, locked_last_seen, locked_last_interacting, last_attend_sent, smoothed_xyz, last_commanded_xyz

    now = time.time()
    prune_stale_people(now, args.person_stale_time)
    cleanup_acknowledged_waiting(now, args.person_stale_time)

    locked_message = get_locked_message(now, args)

    if locked_person_id is not None and locked_message is None:
        release_lock("current person is stale or missing")
        await furhat.attend_nobody()

    locked_message = get_locked_message(now, args)

    if locked_person_id is None:
        # Try to find the best person to focus on right now.
        candidate = get_best_interacting_candidate(now, args)
        if candidate is None:
            return
        locked_person_id = candidate["person_id"]
        locked_last_seen = now
        locked_last_interacting = now
        smoothed_xyz = None
        locked_message = candidate["message"]
        print(f"[LOCK] locked onto {locked_person_id}")
        asyncio.create_task(greet_primary_person(furhat, args))

    if locked_person_id is not None and (now - locked_last_seen) > args.lock_timeout:
        # Hard timeout in case we somehow keep a lock longer than we should.
        release_lock("timeout")
        await furhat.attend_nobody()
        candidate = get_best_interacting_candidate(now, args)
        if candidate is None:
            return
        locked_person_id = candidate["person_id"]
        locked_last_seen = now
        locked_last_interacting = now
        smoothed_xyz = None
        locked_message = candidate["message"]
        print(f"[LOCK] locked onto {locked_person_id}")
        asyncio.create_task(greet_primary_person(furhat, args))

    locked_message = get_locked_message(now, args)
    if locked_message is None:
        return

    currently_interacting = choose_interacting(locked_message) == "Yes"
    if currently_interacting:
        locked_last_interacting = now
    elif (now - locked_last_interacting) >= args.release_after_non_interacting:
        # Give the current person a small grace period before dropping them.
        old_person = locked_person_id
        release_lock(f"{old_person} was not interacting for {args.release_after_non_interacting:.2f}s")
        candidate = get_best_interacting_candidate(now, args)
        if candidate is None:
            await furhat.attend_nobody()
            return
        locked_person_id = candidate["person_id"]
        locked_last_seen = now
        locked_last_interacting = now
        smoothed_xyz = None
        locked_message = candidate["message"]
        currently_interacting = True
        print(f"[LOCK] switched to {locked_person_id}")
        asyncio.create_task(greet_primary_person(furhat, args))

    if locked_person_id is None or locked_message is None:
        return

    locked_last_seen = now
    await maybe_schedule_waiting_ack(furhat, args)

    if ack_active:
        # Do not fight with the temporary acknowledgement movement.
        return

    # Turn the chosen person's image target into robot coordinates.
    raw_xyz = message_to_xyz(locked_message, args.default_z)
    if raw_xyz is None:
        return

    # Smooth the target before sending it on to Furhat.
    x_m, y_m, z_m = smooth_xyz(raw_xyz, alpha=args.smoothing)

    if (now - last_attend_sent) < args.min_send_interval:
        return

    if not xyz_change_big_enough((x_m, y_m, z_m), args):
        return

    last_attend_sent = now
    last_commanded_xyz = (x_m, y_m, z_m)
    await furhat.attend_location(
        x_m,
        y_m,
        z_m,
        speed=args.speed,
        slack_pitch=args.slack_pitch,
        slack_yaw=args.slack_yaw,
        slack_timeout=args.slack_timeout,
    )


# Safety net for cases where messages stop arriving. Without this, the robot
# might keep a stale lock forever if the upstream sender dies silently.
async def periodic_state_check(furhat: FurhatClient, args):
    # Keeps release logic running even if no new predictor messages arrive.
    while True:
        try:
            await handle_robot_behaviour({}, furhat, args)
        except Exception as exc:
            print(f"[STATE] periodic check error: {exc}")
        await asyncio.sleep(args.state_check_interval)


# Read prediction messages from the upstream script one by one.
# Each message updates the latest known state for a person, then the robot
# behaviour logic is run again.
async def on_message(websocket, furhat: FurhatClient, args):
    async for raw_message in websocket:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            continue

        person_id = message.get("person_id", "unknown")
        if message.get("present") is False:
            latest_by_person.pop(person_id, None)
            acknowledged_waiting_people.discard(person_id)
            print(json.dumps(message, indent=2))
            if locked_person_id == person_id:
                release_lock("predictor marked current person absent")
                await furhat.attend_nobody()
            await handle_robot_behaviour(message, furhat, args)
            continue

        # Store the newest message for this person. Older information for the same
        # person is replaced because only the latest state matters for live control.
        latest_by_person[person_id] = {"message": message, "received_at": time.time()}
        print(json.dumps(message, indent=2))
        await handle_robot_behaviour(message, furhat, args)


# Parse command-line options, connect the pieces together, and start the
# WebSocket server that receives predictions from the model side.
async def main():
    parser = argparse.ArgumentParser(
        description="Robot-side WebSocket receiver for SocialEgoNet predictions and Furhat attention"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--furhat_ip", type=str, required=True)
    parser.add_argument("--furhat_key", type=str, required=True)
    parser.add_argument("--furhat_port", type=int, default=9000)
    parser.add_argument("--lock_timeout", type=float, default=2.0)
    parser.add_argument("--release_after_non_interacting", type=float, default=2.0,
                        help="Hold the current person for this long after they stop interacting")
    parser.add_argument("--person_stale_time", type=float, default=2.0,
                        help="How long to keep a person candidate without a fresh message")
    parser.add_argument("--min_send_interval", type=float, default=0.06)
    parser.add_argument("--smoothing", type=float, default=0.15, help="Higher = smoother / less reactive")
    parser.add_argument("--deadband_x_m", type=float, default=0.03,
                        help="Ignore tiny left-right target changes smaller than this")
    parser.add_argument("--deadband_y_m", type=float, default=0.03,
                        help="Ignore tiny up-down target changes smaller than this")
    parser.add_argument("--deadband_z_m", type=float, default=0.08,
                        help="Ignore tiny distance changes smaller than this")
    parser.add_argument("--default_z", type=float, default=1.2)
    parser.add_argument("--speed", type=str, default="fast")
    parser.add_argument("--slack_pitch", type=float, default=12.0)
    parser.add_argument("--slack_yaw", type=float, default=4.0)
    parser.add_argument("--slack_timeout", type=int, default=3000)
    parser.add_argument("--state_check_interval", type=float, default=0.10,
                        help="How often to run release/idle logic even when no messages arrive")
    parser.add_argument("--primary_greeting", type=str, default="Hello, how can I be of service?")
    parser.add_argument("--waiting_greeting", type=str, default="Hi, I will be with you in a moment")
    parser.add_argument("--hello_cooldown", type=float, default=1.0,
                        help="Minimum spacing between primary greetings")
    parser.add_argument("--ack_hold_time", type=float, default=1.0,
                        help="How long to keep attention on the waiting person before returning")
    parser.add_argument("--ack_pre_speech_delay", type=float, default=0.10,
                        help="Short pause after glancing to the waiting person before speaking")
    parser.add_argument("--return_settle_time", type=float, default=0.10,
                        help="Short pause after returning to the original person")
    parser.add_argument("--ack_speed", type=str, default="fast",
                        help="Head movement speed used for the temporary waiting-person acknowledgement")
    parser.add_argument("--return_speed", type=str, default="fast",
                        help="Head movement speed used when returning to the original person")
    parser.add_argument("--ack_slack_pitch", type=float, default=8.0,
                        help="Pitch slack while glancing to the waiting person")
    parser.add_argument("--ack_slack_yaw", type=float, default=2.5,
                        help="Yaw slack while glancing to the waiting person")
    args = parser.parse_args()

    furhat = FurhatClient(
        furhat_ip=args.furhat_ip,
        furhat_key=args.furhat_key,
        furhat_port=args.furhat_port,
    )

    state_task = asyncio.create_task(periodic_state_check(furhat, args))

    async with websockets.serve(lambda ws: on_message(ws, furhat, args), args.host, args.port):
        print(f"Robot intention server listening on ws://{args.host}:{args.port}")
        print(f"Forwarding attend commands to Furhat at ws://{args.furhat_ip}:{args.furhat_port}/v1/events")
        try:
            await asyncio.Future()
        finally:
            state_task.cancel()
            try:
                await state_task
            except asyncio.CancelledError:
                pass
            if ack_task is not None:
                ack_task.cancel()
                try:
                    await ack_task
                except asyncio.CancelledError:
                    pass
            await furhat.close()


if __name__ == "__main__":
    asyncio.run(main())
