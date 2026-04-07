# Furhat Interaction Pipeline Overview

This folder contains the commented live-control files plus this README.

The system is split into separate stages so each part has one clear job:
- get frames and body keypoints
- turn keypoints into SocialEgoNet predictions
- turn predictions into robot behaviour

That makes it much easier to test and debug.

## Files in the overall system

### 1. `demo_inference_furhat_commented.py`
This is the **front end of the perception pipeline**.

Its job is to:
- connect to Furhat and request live camera frames
- run AlphaPose on each frame
- separate detections by person
- write one JSON file per person in the custom format used by the next stage

In simple terms, this file answers:
**"Who is in front of the robot, and what are their body keypoints over time?"**

---

### 2. `prediction.py`
This is the **simple offline prediction script**.

Its job is to:
- load a folder of pose JSON files
- build a SocialEgoNet dataloader
- run the model on each sample
- take the top class from each output using `argmax`
- save the final labels to `predictions.json`

It outputs:
- `intention`
- `attitude`
- `action`

It does **not**:
- send live messages to Furhat
- estimate a head target for Furhat to look at
- track whether a person has disappeared
- create a special live `interacting = Yes/No` control field
- smooth or stabilise behaviour over time

In simple terms, this file answers:
**"If I run SocialEgoNet on saved pose files, what class labels do I get?"**

---

### 3. `realtime_predict_ws_furhat_v5.py`
This is the **live prediction stage**.

Its job is to:
- watch the JSON files written by the AlphaPose stage
- pull out the newest sequence of frames for each person
- convert those frames into the graph input SocialEgoNet expects
- run SocialEgoNet live
- predict intention, attitude, and action
- turn those outputs into a simpler live control label such as `interacting = Yes/No`
- estimate a sensible point for Furhat to look at
- send the result to the robot-side controller

In simple terms, this file answers:
**"Out of the people currently visible, who looks like they want to interact right now?"**

---

### 4. `robot_intention_server_furhat_v5_speak.py`
This is the **robot behaviour stage**.

Its job is to:
- receive live prediction messages
- keep track of all visible people
- decide who Furhat should focus on right now
- send `attend.location`, `attend.nobody`, and `speak.text` commands to Furhat
- briefly acknowledge a second person if someone else is already being served

In simple terms, this file answers:
**"Given the predictions, what should the robot do right now?"**

---

## Full data flow

The whole live system works like this:

1. **Furhat camera** provides live images.
2. **`demo_inference_furhat_commented.py`** runs AlphaPose and writes per-person JSON files.
3. **`realtime_predict_ws_furhat_v5.py`** reads those JSON files and runs SocialEgoNet.
4. It sends one live message per person to **`robot_intention_server_furhat_v5_speak.py`**.
5. The robot-side file decides where Furhat should look and what it should say.
6. **Furhat** moves its head and speaks through the Realtime API.

The offline path is simpler:

1. Pose JSON files already exist.
2. **`prediction.py`** loads them through the dataset/dataloader.
3. It outputs one set of labels per file.
4. Nothing is sent to the robot.

---

## Why `prediction.py` and `realtime_predict_ws_furhat_v5.py` are different

These two files both use SocialEgoNet, but they are for different jobs.

### `prediction.py`
This is mainly for **offline testing**.
It takes each model output and simply chooses the biggest logit using `argmax`.
That gives one final label for:
- intention
- attitude
- action

This is fine when you only want saved predictions from saved files.

### `realtime_predict_ws_furhat_v5.py`
This is for **live robot control**.
A live robot needs more than just three class labels. It also needs to know:
- is the person still present?
- should I treat this person as interacting right now?
- where should I look?
- has the result been stable for long enough?

So the realtime file adds extra live logic on top of the raw model output.

---

## Very simple explanation of the 0.9 probability rule

When we used the more direct logic from `prediction.py`, the live system became less accurate in practice.
That happened because `prediction.py` just takes the **top class**. In live use, the top class can flip on noisy frames, especially when the model is not strongly sure.

So in `realtime_predict_ws_furhat_v5.py`, we made the rule more careful:
- only say `interacting = No` if `P(Not_Interested) >= 0.90`
- **and** `P(Not_Interacting) >= 0.90`
- otherwise keep `interacting = Yes`

This means:
- a person is only marked as **not interacting** when the model is very confident in both of the negative signals
- if the model is unsure, the robot does not give up too easily

A very simple way to think about it is:

**Old idea:**
"Whichever class is biggest wins immediately."

**New idea:**
"Only call someone not interacting when the model is really sure."

This was added because the direct argmax-style approach worked less well for the live system.

---

## Why this makes sense in a live system

A live system is noisier than an offline script.
Small changes in pose, missed keypoints, short partial views, or awkward frames can make the top class jump around.

The interaction-initiation paper also shows that this kind of timing decision is not perfectly stable, and that some classes are harder to separate than others. The timing classifier achieved about 73.6% accuracy overall, with a lower macro F1 of 69%, and the paper notes that one class is harder to distinguish because the difference can be very small. That is another reason to be more conservative in the live robot controller. fileciteturn5file12

So the 0.9 rule is basically a **confidence check** to stop the robot from switching to `No` too easily.

---

## How `prediction.py` works

The script:
1. loads the config
2. builds a `JPL_Social_Dataset`
3. builds a `JPL_Social_DataLoader`
4. loads the trained SocialEgoNet checkpoint
5. runs the model on each batch
6. uses `torch.argmax(...)` for intention, attitude, and action
7. saves the final labels to a JSON file

It does not calculate softmax probabilities for decision-making. It just picks the index of the largest output for each task. fileciteturn5file0

---

## How `realtime_predict_ws_furhat_v5.py` works

This file is designed for live use rather than offline evaluation.

### Main loop
For every JSON file in the watched folder, it:

1. loads the file safely
2. checks whether the file is stale
3. extracts the newest `sequence_length` frames
4. skips the file if nothing new has appeared
5. converts the pose window into graph data
6. runs SocialEgoNet
7. applies `softmax` so the outputs become probabilities
8. calculates `p_not_interested` and `p_not_interacting`
9. uses the 0.9 thresholds to decide `interacting = Yes/No`
10. estimates a target point for Furhat to look at
11. sends the message onward

Unlike `prediction.py`, this file does use softmax probabilities and a special binary control rule for interaction. It sets `interacting = No` only when both negative probabilities are above the chosen thresholds, which default to 0.90. fileciteturn5file2 fileciteturn5file6

### Why there is a stability buffer
The script keeps short histories for `intention` and `interacting`.
That allows it to compute `stable_intention` and `stable_interacting` when the same result repeats.
This is useful because single updates can be noisy.

### Extra live checks
The realtime file also:
- marks a person absent if their file stops updating
- can refuse to say `Yes` if there is no valid head target
- sends target coordinates for Furhat head movement

These are all things the offline script does not need. fileciteturn5file5 fileciteturn5file6

---

## How `demo_inference_furhat_commented.py` works

This script requests frames from Furhat using the Realtime API.
Each frame is passed through the AlphaPose pipeline:
- person detection
- crop / transform
- pose model forward pass
- heatmap decoding
- optional tracking

The results are repacked into the custom JSON format.

### Important behaviour in this file
- each run gets a new timestamped prefix
- each person gets their own JSON file
- a person only becomes a saved JSON once they have been seen for enough frames
- short false detections are kept out by `person_min_frames`

This stage is mostly about producing stable pose sequences for later prediction. fileciteturn5file18

---

## How `robot_intention_server_furhat_v5_speak.py` works

This file is the decision layer between predictions and robot actions.

### Person memory
The file stores the latest message for each visible person in `latest_by_person`.
That means Furhat can compare people instead of reacting only to the newest message.

### Locking onto one person
The main idea is that Furhat should usually have one current person it is serving.
That person is stored in `locked_person_id`.

A person can become the lock if:
- they are present
- they are predicted as interacting
- they have a valid target point to look at

### Releasing the lock
The lock is dropped if:
- the person goes stale
- the person disappears
- the person has not been interacting for long enough

### Head movement
The script does not send every tiny movement directly to Furhat.
It first:
- converts image coordinates into robot coordinates
- smooths the movement
- ignores tiny changes using deadbands
- rate-limits how often commands are sent

This is why the head movement looks steadier. fileciteturn5file13 fileciteturn5file19

### Greeting logic
When Furhat chooses a new primary person, it can say the main greeting.
A cooldown stops it from repeating the greeting too often.

### Second-person acknowledgement
If one person is already being served and another person starts interacting,
Furhat can briefly:
- glance at the second person
- say a waiting message
- return to the first person

That behaviour is handled in a separate async task so the main message loop keeps running. fileciteturn5file16 fileciteturn5file17

---

## Typical run order

Start the scripts in this order:

1. start the robot-side server: `robot_intention_server_furhat_v5_speak.py`
2. start the live predictor: `realtime_predict_ws_furhat_v5.py`
3. start the Furhat/AlphaPose capture script: `demo_inference_furhat_commented.py`

For offline testing only:
1. make sure the pose JSON files already exist
2. run `prediction.py`
3. inspect the output JSON

---

## Settings that matter most

### In `demo_inference_furhat_commented.py`
- `--furhat_fps`: how often to request frames
- `--person_iou_threshold`: how strict fallback matching is
- `--person_max_missing`: how long a person can disappear before being treated as new
- `--person_min_frames`: how long a track must last before a JSON is saved

### In `prediction.py`
- `--cfg`: config file
- `--check_point`: trained model weights
- `--output`: where to save predictions

### In `realtime_predict_ws_furhat_v5.py`
- `--poll_interval`: how often the folder is checked
- `--stability`: how many repeated labels are needed before something counts as stable
- `--person_file_stale_time`: when a person file is considered old
- `--no_intention_prob_threshold` and `--no_attitude_prob_threshold`: how hard it is for the script to say someone is not interacting

### In `robot_intention_server_furhat_v5_speak.py`
- `--lock_timeout`: how long to keep a lock with no updates
- `--release_after_non_interacting`: how long to hold onto a person after they stop interacting
- `--smoothing`: how smooth vs reactive the head movement is
- `--deadband_x_m`, `--deadband_y_m`, `--deadband_z_m`: how much movement is ignored
- `--primary_greeting` and `--waiting_greeting`: what Furhat says

---

## Where delays usually come from

In this design, delay usually comes more from:
- camera frame rate
- AlphaPose inference
- waiting for enough frames for SocialEgoNet
- JSON file polling
- smoothing and hold timers

and less from the link between scripts.

So if the system feels slow, the first places to inspect are usually:
- the frame rate
- sequence length
- poll interval
- stale timers
- smoothing / lock settings

---

## Good way to debug the system

If something looks wrong, check each stage separately.

### Stage 1 check
Do the person JSON files appear, and do they grow over time?

### Stage 2 check
Do the prediction messages contain sensible `intention`, `attitude`, `interacting`, and target coordinates?

### Stage 3 check
Does Furhat receive the messages and choose the correct person to lock onto?

### Offline check
Does `prediction.py` produce sensible labels on saved files before trying the live system?

Working stage-by-stage is much easier than trying to debug the whole pipeline at once.
