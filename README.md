Furhat Multi-User Greeter
This project is a live interaction pipeline for Furhat. It detects people in view, estimates their body pose, runs SocialEgoNet to predict social intent, and then decides how the robot should move and speak.
The system is split into separate stages so each file has one clear job:
get frames and pose keypoints
turn pose sequences into SocialEgoNet predictions
turn predictions into robot behaviour
That split makes the system easier to test, debug, and explain.
---
Files in the system
1. `demo_inference_furhat_commented.py`
This is the front end of the perception pipeline.
Its job is to:
connect to Furhat and request live camera frames
run AlphaPose on each frame
separate detections by person
write one JSON file per person in the custom format used by the next stage
In simple terms, this file answers:
"Who is in front of the robot, and what are their pose keypoints over time?"
---
2. `prediction.py`
This is the simple offline prediction script.
Its job is to:
load a folder of saved pose JSON files
build a SocialEgoNet dataset and dataloader
run the model on each sample
take the top class from each output using `argmax`
create a simple `interacting` field from the predicted attitude label
save the final results to `predictions.json`
It outputs:
`intention`
`attitude`
`action`
`interacting`
In this offline file, `interacting` is decided in a very direct way:
if `attitude == Not_Interacting`, then `interacting = No`
otherwise `interacting = Yes`
This is a simple saved-file prediction script. It does not:
send live messages to Furhat
estimate where Furhat should look
check whether a person has disappeared
stabilise results over time
use extra probability thresholds for live decision-making
In simple terms, this file answers:
"If I run SocialEgoNet on saved pose files, what labels do I get?"
---
3. `realtime_predict_ws_furhat_v5.py`
This is the live prediction stage.
Its job is to:
watch the JSON files written by the AlphaPose stage
pull out the newest sequence of frames for each person
convert those frames into the graph input SocialEgoNet expects
run SocialEgoNet live
predict intention, attitude, and action
create a live `interacting = Yes/No` decision
estimate a sensible point for Furhat to look at
send the result to the robot-side controller
In simple terms, this file answers:
"Out of the people currently visible, who looks like they want to interact right now?"
---
4. `robot_intention_server_furhat_v5_speak.py`
This is the robot behaviour stage.
Its job is to:
receive live prediction messages
keep track of all visible people
decide who Furhat should focus on right now
send `attend.location`, `attend.nobody`, and `speak.text` commands to Furhat
briefly acknowledge a second person if someone else is already being served
In simple terms, this file answers:
"Given the predictions, what should the robot do right now?"
---
Full data flow
Live path
Furhat camera provides live images.
`demo_inference_furhat_commented.py` runs AlphaPose and writes per-person JSON files.
`realtime_predict_ws_furhat_v5.py` reads those JSON files and runs SocialEgoNet.
It sends one live message per person to `robot_intention_server_furhat_v5_speak.py`.
The robot-side file decides where Furhat should look and what it should say.
Furhat moves its head and speaks through the Realtime API.
Offline path
Pose JSON files already exist.
`prediction.py` loads them through the dataset and dataloader.
It outputs one set of labels per file.
Nothing is sent to the robot.
---
How `prediction.py` works
The script:
loads the config
builds a `JPL_Social_Dataset`
builds a `JPL_Social_DataLoader`
loads the trained SocialEgoNet checkpoint
runs the model on each batch
uses `torch.argmax(...)` for intention, attitude, and action
creates a simple `interacting` field from the attitude result
saves the final labels to a JSON file
This file is mainly for offline testing and checking saved data.
---
How `realtime_predict_ws_furhat_v5.py` is different
Both `prediction.py` and `realtime_predict_ws_furhat_v5.py` use SocialEgoNet, but they are not doing the same job.
`prediction.py`
This is the simple offline version.
It just picks the biggest output for each head and saves the labels.
Its `interacting` field is made directly from the attitude label.
`realtime_predict_ws_furhat_v5.py`
This is for live robot control.
A live robot needs more than class labels. It also needs to know:
is the person still present?
should I treat this person as interacting right now?
where should I look?
has the result stayed stable for long enough?
So the realtime file adds extra live logic on top of the raw model output.
It also:
checks if a person file has gone stale
keeps short histories for more stable outputs
estimates a target point for head movement
sends the result to the robot controller
---
Very simple explanation of the 0.9 probability rule
When we tried using the exact same simple interaction logic from `prediction.py` in the live system, the accuracy dropped in practice.
The reason is simple:
`prediction.py` works on saved files and uses a direct label decision
live predictions are noisier because people move, keypoints can flicker, frames can be partial, and the model can be uncertain from one moment to the next
So in the realtime file, the interaction rule was made stricter.
Instead of saying No too easily, it only says:
`interacting = No`
when both of these are true:
`P(Not_Interested) >= 0.90`
`P(Not_Interacting) >= 0.90`
Otherwise it keeps:
`interacting = Yes`
Simple way to think about it
Offline rule in `prediction.py`:
> If the chosen attitude label is `Not_Interacting`, say No.
Live rule in `realtime_predict_ws_furhat_v5.py`:
> Only say No if the model is really sure.
That extra 0.9 check was added because the direct offline-style rule caused more mistakes in the live system.
---
Why this helps in live use
A live system is noisier than an offline script. Small changes in pose, missed detections, short partial views, or awkward frames can make the output jump around.
The stricter probability rule helps stop the robot from deciding someone is not interacting too quickly. That makes the behaviour more stable and less jumpy.
---
How `demo_inference_furhat_commented.py` works
This script requests frames from Furhat using the Realtime API.
Each frame is passed through the AlphaPose pipeline:
person detection
crop / transform
pose model forward pass
heatmap decoding
optional tracking
The results are repacked into the custom JSON format.
Important behaviour in this file
each run gets a new timestamped prefix
each person gets their own JSON file
a person only becomes a saved JSON once they have been seen for enough frames
short false detections are filtered out by `person_min_frames`
This stage is mostly about producing stable pose sequences for later prediction.
---
How `robot_intention_server_furhat_v5_speak.py` works
This file is the decision layer between predictions and robot actions.
Person memory
The file stores the latest message for each visible person. That means Furhat can compare people instead of reacting only to the newest message.
Locking onto one person
The main idea is that Furhat should usually have one current person it is serving. That person is stored as the locked person.
A person can become the lock if:
they are present
they are predicted as interacting
they have a valid target point to look at
Behaviour once locked
Once Furhat is locked onto someone, the file:
keeps updating the head target
smooths small movements
avoids sending tiny unnecessary position changes
releases the person if they disappear or stop interacting for long enough
Multi-person behaviour
If Furhat is already engaged with one person and someone else starts interacting, the file can:
glance toward the new person
say a short waiting message
return to the original person
This makes the robot feel more aware of multiple people without constantly switching focus.
---
Summary
The easiest way to think about the system is:
`demo_inference_furhat_commented.py` = get frames and pose data
`prediction.py` = simple offline SocialEgoNet prediction
`realtime_predict_ws_furhat_v5.py` = live SocialEgoNet prediction with extra control logic
`robot_intention_server_furhat_v5_speak.py` = turn predictions into Furhat behaviour
