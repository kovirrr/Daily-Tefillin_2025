### explanation of app in general

MAYBE = {}

An app that checks to see if you are wearing your tefillin correctly.
Every day you open the app and you take a picture with your tefillin on.
The app checks if it is on correctly by seeing if it is between your eyes and above your hairline.
If it is on correctly it updates your daily streak, accounting for the days where wearing tefillin is not mandatory (holidays and sabbath).
Otherwise it will tell you it is not on correctly and will ask you to take another picture (without instructions on how to fix).
If the hairline is not visible, it is ok because when you first download the app it asks for a picture of your holding your hair up and then from then on will be able to figure out where your hairline is from other facial features and calculating it.
If the picture taken is not in the same position as the reference photo it will give a warning.
If the user is bald or has a receding hairline it has an option to manually draw a hairline for the reference picture.
If you miss a day you lose your streak. You can add friends on the app and check their streaks.
There will also be a section on how to put on tefillin + brachot.

Premium (added features):
1: there will be a “group” section that, if you pay for premium, you will be able to create one. You can add people (even those without premium) into a group and it shows the leader of the group who wore tefillin on what day. (like attendance)
2: When your tefillin is not on correctly the app will tell you how to fix it so that it is.
3: an optional halacha of the day

{streak is public unless you have premium (optional then)}
{achievements like duolingo}
{FAQ page;}
{freeze strikes for premium}
{siddur on app (convenience) and it can give a monthly breakdown like spotify wrapped}
{worldwide photo group}
{davening page finder; listens to chazan and skips to where it thinks they are}

### This code

This is the Python code (future API) that returns the status of someone's tefillin (whether it's on correclty or not). It takes in the reference photo and the current tefillin photo

### Getting project up and running

RUN:
/opt/homebrew/bin/python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Run the placement-logic unit tests from the project root:

```sh
python -m unittest discover -s API/tests -v
```

These tests stub image detectors and verify decision logic; they do not test the
trained models, camera capture, or running the analysis on a phone.

### iPhone local-model feasibility test

The iOS-only benchmark exports the current hairline and train6 tefillin
checkpoints to Core ML and runs them on-device. It does not port `tef_good()`.

1. Activate `.venv311`, install Core ML Tools with
   `python -m pip install coremltools==9.0`, and run
   `python API/mobile_test/export_coreml.py`.
2. In Xcode, open `Frontend/ios/Frontend.xcworkspace`, select your iPhone, and
   add `--coreml-benchmark` under **Product > Scheme > Edit Scheme > Run >
   Arguments Passed On Launch**.
3. Run the app, choose a photo already on the phone, enable Airplane Mode, and
   tap **Run both models**. Repeat with varied photos and check that the green
   box follows the tefillin.

The report gives model inference timings and compiled model size. A confidence
score is not an accuracy measurement; verify detections on varied labeled
photos. Hairline output is reported but not rendered as a mask. The benchmark
does not include MediaPipe face landmarks or the full `tef_good()` check.
Remove the launch argument to return to the normal app.

You may need to select intrepreter to the venv:
cmd+shift+p --> select interpreter --> .venv

### TO-DO

improve head positioning code. Apparently it just assumes
plan:
Reference: In initialize, we will find the 3D coordinate of the detected hairline point.Live:
In calc_hairline, we will use the Rotation (R) and Translation (T) vectors (the output of cv2.solvePnP) to project the fixed 3D Model Points (Chin, Nose) into the 3D World Coordinate System of the camera.
We will calculate the 3D distance between the Chin and Nose in the current frame and use the pre-calculated 3D ratio to predict the Tefillin placement height.

MAYBE: add a warning function for reference photos (only) if their yaw value is too far left or right
Turn into functional API
