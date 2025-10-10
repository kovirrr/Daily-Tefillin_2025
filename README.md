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
4: it will tell you how to fix the tefillin's position when off

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

You may need to select intrepreter to the venv:
cmd+shift+p --> select interpreter --> .venv

### TO-DO

improve head positioning code. Apparently it just assumes

MAYBE: add a warning function for reference photos (only) if their yaw value is too far left or right
Turn into functional API
