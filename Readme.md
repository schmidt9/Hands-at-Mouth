# Hands at Mouth
Based on [Finger Counter](https://github.com/paveldat/finger_counter), see also article https://habr.com/ru/post/679460/

## Features
* Track lips and 2 hands in real-time using [MediaPipe framework](https://google.github.io/mediapipe/) 
* Detect cases where a hand touches/overlaps/covers lips using areas intersection detection
* Execute custom action on intersection event (as an example executes Google Chrome window minimization on Windows)

## How to install
1. Clone this repository on your computer
`https://github.com/schmidt9/Hands-at-Mouth.git`
2. Install all the requirements
`run libraries.bat` or
`pip install -r requirements.txt`
3. Run the program
`python main.py`

## Help
You might face issue with webcam not showing, and you get errors.
To solve it just change the value in this line (for example to `1`).
`capture = cv2.VideoCapture(0)`
Increment this number until you see your webcam.

## Result
TODO
