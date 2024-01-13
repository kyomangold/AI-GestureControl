
# AI Hand-Gesture Controlled Cursor with OAK-D Lite

This system integrates advanced AI capabilities to allow users to control their computer cursor through hand and gesture tracking, leveraging Google's MediaPipe and the OAK-D Lite camera's on-board processing. 

## Features

- Control the cursor with hand gestures.
- Different gestures for moving, clicking, and scrolling.
- Efficient processing on OAK-D Lite's onboard chip.
- Intuitive, natural human-computer interaction.

## Installation

Before you start, ensure you have the following packages installed:

```bash
python3 -m pip install -U pip
python3 -m pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ depthai
pip install numpy
pip install opencv-python
pip install pynput
pip install screeninfo
```

Ensure your OAK-D Lite camera is connected to your computer before proceeding.

## Usage

The system recognizes specific hand poses to control the cursor:

- **Move**: FIVE pose (open hand, move around freely)
- **Click**: FIST pose (close fist)
- **Scroll**: PEACE pose (index and middle finger together, move up and down)

### Running the Application

1. To run the application, use the following command:
   ```bash
   python3 mouse_controller.py
   ```
   The system will start in headless mode by default, with no display output.

2. To enable a real-time rendering of the hand tracking, use the `-r` flag:
   ```bash
   python3 mouse_controller.py -r
   ```
   This will open a new window showing the live hand tracking process.

Ensure your hand is in the view of the OAK-D Lite camera, and the system will track your hand movements, interpreting them to control the cursor accordingly.

## Contributions and Acknowledgements

This project was inspired by the [depthai_hand_tracker](https://github.com/geaxgx/depthai_hand_tracker) repository, and I acknowledge the groundbreaking work they have shared with the community and do not claim any rights for their work. Contributions to enhance functionality or performance are warmly welcomed. Feel free to submit issues or pull requests.

## License

[MIT](LICENSE) - See the LICENSE file for more details.

## Disclaimer

This system is a demonstration of AI and computer vision capabilities with the OAK-D Lite camera. It is not intended for critical use cases. Use at your own discretion and risk.


# AI-GestureControl
