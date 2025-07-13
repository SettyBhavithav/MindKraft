Gesture-Based Multi-Game Platform
Welcome to the Gesture Multi-Game Platform—an advanced, touchless gaming environment where you control everything with your hands! This platform combines classic arcade games and IQ-testing mini-games, all playable using intuitive hand gestures detected via your webcam.

🚀 Features
Gesture-Driven Control:

Pinch (thumb + index): Select, confirm, shoot, move, or catch

Open palm (hold 2s): Return to menu or pause

Fist (hold 2s): Exit application

Move index finger: Highlight menu items, move paddle/player/cursor

Multi-Game Platform:
Play Shooter, Pong, Breakout, Maze (with AI and hint), Fruit Catcher, and IQ mini-games—all in one place.

AI & Hint Systems:

Maze game includes an AI bot opponent and a hint feature showing the shortest path.

IQ games adapt to your performance.

Modern UI/UX:

Gradient backgrounds, headers, gesture hints, progress circles, and always-on camera feed.

On-screen instructions and visual feedback for detected gestures.

Levels & Progression:

Each game features 25 levels with increasing difficulty.

Unlock new challenges and track your progress.

IQ Tester:

Play a series of fun brain games (memory, logic, pattern, reaction).

Get an IQ-like score based on your performance.

🕹️ Games Included
Game	Description	Levels	AI/Hint	IQ Tested
Shooter	Move and shoot enemies using gestures	25	Planned	Reaction
Pong	Paddle follows your finger, play against AI	25	Yes	Coordination
Breakout	Classic block breaking, paddle by finger	25	-	Spatial
Maze	Navigate maze, pinch to move, AI bot, hint system	25	Yes	Reasoning
Fruit Catcher	Move basket, pinch to catch falling fruits	25	-	Reaction
IQ Tester	Memory, logic, pattern, and reaction mini-games	25+	Yes	All
✋ Gesture Mapping Table
Gesture	Menu/Instructions	In-Game (All Games)
Pinch (thumb+index)	Select/confirm	Fire/Shoot/Move/Catch/Continue
Open palm (hold 2s)	Return to previous/menu	Return to menu (pause/exit game)
Fist (hold 2s)	Exit application	Exit application
Move index finger	Highlight/select options	Move paddle/player/basket/cursor
🧠 IQ Feature
Performance-Based IQ Score:
Your IQ score is calculated based on time, accuracy, moves, and hints used across all games.

Adaptive Challenge:
As your IQ rises, the games become more challenging.

Leaderboard:
Track your best IQ scores and compare with others.

📦 Installation
Install dependencies:

text
pip install opencv-python mediapipe numpy pygame
Run the application:

text
python your_script.py
📝 How to Play
Menu:
Move your index finger to highlight a game, pinch to select, fist (hold 2s) to exit.

Instructions:
Each game shows a gesture-navigable instruction page before starting.

Gameplay:

Pinch to perform the main action (move, shoot, catch, etc.).

Move your finger to control paddles or highlight cells.

Open palm (hold 2s) to return to menu or pause.

Fist (hold 2s) to exit at any time.

Maze Game:

Pinch to move to an adjacent cell.

Open palm (hold 2s) to toggle the hint (shortest path).

Fist (hold 2s) to toggle the AI bot opponent.

🛠️ Extending the Platform
Add More Games:
Follow the modular structure—each game is a function/class with its own gesture mapping and level logic.

Add More IQ Tests:
Implement new mini-games and add them to the IQ Tester module.

Customize Gestures:
Adjust gesture thresholds or add new gestures in the utility functions.

🎨 Credits
Built with Python, Pygame, OpenCV, and MediaPipe.

Inspired by classic arcade games and modern gesture interfaces.

💡 Contributing
Fork the repo, add your new game or feature, and submit a pull request!

Ideas for new IQ games, better AI, or UI themes are welcome.

📄 License
This project is open-source and free to use for educational and non-commercial purposes.

Enjoy your futuristic, touchless gaming experience!
