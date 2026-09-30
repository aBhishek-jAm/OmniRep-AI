"""
FitVision AI - Main CLI Entry Point

Provides a unified command-line interface for all features:
- collect: Collect training data
- train: Train the ML model
- predict: Real-time exercise recognition
- track: Rule-based exercise tracking (GymLytics-style)
- webapp: Launch the Streamlit web app

Usage:
    python main.py collect --exercise squat --samples 300
    python main.py train
    python main.py predict --confidence 0.7
    python main.py track --type pushup --source 0
    python main.py webapp
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="FitVision AI - Exercise Recognition & Rep Counting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py collect -e squat -s 300    Collect 300 squat samples
  python main.py train                      Train the model
  python main.py predict                    Live exercise prediction
  python main.py track -t pushup -src 0     Track pushups (camera)
  python main.py webapp                     Launch Streamlit web app
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- Collect command ---
    collect_parser = subparsers.add_parser('collect', help='Collect training data')
    collect_parser.add_argument('-e', '--exercise', type=str, default=None,
                                help='Exercise type')
    collect_parser.add_argument('-s', '--samples', type=int, default=300,
                                help='Samples per session (default: 300)')
    collect_parser.add_argument('--basic', action='store_true',
                                help='Use basic features only (132)')

    # --- Train command ---
    train_parser = subparsers.add_parser('train', help='Train the ML model')
    train_parser.add_argument('--data-dir', type=str, default=None,
                              help='Training data directory')
    train_parser.add_argument('--output-dir', type=str, default=None,
                              help='Model output directory')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                              help='Test split ratio (default: 0.2)')

    # --- Predict command ---
    predict_parser = subparsers.add_parser('predict', help='Real-time exercise recognition')
    predict_parser.add_argument('--model-dir', type=str, default=None,
                                help='Trained model directory')
    predict_parser.add_argument('--confidence', type=float, default=0.7,
                                help='Confidence threshold (default: 0.7)')
    predict_parser.add_argument('--camera', type=int, default=0,
                                help='Camera index (default: 0)')

    # --- Track command ---
    track_parser = subparsers.add_parser('track', help='Rule-based exercise tracking')
    track_parser.add_argument('-t', '--type', type=str, required=True,
                              choices=['pushup', 'squat', 'plank', 'shouldertap', 'lunges'],
                              help='Exercise type to track')
    track_parser.add_argument('-src', '--source', type=str, default='0',
                              help='Video source: camera index or file path (default: 0)')

    # --- Web App command ---
    subparsers.add_parser('webapp', help='Launch Streamlit web app')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n  💡 Tip: Use 'python main.py <command> --help' for command-specific help\n")
        sys.exit(0)

    if args.command == 'collect':
        from src.data_collection import collect_data, EXERCISES
        exercise = args.exercise
        if exercise is None:
            print(f"\nAvailable exercises: {EXERCISES}")
            exercise = input("Enter exercise label: ").lower().strip()
            while exercise not in EXERCISES:
                print(f"Invalid! Choose from: {EXERCISES}")
                exercise = input("Enter exercise label: ").lower().strip()
        collect_data(exercise, args.samples, enhanced=not args.basic)

    elif args.command == 'train':
        from src.training import main as train_main
        # Override sys.argv for argparse in training module
        train_args = []
        if args.data_dir:
            train_args.extend(['--data-dir', args.data_dir])
        if args.output_dir:
            train_args.extend(['--output-dir', args.output_dir])
        train_args.extend(['--test-size', str(args.test_size)])
        sys.argv = ['train'] + train_args
        train_main()

    elif args.command == 'predict':
        from src.prediction import main as predict_main
        predict_args = []
        if args.model_dir:
            predict_args.extend(['--model-dir', args.model_dir])
        predict_args.extend(['--confidence', str(args.confidence)])
        predict_args.extend(['--camera', str(args.camera)])
        sys.argv = ['predict'] + predict_args
        predict_main()

    elif args.command == 'track':
        from src.exercises.pushup import Pushup
        from src.exercises.squat import Squat
        from src.exercises.plank import Plank
        from src.exercises.shoulder_tap import ShoulderTap
        from src.exercises.lunges import Lunges

        exercise_map = {
            'pushup': Pushup,
            'squat': Squat,
            'plank': Plank,
            'shouldertap': ShoulderTap,
            'lunges': Lunges,
        }

        # Convert source to int if it's a camera index
        source = args.source
        try:
            source = int(source)
        except ValueError:
            pass

        tracker = exercise_map[args.type]()
        print(f"\n  Starting {args.type} tracker... Press ESC to quit.\n")
        tracker.exercise(source)

    elif args.command == 'webapp':
        import subprocess
        print("\n  Launching Streamlit web app...\n")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])


if __name__ == '__main__':
    main()
