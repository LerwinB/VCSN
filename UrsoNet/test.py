import os
import sys
print("hello")
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="'train' or 'evaluate'")
    parser.add_argument('--weights', required=True, help="Path to weights .h5 file or 'coco' or 'imagenet' for coco pre-trained weights")
    
    args = parser.parse_args()
    print("Command: ", args.command)
    if args.command == "train":
        print("train")
    if args.weights.lower() == "coco":
        print("coco")
    elif args.weights.lower() == "last":
        # Find last trained weights
        print("last")
    elif args.weights.lower() in ['soyuz_hard', 'dragon_hard', 'speed']:
        print(args.weights.lower()) 
    elif args.weights.lower() != "none":
        print(args.weights)
    print(args.weights.lower()) 

