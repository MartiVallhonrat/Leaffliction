import argparse

def parse_args():
	parser = argparse.ArgumentParser(
		prog="Transformation",
		description="Program transforms and image"
		)

	parser.add_argument("-src", "--source", type=str, required=True, help="Path to image or image directory.")
	parser.add_argument("-dst", "--destination", type=str, required=True, help="Path to save the image transformation.")
	parser.add_argument('-m', '--mode', type=str, choices=['\"Gaussian blur\"', 'Mask', '\"Roi objects\"', '\"Analize object\"', 'Pseudolandmarks'], default='Gaussian blur', help='Improvement mode, if has spaces please use "Roi objects"')
	return parser.parse_args()

def main(args):
    print(f"Path de entrada: {args.source}")
    print(f"Path de save: {args.destination}")
    print(f"Path de save: {args.mode}")

if __name__ == "__main__":
    args = parse_args()
    main(args)