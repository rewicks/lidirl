import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bytes', action='store_true', default=False)
    parser.add_argument('--label', type=str)
    parser.add_argument('--infiles', nargs='*')

    args = parser.parse_args()

    for file_path in args.infiles:
        with open(file_path) as infile:
            for line in infile:
                if args.bytes:
                    line = line.strip().encode('utf-8')
                else:
                    line = [_ for _ in line.strip()]
                out = []
                for o in line:
                    if o == ' ':
                        out.append('[SPACE]')
                    else:
                        out.append(str(o))
                print(f'{args.label}\t{" ".join(out)}')
