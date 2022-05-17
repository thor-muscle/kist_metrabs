import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

# 변수공간 만들음.
parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')

# 옵션 만들음.
parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))