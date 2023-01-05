import argparse


def main():
    parser = argparse.ArgumentParser(description="argparse使用-小试牛刀")
    parser.add_argument('-n', '--name', default="juzipi")
    parser.add_argument("-j", '--job', default="程序员")
    args = parser.parse_args()
    print(args)
    name = args.name
    job = args.job
    print("Hello， 大家好，我是{}{}!".format(job, name))


if __name__ == '__main__':
    main()