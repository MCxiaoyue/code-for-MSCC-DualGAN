import argparse
def parseArgs():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--save_path", type=str, default="./sketch-photo/save1/")
    parser.add_argument("--data_path", type=str, default="./sketch-photo/")
    parser.add_argument("--trainC", type=str, default="train/C")
    parser.add_argument("--trainB", type=str, default="train/B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--g_b_path", type=str, default="generator_b.pkl")
    parser.add_argument("--g_c_path", type=str, default="generator_c.pkl")
    parser.add_argument("--d_b_path", type=str, default="discriminator_b.pkl")
    parser.add_argument("--d_c_path", type=str, default="discriminator_c.pkl")
    args = parser.parse_args()
    return args
