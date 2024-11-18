import torch
from util import parseArgs
from util.pre_loader import pre_loader
from torchvision.utils import save_image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_AtoB_arr = []
real_B_arr = []

def predict():
    parser = parseArgs.parseArgs()
    data_path = parser.data_path
    batch_size = parser.batch_size
    image_size = parser.image_size
    G_A = torch.load('F07/checkpoint/generator_a.pkl')

    testA_loader = pre_loader(data_path, "./test/A", image_size=image_size, batch_size=batch_size)
    # realB_loader = pre_loader(data_path, "./test/B", image_size=image_size, batch_size=batch_size)

    for X, labelx in iter(testA_loader):
        print(labelx)
        x = X.to(device)
        g_a = G_A(x)
        # torch.nn.functional.interpolate(g_a, (labelx[0], labelx[1]),mode='bilinear', align_corners=True)
        test_AtoB_arr.append(g_a.detach().cpu().numpy())
        save_image(g_a, "./F07/test_AtoB_results/AtoB_" + labelx[2][0].split('.')[0].split('_')[1] + '.PNG')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')



## 运行 ##
if __name__ == '__main__':
    predict()