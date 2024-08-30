import torch
from util import parseArgs
from util.pre_loader import pre_loader
from torchvision.utils import save_image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_BtoC_arr = []
real_C_arr = []

def predict():
    parser = parseArgs.parseArgs()
    data_path = parser.data_path
    batch_size = parser.batch_size
    image_size = parser.image_size
    G_B = torch.load('./256voicedataset_time2that1_abc/checkpoint1/generator_b.pkl')

    testB_loader = pre_loader(data_path, "./test_AtoB_results", image_size=image_size, batch_size=batch_size)
    realC_loader = pre_loader(data_path, "./test/C", image_size=image_size, batch_size=batch_size)

    for X, labelx in iter(testB_loader):
        print(labelx)
        x = X.to(device)
        g_b = G_B(x)
        # torch.nn.functional.interpolate(g_a, (labelx[0], labelx[1]),mode='bilinear', align_corners=True)
        test_BtoC_arr.append(g_b.detach().cpu().numpy())
        save_image(g_b, "./256voicedataset_time2that1_abc/test_BtoC_results/BtoC_" + labelx[2][0].split('.')[0].split('_')[1] + '.PNG')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')




## 运行 ##
if __name__ == '__main__':
    predict()