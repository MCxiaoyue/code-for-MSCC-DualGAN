# code and dataset-for-paper

# 文件夹

* new_brain2voiceDataset_official2包含我们本文提出模型MSCC-DualGAN的相关实验操作代码和结果对比代码

* dual-dualgan-main_offical是本团队之前的研究提出的[Dual-DualGAN](https://github.com/qwe1218088/dual-dualgan)的相关实验操作代码和结果对比代码

# 数据集

* new_brain2voiceDataset_official2文件夹包含了paper中已经切好的公开数据集和本文自采数据集

* dual-dualgan-main_offical的数据集和训练好的相应权重以及实验结果过于大, 保存到了[百度网盘](https://pan.baidu.com/s/17xBNx6JHPrpHGZsjl6EAng?pwd=c8si 
), 下载后直接解压到dual-dualgan-main_offical文件夹里即可, 即dual-dualgan-main_offical/dataset

# 指标对比

* cal_ssim.py为对比结构相似性, cal_pcc.py为对比皮尔逊相关系数, 6_img2wav.py为将梅尔倒谱图转回为音频文件, cal_mcd.py为对比两个音频的MCD值

# 运行说明

* 对于new_brain2voiceDataset_official2：<br>
(1)  配置文件，即util包下的parseArgs.py，--data_path指定数据集的名称，--save_path指定保存文件路径，基本上三个数据集的名称data1_time2that_orignEEG_5to14，sketch-photo和swpd1互相替换就行<br>
(2)  训练文件，即train.py，参数不变就行，直接Run运行就好<br>
(3)  预测文件，即predict.py，其中save_image方法保存的路径为生成网络单向生成的运行结果<br>
(4)  对比指标文件，即各自数据集下的cal_ssim.py，cal_pcc.py和cal_mcd.py。前两个直接对比的是原始测试数据和预测结果的图片，后一个指标需要先运行img2wav.py，将预测结果的图片转为语音，之后再运行cal_mcd.py，得到对比指标的结果<br>
(5)  net文件夹里有很多不同尝试的生成器，想要换的时候把train.py里的关键词“Generator1”替换成各自的文件名称即可<br>

* 对于dual-dualgan-main_offical：<br>
