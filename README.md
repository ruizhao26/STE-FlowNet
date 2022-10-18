## [AAAI 2022] Spatio-Temporal Recurrent Networks for Event-Based Optical Flow Estimation

<h4 align="center"> Ziluo Ding, Rui Zhao, Jiyuan Zhang, Tianxiao Gao, Ruiqin Xiong, Zhaofei Yu, Tiejun Huang </h4>
<h4 align="center"> Peking University </h4><br> 

This repository contains the official source code for our paper:

Spatio-Temporal Recurrent Networks for Event-Based Optical Flow Estimation.  
AAAI 2022 

Paper:  
[AAAI version](https://ojs.aaai.org/index.php/AAAI/article/view/19931)  
[Arxiv version](https://arxiv.org/pdf/2109.04871.pdf)

* [STEFlow](#Spatio--Temporal-Recurrent-Networks-for-Event--Based-Optical-Flow-Estimation.)
  * [Environments](#Environments)
  * [Prepare the Data](#Prepare-the-Data)
    * [Encode the events of MVSEC](#Encode-the-events-of-MVSEC)
    * [Prepare the ground truth of MVSEC for Testing](#Prepare-the-ground-truth-of-MVSEC-for-Testing)
    * [Download the pretrained models](#Download-the-pretrained-models)
  * [Evaluate](#Evaluate)
  * [Train](#Train)
  * [Citation](#Citations)
  * [Acknowledgement](#Acknowledgement)

## Environments

You will have to choose cudatoolkit version to match your compute environment. The code is tested on PyTorch 1.10.2+cu113 and spatial-correlation-sampler 0.3.0 but other versions might also work. 

```bash
conda create -n steflow python==3.9
conda activate steflow
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip3 install spatial-correlation-sampler matplotlib opencv-python h5py tensorboardX
```

We don't ensure that all the PyTorch versions can work well. For example, some PyTorch versions fail in encoding the flow ground truth since the data precision of the time stamps will be lost during conversion from PyTorch Tensor to Numpy array.

## Prepare the Data

### Encode the events of MVSEC

```bash
# Encoding the Events of MVSEC dataset
# You can set your data path in flow_gt_encoding.py or through argparser (--data)
cd ./encoding
# For training data
python3 split_encoding_torch.py -sp=5 -se='outdoor_day2' -s
# For testing data, you can choose the scenes you want to test
# The scenes can be choosen below
scenes=(indoor_flying1 indoor_flying2 indoor_flying3 outdoor_day1)
for ss in ${scenes[*]};
do
	python3 split_encoding_torch.py -sp=5 -se=$ss -s
done
```

###  Prepare the ground truth of MVSEC for testing.

The test code is derived from [Spike-FlowNet](https://github.com/chan8972/Spike-FlowNet). In the original codes, the ```estimate_corresponding_gt_flow``` cost a lot of time. We implement this function before testing and save the estimated gt flow.

If you want to plot the visualization of flow results, please refer to the code repository of  [Spike-FlowNet](https://github.com/chan8972/Spike-FlowNet).

```bash
# Prepare ground truth of flow
# You can set your data path in flow_gt_encoding.py or through argparser (--data)
# You can choose the type of dt below to generate flow gt
dts=(1 4)
# You can choose the type of scenes below to generate flow gt
scenes=(indoor_flying1 indoor_flying2 indoor_flying3 outdoor_day1)

for dt in ${dts[*]};
do
	for ss in ${scenes[*]};
	do
		python3 flow_gt_encoding.py -dt=$dt -ts=$ss
	done
done
```

### Download the pretrained models

The pretrained models for ```dt=1``` and ```dt=4``` can be download in the Google Drive link below

[Link for pretrained models](https://drive.google.com/drive/folders/1EGwlpNZEqNYs23ZBSYUIHKwHJJM3kudu?usp=sharing)

You can download the pretrained models to ```./ckpt```

## Evaluate

```bash
### You can set the data path in the .py files or through argparser (--data)
scenes=(indoor_flying1 indoor_flying2 indoor_flying3 outdoor_day1)
# for dt=1 case
python3 main_steflow_dt1.py -e \
--test-set 'Choose from the scenes' \
--pretrained './ckpt/steflow_dt1.pth.tar'

# for dt4 case
python3 main_steflow_dt4.py -e \
--test-set 'Choose from the scenes' \
--pretrained './ckpt/steflow_dt4.pth.tar'
```



Note that we have provided a better pretrained model for evaluation, which is slightly different with the original results shown in paper. In more detail, the model achieves better results in all three indoor scenes and demonstrates better generalization. However, the model degrates only in outdoor scene. The following table shows the detailed results.

<img src="https://github.com/ruizhao26/STE-FlowNet/blob/main/fig/Results.png" width="90%">



## Train

### Some Useful Command-line arguments

About training paths:

```--data```: your data path

```--savedir```: folder for saving training results

About hyperparameter for training:

```--batch-size```: batch size for training

```--lr```: initial learning rate

```--milestones```: milestones for learning rate decay

```--gamma```: factor for learning rate decay

About other training settings:

``` --workers```: number of workers to use

```--test-set```: the validation dataset during training

```--evaluate-interval```: how many epochs to evaluate after

```--print-freq```: how many iterations to print training details after

All the command line arguments for hyperparameter tuning can be found in the `main_steflow_dt*.py` files.

```bash
### You can set the data path in the .py files or through argparser (--data)
# for dt=1 case
python3 main_steflow_dt1.py --lr 4e-4 -b 8
# for dt=4 case
python3 main_steflow_dt4.py --lr 2e-4 -b 8
```

## Citations

If you find this code useful in your research, please consider citing our paper.
AAAI version: 

```
@inproceedings{ding2021spatio,
  title={Spatio-temporal recurrent networks for event-based optical flow estimation},
  author={Ding, Ziluo and Zhao, Rui and Zhang, Jiyuan and Gao, Tianxiao and Xiong, Ruiqin and Yu, Zhaofei and Huang, Tiejun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  volume={36},
  number={01},
  pages={525--533},
  year={2022}
}
```

Arxiv version:

```
@article{ding2021spatio,
  title={Spatio-temporal recurrent networks for event-based optical flow estimation},
  author={Ding, Ziluo and Zhao, Rui and Zhang, Jiyuan and Gao, Tianxiao and Xiong, Ruiqin and Yu, Zhaofei and Huang, Tiejun},
  journal={arXiv preprint arXiv:2109.04871},
  year={2022}
}
```

## Acknowledgement

Parts of this code were derived from [chan8972/Spike-FlowNet](https://github.com/chan8972/Spike-FlowNet). Please also consider to cite [Spike-FlowNet](https://link.springer.com/chapter/10.1007/978-3-030-58526-6_22) if you'd like to cite our paper.

We thank [Chankyu Lee](https://github.com/chan8972) for replying our email about the evaluation details about ```outdoor_day1``` scene.
