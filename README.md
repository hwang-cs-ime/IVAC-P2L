# IVAC-P<sup>2</sup>L

> **IVAC-P<sup>2</sup>L: Leveraging Irregular Repetition Priors for Improving Video Action Counting** <br>
> Hang Wang<sup>1,2</sup> | 
[Zhi-Qi Cheng](https://github.com/zhiqicheng)<sup>3</sup> |
Youtian Du<sup>1</sup> |
[Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/)<sup>2</sup> <br>
<sup>1</sup>Xi'an Jiaotong University, <sup>2</sup>The Hong Kong Polytechnic University, <sup>3</sup>Carnegie Mellon University <br>


## Preparing Datasets
We train on the training set of the RepCount-A dataset, and test on the testing set of the RepCount-A dataset, the validation sets of the UCFRep and Countix datasets.

Download datasets: [RepCount-A](https://svip-lab.github.io/dataset/RepCount_dataset.html), [UCFRep](https://www.crcv.ucf.edu/data/UCF101.php), Countix

## Train

on the training set of the RepCount-A dataset
```bash
python train.py
```

## Test

on the testing set of the RepCount-A dataset
```bash
python test.py
```

## Pre-trained Checkpoint on RepCount-A

We also provide a pre-trained model for RepCount-A, which can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1gFUhs-Kjacpy6wMxvIi0B4VnVlAlxnhP/view?usp=sharing).

### Acknowledgement
This project is built based on the excellent [TransRAC](https://github.com/SvipRepetitionCounting/TransRAC) project.


### Contact
If you have any questions, please feel free to contact: cshwang@comp.polyu.edu.hk


<details>
<summary>statistics</summary>

<a href="https://info.flagcounter.com/aecG"><img src="https://s01.flagcounter.com/mini/aecG/bg_FFFFFF/txt_000000/border_CCCCCC/flags_0/" alt="Flag Counter" border="0"></a>

</details>

