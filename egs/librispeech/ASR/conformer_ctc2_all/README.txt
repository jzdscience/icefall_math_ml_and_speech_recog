
Author: Ju Zhang, jz3702

Date: 12/17/2023
===================================================================================================================

Project Title: Assessing The Impact Of Different Activation Functions On Training Time And Performance In Conformer-Ctc End-To-End Automatic Speech Recognition Model

Project summary:

The rapid development of neural network models, coupled with increased computing power, has enabled advancements in end-to-end automatic speech recognition (ASR) tasks. 
A notable recent development in ASR systems is the conformer model. Drawing inspiration from the transformer model, it consists of multiple conformer blocks. 
Each block is composed of two feed-forward modules, a multi-head self-attention module, a convolution module, and a Layernorm module. 
In the feed-forward and convolution modules of the conformer block, the Swish function is used by default as the activation function.
The Swish function, known for its unique properties such as non-monotonicity, has higher computational complexity compared to traditional activation functions like ReLU. 
This raises an interesting question about how different activation functions impact the training time and accuracy of conformer-based end-to-end ASR systems. 
In this study, the author investigated this issue and found that certain activation functions, such as ReLU, Simplified Double Swish, SoftPlus, and GELU, are associated with improved trainability of the model.
Additionally, it was discovered that mathematically simpler activation functions, such as ReLU, lead to reduced training time.
This study is significant as it represents the first thorough comparison of the impact of activation functions on conformer-based ASR systems. 
It provides insights that can guide the selection of optimal activation functions in conformer-based ASR systems when balancing between computational complexity and model accuracy.

===================================================================================================================
List of tools:

Python, Anaconda, pytoch, CUCA, CuDNN, torchaudio, k2, icefall, lhotse

See ENVIRONMENT SET UP section for more details, including version information

===================================================================================================================
The study is based on one librispeech recipe of the Icefall module of Kaldi project. 
https://github.com/k2-fsa/icefall

The `conformer_ctc2_all` is derived from `icefall\egs\librispeech\ASR\conformer_ctc2` of the original repo
and it is to be placed in `icefall\egs\librispeech\ASR\` folder.

==================================================================================================================
The file structure is illustrated as below,  asteriks *** shows where code addition/revision occurs
icefall/
-egs/
--librispeech/
---ASR/
----prepare.sh # call this to prepare data
----download/
----data/
----conformer_ctc2_all/
-----README.txt                     # *** new script *** this file
-----asr_datamodule.py              # module to load data
-----attention.py                   # define attention architecture
-----conformer.py                   # *** revised***  confomer model architecture, adding the capability of using different activation functions
-----decode.py                      # script to call for decoding
-----exp/                           # experiment results are saved here, the default name is exp/, it is empty without training model
-----experiment_evaluation.ipynb    # *** new script ***, a Jupyter NB to analyze the experiment result and aggregate results for table/plot
-----export.py                      # functionality to export model
-----label_smoothing.py             # functionality to do label_smoothing
-----model_structure.md             # *** new script *** just a note on model structure, optional
-----optim.py                       # optimzer definition
-----run_all_decoders.sh            # *** new script *** shell script to run decoding process for all models, i.e. call decode.py in a loop
-----run_all_trainers.sh            # *** new script *** shell script to run training process for all models, i.e. call train.py in a loop
-----scaling.py                     # *** revised***  Here alternative activation functions are defined
-----subsampling.py                 # *** revised***  2D convolutional subsampling model architecture, adding the capability of using different AFs
-----train.py                       # script to call for model training
-----transformer.py                 # *** revised***  transformer model architecture, adding the capability of using different activation functions



=====================================================================================================

To run the code, you need to step up the Kaldi envrionment first.
WARNING: It is extremely fragile. May take a few hours even days to set up and run expriment without any issue. 
I am very happy to help to trouble shooting!

I. ENVIRONMENT SET UP

To have the Same envrionment I have:

1. Spin a Google Cloud VM 
1.1 Choose N1- 8vCPU 52GB RAM, 1200GB HDD with 4 T4 GPU
1.2 Choose Ubuntu 20.04 LTS as OS, not pre-built deep-learning environment 

2. Launch

3. SSH to the VM and let us start install things:
3.1. Install NVIDIA driver https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installdriver
After this, the `nvidia-smi` command should work...

3.2. Install CUDA toolkit 12.1 and CuDNN following https://k2-fsa.github.io/k2/installation/cuda-cudnn.html
After this, the `nvcc --version` command should work
WARNING: Do not use pip/conda install cudatoolkit

3.3. Install Anaconda to manage the virtual environment
3.3.1 Create an virtual enviroment with Python=3.11
3.3.2 Activate the virtual envrionment

3.4. Install ffmpeg. conda install -c conda-forge 'ffmpeg<7'

3.5 Instal the rest of packages the *Installation Example* on https://icefall.readthedocs.io/en/latest/installation/index.html#installation-example
which includes: 
3.5.1 Torch (2.1.0 for CUDA 12.1) and torchaudio (2.1.0 for CUDA 12.1) at the same time from https://download.pytorch.org/whl/torch_stable.html
    pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
3.5.2 K2 (1.24.4 for CUDA 12.1 and Torch 2.1.0) from https://k2-fsa.github.io/k2/cuda.html
    pip install k2==1.24.4.dev20231123+cuda12.1.torch2.1.0 -f https://k2-fsa.github.io/k2/cuda.html
3.5.3 Lhotse (newest version)
pip install git+https://github.com/lhotse-speech/lhotse
3.5.4 download Icefall and install the requirements in the repo
    git clone git@github.com:jzdscience/icefall_w4995_mathml.git   # this is my fork from https://github.com/k2-fsa/icefall
    pip install -r ./icefall/requirements.txt

3.6 Fix a compatibility issue in the Pytorch you just installed by
3.6.1 Open "/torch/nn/modules/transformer.py" in package installation folder with a text editor
3.6.2 Found the forward function with
tgt_is_causal: Optional[bool] = None,
memory_is_causal: bool = False
3.6.3 Comment out these two arguments. Having these two cause the error of unknown arguments.
It should be due to some version imcompatibility...

3.7 Download and unzip my .zip package uploaded as code submission

3.8 Put the whole folder `conformer_ctc2_all` in `icefall\egs\librispeech\ASR\` folder.

II. RUN THE EXPERIMENT
A. Prepare data
1. cd to `icefall\egs\librispeech\ASR\
2. call `./prepare.sh` to prepare data
basically, it will first download voice data, lexicon, lm, etcs into `icefall\egs\librispeech\ASR\download\` ... then prepare and save data to
`icefall\egs\librispeech\ASR\data\`

WARNING: You do not need to do anything else unless it throw error, which is very likely!

B. Training
1. cd to icefall\egs\librispeech\ASR\conformer_ctc2_all

2. call './run_all_trainers.sh', it should start to train for all models. 
WARNING: 
2.1 It take ~32min*30 epoch* 9 models = 144hrs to finish on a VM with 4 Tesla T4 GPU
2.2 You can reduce the max-duration is your vRAM is lower than 16GB, otherwise it is not enough.
2.3 Alternatively, you can call individual training 
python train.py \
        --exp-dir exp/leaky_relu \  # this is where to save the result
        --start-epoch 1 \ # if you have a checkpoint saved after epoch,say 30, then you can start from 31 to train
        --world-size 4 \  # this is how many GPU you want to use
        --full-libri 0 \  # whether to use full librispeech dataset
        --max-duration 250 \ # size of voice duration used (deciding batch size)
        --activation_type leaky_relu \  # which activation function you want to try
        --all_activation no # this is the switcher for whether replace all activations besides the one in conformer block
2.4 (optional) You can skip training by downloading my model as in section D -1.1

C. Analysis 1:
Open experiment_evaluation.ipynb and run through section 1.1 to obtain the epoch number with best validation loss for each model
(e.g. so you know for leaky ReLU, the best validation loss occurs at the epoch 23)

D. Decoding 
1. (Optional) If you do want to save time for training model and use my checkpoints
1.1 download all my checkpoints at https://drive.google.com/drive/folders/1gaGhFexsxj7gNvZ7f8sghRJZ5yvpu4Ng?usp=sharing
1.2 put the content of downloaded `exp/` folder into your `exp/` folder

2. Call `run_all_decoders.sh` to decode test dataset with all models using checkpoints saved after 30 epoch of training

3. You need to call individual experiment to decode test dataset with the model checkpoints which contains the "best validation loss"
Example:
`python decode.py \
--epoch 23 \   # this is the epoch containing the best validation loss
--avg 1 \
--max-duration 150  \
--exp-dir exp/leaky_relu  \
--lang-dir \
../data/lang_bpe_500  \
--method ctc-decoding`

Apologize that you need to run individual experiement and change epoch argument manually to the best validation loss epoch obtained from C. Analysis 1, 
`run_all_decoders.sh` does not support a variable epoch argument yet...

The results of WER will be saved as `wer.summary.clean.txt` `wer.summary.other.txt` in individual run folder, for example,  `\exp\leaky_relu\`

WARNING: `wer.summary.clean.txt` `wer.summary.other.txt` will be overwritten, if you run the decoding process again for the same model. It does not have a renaming mechanism yet.

E. Analysis 2:
Back to experiment_evaluation.ipynb and run through section 1.3 to obtain the WER information in batch

You will have something like 
{'run_20231210_gelu_clean': {'clean_wer': 9.93, 'other_wer': 25.42},
 'run_20231206_relu_clean': {'clean_wer': 12.72, 'other_wer': 33.96},
 'run_20231211_selu_clean': {'clean_wer': 97.73, 'other_wer': 97.74},
 'run_20231208_singleswish_clean': {'clean_wer': 98.7, 'other_wer': 98.76},
 'run_20231209_tanh_clean': {'clean_wer': 98.92, 'other_wer': 98.8},
 'run_20231208_leakyRelu_clean': {'clean_wer': 21.62, 'other_wer': 47.37},
 'run_20231210_elu_clean': {'clean_wer': 99.99, 'other_wer': 99.99},
 'run_20231213_softplus_all_activation_clean': {'clean_wer': 99.99,
  'other_wer': 99.99},
 'run_20231211_softplus_clean': {'clean_wer': 8.82, 'other_wer': 23.68},
 'run_20231206_doubleswish_clean': {'clean_wer': 8.28, 'other_wer': 22.21}}

F. Analysis 3:
Back to experiment_evaluation.ipynb and run through section 2 to obtain the computation time for training each model

You will have something like 
{'gelu_clean': '0:32:16',
 'relu_clean': '0:31:19',
 'selu_clean': '0:32:11',
 'singleswish_clean': '0:33:00',
 'tanh_clean': '0:32:02',
 'leakyRelu_clean': '0:32:13',
 'elu_clean': '0:32:03',
 'softplus_all_activation_clean': '0:32:37',
 'softplus_clean': '0:32:27',
 'doubleswish_clean': '0:33:35'}

Hopefully by now you would have the results shown in my paper.

===========================================================================================================================
Thank you so much! Please sent me an email jz3702@columbia.edu if you encounter any problem!