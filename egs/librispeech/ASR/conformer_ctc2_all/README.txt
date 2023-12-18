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
-----asr_datamodule.py              # module to load data
-----attention.py                   # define attention architecture
-----conformer.py                   # *** revised***  confomer model architecture, adding the capability of using different activation functions
-----decode.py                      # script to call for decoding
-----exp/                           # experiment results are saved here, the default name is exp/, it is empty without training model
-----experiment_evaluation.ipynb    # *** new script ***, to analyze the experiment result and generating results for table/plot
-----export.py                      # functionality to export model
-----label_smoothing.py             # functionality to do label_smoothing
-----model_structure.md             # *** new script *** just a note on model structure, optional
-----optim.py                       # optimzer definition
-----README.txt                     # *** new script *** this file
-----run_all_decoders.sh            # *** new script *** shell script to run decoding process for all models, i.e. call decode.py in a loop
-----run_all_trainers.sh            # *** new script *** shell script to run training process for all models, i.e. call train.py in a loop
-----scaling.py                     # *** revised***  Here alternative activation functions are defined
-----subsampling.py                 # *** revised***  2D convolutional subsampling model architecture, adding the capability of using different activation functions
-----train.py                       # script to call for model training
-----transformer.py                 # *** revised***  transformer model architecture, adding the capability of using different activation functions



=====================================================================================================

To run the code, you need to step up the Kaldi envrionment first.
WARNING: It is extremely fragile. May take a few hours even days to set up and run expriment without any issue. 
I am very happy to help to trouble shooting!

I. SET UP

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
    git clone https://github.com/k2-fsa/icefall
    pip install -r ./icefall/requirements.txt

3.6 Fix a compatibility issue in the Pytorch you just installed by
3.6.1 Open "/torch/nn/modules/transformer.py" in package installation folder with a text editor
3.6.2 Found the forward function with
tgt_is_causal: Optional[bool] = None,
memory_is_causal: bool = False
3.6.3 Comment out these two arguments. Having these two cause the error of unknown arguments.
It should be due to some version imcompatibility...

3.7 Download and unzip my .zip package

3.8 Put the whole folder `conformer_ctc2_all` in `icefall\egs\librispeech\ASR\` folder.

II. RUN THE EXPERIMENT
A. Prepare data
1. cd to `icefall\egs\librispeech\ASR\
2. call `./prepare.sh` to prepare data
basically, it will first download voice data, lexicon, lm, etcs into `icefall\egs\librispeech\ASR\download\` ... then prepare and save data to
`icefall\egs\librispeech\ASR\data\`

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

C. Decoding
