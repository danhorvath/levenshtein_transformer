#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J transformer
### -- ask for number of cores (default: 1) --
#BSUB -n 2
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=4"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=25GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o transformer_%J.out
#BSUB -e transformer_%J.err
# -- end of LSF options --

sxm2sh
# module load cuda/10.1
# module load python3

# pip3 install -r requirements.txt --user
# python3 -m spacy download en --user
# python3 -m spacy download de --user

python3 -m wandb.cli login 789d100f03fa3d5dbfbe897a22e4b9f08ed01598
python3 -m wandb.cli on

python3 ~/repos/transformer/en_de_translate.py


