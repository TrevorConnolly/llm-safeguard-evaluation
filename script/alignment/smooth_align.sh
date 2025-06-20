#!/bin/bash
#SBATCH -J smooth_align                 # Job name

#SBATCH -t 60                                 # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=40G
#SBATCH -o smooth_align-%j.out                         # Combined output and error messages file



#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --constraint=a100        # only a100 gpu
#SBATCH --nodelist=adroit-h11g1

#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=vi6908@princeton.edu


module load anaconda3/2023.3


conda activate boostertemp


export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


lamb=${1:-5} 
alpha=${2:-0.1}   
bad_sample_num=${3:-5000} 
sample_num=5000
model_path=${4:-meta-llama/Llama-3.2-1B}   
path_after_slash=$(basename "$model_path") 
echo "The value of lamb is: $lamb"
echo "The value of alpha is: $alpha"
echo "The value of bad_sample_num is: $bad_sample_num"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory





CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path} \
	--data_path PKU-Alignment/BeaverTails_safe \
	--bf16 True \
	--output_dir ckpt/${path_after_slash}_smooth_${lamb}_${alpha}_${bad_sample_num}_${sample_num} \
	--num_train_epochs 20 \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  5e-4 \
	--weight_decay 0.1 \
	--warmup_ratio 0 \
	--lr_scheduler_type "constant" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer booster \
	--sample_num $sample_num \
	--bad_sample_num $bad_sample_num \
	--lamb ${lamb} \
	--alpha ${alpha} \
	--eval_steps 5000
	
	

cd poison/evaluation  

CUDA_VISIBLE_DEVICES=0 python pred.py \
	--lora_folder ../../ckpt/${path_after_slash}_smooth_${lamb}_${alpha}_${bad_sample_num}_${sample_num}\
	--model_folder ${model_path} \
	--output_path ../../data/poison/${path_after_slash}_smooth_${lamb}_${alpha}_${bad_sample_num}_${sample_num}

CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/${path_after_slash}_smooth_${lamb}_${alpha}_${bad_sample_num}_${sample_num}

cd sst2

CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
	--lora_folder ../ckpt/${path_after_slash}_smooth_${lamb}_${alpha}_${bad_sample_num}_${sample_num} \
 	--model_folder ${model_path} \
 	--output_path ../data/sst2/${path_after_slash}_smooth_${lamb}_${alpha}_${bad_sample_num}_${sample_num}_baser
