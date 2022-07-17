bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=11000]" ./run_resunet_gmap.sh
bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=11000]" ./run_resnet_gmap.sh
bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=11000]" ./run_resnet_ethz.sh
bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=11000]" ./run_resunet_ethz.sh
bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=12000]" ./run_resnet_finetune.sh
bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=12000]" ./run_resunet_finetune.sh
bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=12000]" ./run_resnet_scratch_gmap.sh
bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=12000]" ./run_resnet_scratch_finetune.sh