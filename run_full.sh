bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=11000]" ./run_resnet_gmap.sh
bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=11000]" ./run_resnet_ethz.sh
bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=11000]" ./run_resunetsym_ethz.sh
bsub -n 4 -W 12:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=11000]" ./run_resunetsym_gmap.sh

