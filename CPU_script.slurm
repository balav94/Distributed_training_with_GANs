#!/usr/bin/bash
#
#SBATCH --time=05:00:00
#SBATCH --nodes=4 --ntasks-per-node=28
#SBATCH --mem=115gb
#SBATCH --account=PAS2056

## Best global batch size is 128

BATCH=8
GCPU="CPU"
EPOCH=2
NODES=4
MODEL="SNGAN"
STRATEGY="HIERARCHICAL"
FUSION_THRESHOLD=64

#SBATCH --job-name="${GCPU}_m${MODEL}_b${BATCH}_e${EPOCH}_n${NODES}_${STRATEGY}_ft${FUSION_THRESHOLD}.job"

source /users/PAS1064/osu10431/miniconda_new/bin/activate
module load mvapich2-gdr/2.3.5
module load cuda/11.2.2
export PYTHONNOUSERSITE=true
conda activate tf_w_hvd
export HOROVOD_FUSION_THRESHOLD=FUSION_THRESHOLD*1024*1024
export HOROVOD_CYCLE_TIME=3.5

## Hierarchical Reduction
export HOROVOD_HIERARCHICAL_ALLREDUCE=1
export HOROVOD_HIERARCHICAL_ALLGATHER=1

# The following are Parameters for Fusion Buffer and clock cycles
export HOROVOD_FUSION_THRESHOLD=536870912 #512MB
export HOROVOD_CYCLE_TIME=20
export HOROVOD_TIMELINE_MARK_CYCLES=1

export MV2_PATH=/opt/mvapich2-gdr/intel/19.0/2.3.5/
export MV2_USE_CUDA=0
export MV2_SUPPORT_DL=1
export MV2_USE_ALIGNED_ALLOC=1
export MV2_CPU_BINDING_POLICY=hybrid
export MV2_HYBRID_BINDING_POLICY=spread
export MV2_SHOW_CPU_BINDING=1
export OMP_NUM_THREADS=28
export HOROVOD_TIMELINE=${GCPU}_m${MODEL}_b${BATCH}_e${EPOCH}_n${NODES}_${STRATEGY}_ft${FUSION_THRESHOLD}.json

cd
cd /users/PAS1064/osu10431/HPDL/GANs_project/keras/Multi_node_training/Hierarchical_AllReduce

OPFILE="${GCPU}_m${MODEL}_b${BATCH}_e${EPOCH}_n${NODES}_${STRATEGY}_ft${FUSION_THRESHOLD}.op"
ERRFILE="${GCPU}_m${MODEL}_b${BATCH}_e${EPOCH}_n${NODES}_${STRATEGY}_ft${FUSION_THRESHOLD}.err"
srun -n4 python SNGAN_dist_hvd.py $BATCH $EPOCH 1>$OPFILE 2>$ERRFILE