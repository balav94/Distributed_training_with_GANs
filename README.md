# Distributed_training_with_GANs
This repository contains the various Generative Adversarial Network (GAN) architectures, specifically Deep Convolutional GAN (DCGAN), Spectral Normalization GAN (SNGAN), Wasserstein GAN with Gradient Penalty (WGAN-GP) used to explore distributed training using Tensorflow/Horovod.

(1) First of all, as a baseline, single-GPU and single-CPU training were carried out by varying the global batch size between 8-128,which shows that as the batch size increased, throughput increases. 

(2) Moreover, weak scaling and strong scaling experiments were conducted on both CPUs and GPUs, by varying the number of nodes between 1-8, which shows the expected trend with weak scaling generating more throughput with distributed training on the same number of nodes,compared to strong scaling.

(3) Further, we also explored other optimizations in Horovod such as FP16 compression, Tensor Fusion, and Hierarchical All-reduce, which owing to relatively small model size, did not yield any substantial improvement, probably because of small model sizes.

Due to the limited scope of the project, futher experiments could not be carried out.
