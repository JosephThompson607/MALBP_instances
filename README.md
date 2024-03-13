# Mixed Model Assembly Line Balancing Instances

This repository contains Mixed Model Assembly Line Balancing Instances and methods for creating Mixed Model Assembly Line Instances.
The instances are stored in yaml files for easy access.

The base precedence relations and task times are created from SALBP instances from the benchmark data set of [Otto et al. (2013)](https://assembly-line-balancing.de/salbp/benchmark-data-sets-2013/). The base instances are in the folder "SALBP_benchmark".

We have created mixed model assembly line instances by either combining 2 or more different SALBP instances, with each instance being a model, or taking one instance and changing it to create different models.

In this library, we have tools for randomly deleting and permuting tasks for assembly line balancing instances.

## Citations

If you use the Mixed Model Assembly Line Balancing Instances or the methods provided in this repository, please cite this repository:



## References

1. Otto, A., Scholl, A., & Klein, R. (2013). Benchmark data sets for mixed-model assembly line balancing. European Journal of Operational Research, 228(2), 402-413. [Link](https://assembly-line-balancing.de/salbp/benchmark-data-sets-2013/)