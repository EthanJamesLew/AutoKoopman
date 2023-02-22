# Online System Identification via Koopman Operator Linearization 

Notes on existing online system id via Koopman operator linearization.

From Modern Koopman theory for dynamical systems [[1]](#1):

> Streaming and parallelized codes. Because of the computational burden of computing the DMD on high-resolution data, several advances have been made to accelerate DMD in streaming applications and with parallelized algorithms. DMD is often used in a streaming setting, where a moving window of snapshots are processed continuously, resulting in savings by eliminating redundant computations when new data becomes available. Several algorithms exist for streaming DMD, based on the incremental SVD [[4]](#4), a streaming method of snapshots SVD [[3]](#3), and rank-one updates to the DMD matrix [[2]](#2). The DMD algorithm is also readily parallelized, as it is based on the SVD. Several parallelized codes are available, based on the QR [[5]](#5) and SVD [[6](#6), [7](#7), [8](#8)] .

## References

<a id="1">[1]</a>  Brunton, S. L., Budišić, M., Kaiser, E., & Kutz, J. N. (2021). Modern Koopman theory for dynamical systems. *arXiv preprint arXiv:2102.12086*.

<a id="2">[2]</a> Zhang, H., Rowley, C. W., Deem, E. A., & Cattafesta, L. N. (2019). Online dynamic mode decomposition for time-varying systems. *SIAM Journal on Applied Dynamical Systems*, *18*(3), 1586-1609.

<a id="3">[3]</a> Pendergrass, S. D., Kutz, J. N., & Brunton, S. L. (2016). Streaming GPU singular value and dynamic mode decompositions. *arXiv preprint arXiv:1612.07875*.

<a id="4">[4]</a> Hemati, M. S., Williams, M. O., & Rowley, C. W. (2014). Dynamic mode decomposition for large and streaming datasets. *Physics of Fluids*, *26*(11), 111701.

<a id="5">[5]</a> Sayadi, T., & Schmid, P. J. (2016). Parallel data-driven decomposition algorithm for large-scale datasets: with application to transitional boundary layers. *Theoretical and Computational Fluid Dynamics*, *30*, 415-428.

<a id="6">[6]</a> Erichson, N. B., Mathelin, L., Kutz, J. N., & Brunton, S. L. (2019). Randomized dynamic mode decomposition. *SIAM Journal on Applied Dynamical Systems*, *18*(4), 1867-1891.

<a id="7">[7]</a> Erichson, N. B., Voronin, S., Brunton, S. L., & Kutz, J. N. (2016). Randomized matrix decompositions using R. arXiv preprint arXiv:1608.02148.

<a id="8">[8]</a> Erichson, N. B., Manohar, K., Brunton, S. L., & Kutz, J. N. (2020). Randomized CP tensor decomposition. Machine Learning: Science and Technology, 1(2), 025012.

