### How to run
`python main.py`

Running on Windows may result in MemoryError due to insufficient memory allocation for the large matrix containing the ratings.

Running on Linux should be fine, but if memory allocation is insufficient, consider running as root the following command for permitting extra memory allocation

`echo 1 > /proc/sys/vm/overcommit_memory`

