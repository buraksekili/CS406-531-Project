# [CS406/531 Project](https://github.com/buraksekili/CS406-531-Project)

The project of CS406/531, Parallel Computing course, includes the design of an algorithm which takes an undirected graph G=(V, E) as the input and finds every single cycle of a given length, denoted by k, whose value ranges between 2 and 6. For every vertex present inside the graph, it records the number of cycles of length k present for each vertex in the process.

## Final Report

The final report of the project is available on [https://github.com/buraksekili/CS406-531-Project/blob/master/CS406_531_Project_FinalReport.pdf](https://github.com/buraksekili/CS406-531-Project/blob/master/CS406_531_Project_FinalReport.pdf)

## Installation

`git clone https://github.com/buraksekili/CS406-531-Project.git`


The required version of the `gcc` is `8.2.0` for the CPU code `main.cpp`. 

You must change the version of gcc by running; 

```shell
$ module load gcc/8.2.0
```

For GPU code, `cuda/10.0` and `gcc/7.5.0` is required. 

You must change the version of gcc and CUDA by running; 

```shell
$ module load gcc/7.5.0
$ module load cuda/10.0
```

## Usage

Compile the program with;

```shell
$ nvcc main.cu kernel.cu -O3 -o program.out -Xcompiler -fopenmp
```

For running on CPU with `n` threads, `c` cycles: 
```shell
./program.out ./amazon.txt c n
```

For running on GPU with `n` threads, `c` cycles: 
```shell
/program.out ./amazon.txt c 0
```

For running CPU and GPU together with c cycles: 

```shell
./program.out ./amazon.txt c -1
```


