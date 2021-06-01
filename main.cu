#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)
#define NO_THREADS 128
#define CHUNK_SIZE 2048
int NO_BLOCKS = ceil(CHUNK_SIZE / NO_THREADS);
struct job {
  int index;
  int cost;
  bool operator<(const job &a) const { return cost < a.cost; }
};
// Error check-----
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
// Error check-----
using namespace std;

int *adj, *xadj, *results, *results_cpu;
int *d_adj, *d_xadj, *d_results;
int k, nnz, nov, breakpoint;
deque<job> jobs;
std::mutex job_mutex;

void wrapper(int *adj, int *xadj, int *results, int nnz, int nov, int k);

void populate_jobs() {
  for (int index = 0; index * CHUNK_SIZE < nov; index++) {
    job j = {};
    j.index = index;
    j.cost =
        xadj[min((index + 1) * CHUNK_SIZE, nov)] - xadj[index * CHUNK_SIZE];
    jobs.push_back(j);
  }
  sort(jobs.begin(), jobs.end());
}
int *get_neighbors(int v) {
  int n = xadj[v + 1] - xadj[v];
  int *out = new int[n];
  std::copy(adj + xadj[v], adj + xadj[v + 1], out);
  return out;
}
int cycles_3(int v1) {
  int count = 0;
  int n1 = xadj[v1 + 1] - xadj[v1];
  for (int i1 = 0; i1 < n1; i1++) {
    int v2 = adj[xadj[v1] + i1];
    int n2 = xadj[v2 + 1] - xadj[v2];
    for (int i2 = 0; i2 < n2; i2++) {
      int v3 = adj[xadj[v2] + i2];
      if (unlikely(v3 == v1))
        continue;
      int n3 = xadj[v3 + 1] - xadj[v3];
      for (int i3 = 0; i3 < n3; i3++) {
        int v4 = adj[xadj[v3] + i3];
        if (unlikely(v4 == v1)) {
          count++;
        }
      }
    }
  }
  return count;
}
int cycles_4(int v1) {
  int count = 0;
  int n1 = xadj[v1 + 1] - xadj[v1];
  for (int i1 = 0; i1 < n1; i1++) {
    int v2 = adj[xadj[v1] + i1];
    int n2 = xadj[v2 + 1] - xadj[v2];
    for (int i2 = 0; i2 < n2; i2++) {
      int v3 = adj[xadj[v2] + i2];
      if (unlikely(v3 == v1))
        continue;
      int n3 = xadj[v3 + 1] - xadj[v3];
      for (int i3 = 0; i3 < n3; i3++) {
        int v4 = adj[xadj[v3] + i3];
        if (unlikely(v4 == v3 || v4 == v2 || v4 == v1))
          continue;
        int n4 = xadj[v4 + 1] - xadj[v4];
        for (int i4 = 0; i4 < n4; i4++) {
          int v5 = adj[xadj[v4] + i4];
          count = (v5 == v1) ? count + 1 : count;
        }
      }
    }
  }
  return count;
}
int cycles_5(int v1) {
  int count = 0;
  int n1 = xadj[v1 + 1] - xadj[v1];
  for (int i1 = 0; i1 < n1; i1++) {
    int v2 = adj[xadj[v1] + i1];
    int n2 = xadj[v2 + 1] - xadj[v2];
    for (int i2 = 0; i2 < n2; i2++) {
      int v3 = adj[xadj[v2] + i2];
      if (v3 == v1)
        continue;
      int n3 = xadj[v3 + 1] - xadj[v3];
      for (int i3 = 0; i3 < n3; i3++) {
        int v4 = adj[xadj[v3] + i3];
        if (v4 == v3 || v4 == v2 || v4 == v1)
          continue;
        int n4 = xadj[v4 + 1] - xadj[v4];
        for (int i4 = 0; i4 < n4; i4++) {
          int v5 = adj[xadj[v4] + i4];
          if (v5 == v3 || v5 == v4 || v5 == v2 || v5 == v1)
            continue;
          int n5 = xadj[v5 + 1] - xadj[v5];
          for (int i5 = 0; i5 < n5; i5++) {
            int v6 = adj[xadj[v5] + i5];
            if (v6 == v1) {
              count++;
            }
          }
        }
      }
    }
  }
  return count;
}

void readFile(string filePath, int k) {
  ifstream infile(filePath);
  if (!infile.is_open()) {
    cout << "cannot open file: " << filePath << endl;
    return;
  }

  // lastLine keeps previous read line. If the last line of the given file
  // is empty (e.g, '\n'), we can use lastLine to get the last line.
  string line, lastLine;
  int adjSize = 0, currMax = 0;
  int u = 0, v = 0;
  unordered_map<int, vector<int>> mat;
  while (getline(infile, line)) {
    stringstream ss(line);

    if (line.length() != 0) {
      ss >> u >> v;
      int localMax = max(u, v);
      if (localMax > currMax) {
        currMax = localMax;
      }
      lastLine = line;
      adjSize++;
      if (mat.find(u) == mat.end()) {
        vector<int> val;
        val.push_back(v);
        mat[u] = val;
      } else {
        mat[u].push_back(v);
      }
      if (mat.find(v) == mat.end()) {
        vector<int> val;
        val.push_back(u);
        mat[v] = val;
      } else {
        mat[v].push_back(u);
      }
    }
  }
  // File contains m number of vertices. Get m from the lastLine.
  nnz = adjSize * 2;
  nov = currMax + 1;
  breakpoint = nov / 2;

  // xadj = new int[nov+1];
  cudaError_t status = cudaMallocHost((void **)&xadj, sizeof(int) * (nov + 1));
  if (status != cudaSuccess)
    printf("Error allocating pinned host memory\n");
  // results = new int[nov-1];
  status = cudaMallocHost((void **)&results, sizeof(int) * (nov));
  if (status != cudaSuccess)
    printf("Error allocating pinned host memory\n");
  status = cudaMallocHost((void **)&results_cpu, sizeof(int) * (nov));
  if (status != cudaSuccess)
    printf("Error allocating pinned host memory\n");
  // adj = new int[adjSize * 2];
  status = cudaMallocHost((void **)&adj, sizeof(int) * (adjSize * 2));
  if (status != cudaSuccess)
    printf("Error allocating pinned host memory\n");
  for (int i = 0; i < nov; i++) {
    results[i] = -1;
    results_cpu[i] = -1;
  }
  infile.clear();
  infile.seekg(0);

  xadj[0] = 0;
  int adj_i = 0;
  for (int i = 0; i < nov; i++) {
    xadj[i + 1] = xadj[i] + mat[i].size();
    for (int j = 0; j < mat[i].size(); j++) {
      adj[adj_i] = mat[i][j];
      adj_i++;
    }
  }
  infile.close();
  populate_jobs();
}
void save_results() {
  ofstream outfile;
  outfile.open("results.txt");
  for (int i = 0; i < nov; i++) {
    int result = max(results[i], results_cpu[i]);
    outfile << i << " " << result << endl;
  }
  outfile.close();
}
__global__ void d_cycles3(int *adj, int *xadj, int *results, int nnz, int nov,
                          int start_point) {
  int v1 = threadIdx.x + blockDim.x * blockIdx.x + start_point;
  if (v1 < start_point + CHUNK_SIZE) {
    int count = 0;
    int n1 = xadj[v1 + 1] - xadj[v1];
    for (int i1 = 0; i1 < n1; i1++) {
      int v2 = adj[xadj[v1] + i1];
      int n2 = xadj[v2 + 1] - xadj[v2];
      for (int i2 = 0; i2 < n2; i2++) {
        int v3 = adj[xadj[v2] + i2];
        if (v3 == v1)
          continue;
        int n3 = xadj[v3 + 1] - xadj[v3];
        for (int i3 = 0; i3 < n3; i3++) {
          int v4 = adj[xadj[v3] + i3];
          if (v4 == v1) {
            count++;
          }
        }
      }
    }
    results[v1] = count;
  }
}
__global__ void d_cycles4(int *adj, int *xadj, int *results, int nnz, int nov,
                          int start_point) {
  int v1 = start_point + threadIdx.x + blockDim.x * blockIdx.x;
  if (v1 < start_point + CHUNK_SIZE) {
    int count = 0;
    int n1 = xadj[v1 + 1] - xadj[v1];
    for (int i1 = 0; i1 < n1; i1++) {
      int v2 = adj[xadj[v1] + i1];
      int n2 = xadj[v2 + 1] - xadj[v2];
      for (int i2 = 0; i2 < n2; i2++) {
        int v3 = adj[xadj[v2] + i2];
        if (v3 == v1)
          continue;
        int n3 = xadj[v3 + 1] - xadj[v3];
        for (int i3 = 0; i3 < n3; i3++) {
          int v4 = adj[xadj[v3] + i3];
          if (v4 == v3 || v4 == v2 || v4 == v1)
            continue;
          int n4 = xadj[v4 + 1] - xadj[v4];
          for (int i4 = 0; i4 < n4; i4++) {
            int v5 = adj[xadj[v4] + i4];
            if (v5 == v1) {
              count++;
            }
          }
        }
      }
    }
    results[v1] = count;
  }
}
__global__ void d_cycles5(int *adj, int *xadj, int *results, int nnz, int nov,
                          int start_point) {
  int v1 = start_point + threadIdx.x + blockDim.x * blockIdx.x;
  if (v1 < start_point + CHUNK_SIZE) {
    int count = 0;
    int n1 = xadj[v1 + 1] - xadj[v1];
    for (int i1 = 0; i1 < n1; i1++) {
      int v2 = adj[xadj[v1] + i1];
      int n2 = xadj[v2 + 1] - xadj[v2];
      for (int i2 = 0; i2 < n2; i2++) {
        int v3 = adj[xadj[v2] + i2];
        if (v3 == v1)
          continue;
        int n3 = xadj[v3 + 1] - xadj[v3];
        for (int i3 = 0; i3 < n3; i3++) {
          int v4 = adj[xadj[v3] + i3];
          if (v4 == v3 || v4 == v2 || v4 == v1)
            continue;
          int n4 = xadj[v4 + 1] - xadj[v4];
          for (int i4 = 0; i4 < n4; i4++) {
            int v5 = adj[xadj[v4] + i4];
            if (v5 == v3 || v5 == v4 || v5 == v2 || v5 == v1)
              continue;
            int n5 = xadj[v5 + 1] - xadj[v5];
            for (int i5 = 0; i5 < n5; i5++) {
              int v6 = adj[xadj[v5] + i5];
              if (v6 == v1) {
                count++;
              }
            }
          }
        }
      }
    }
    results[v1] = count;
  }
}

void gpu_manager(int k) {
  printf("Started GPU Thread\n");

  while (true) {
    job j;
    job_mutex.lock();
    if (jobs.size() != 0) {
      j = jobs.front();
      jobs.pop_front();
    } else {
      job_mutex.unlock();
      return;
    }
    job_mutex.unlock();
    // printf("GPU JOB: Index: %d\tCost:%d\n", j.index, j.cost);

    if (k == 3) {
      d_cycles3<<<NO_BLOCKS, NO_THREADS>>>(d_adj, d_xadj, d_results, nnz, nov,
                                           j.index * CHUNK_SIZE);
    } else if (k == 4) {
      d_cycles4<<<NO_BLOCKS, NO_THREADS>>>(d_adj, d_xadj, d_results, nnz, nov,
                                           j.index * CHUNK_SIZE);
    } else {
      d_cycles5<<<NO_BLOCKS, NO_THREADS>>>(d_adj, d_xadj, d_results, nnz, nov,
                                           j.index * CHUNK_SIZE);
    }
    cudaDeviceSynchronize();
    // printf("GPU Job Finished\n");
  }
}

void cpu_manager(int k) {
  printf("Started CPU Thread\n");

  while (true) {
    job j;
    job_mutex.lock();
    if (jobs.size() != 0) {
      j = jobs.back();
      jobs.pop_back();
    } else {
      job_mutex.unlock();
      return;
    }
    job_mutex.unlock();
    // printf("CPU JOB: Index: %d\tCost:%d\n", j.index, j.cost);
    if (k == 3) {
#pragma omp parallel for schedule(dynamic) num_threads(8)
      for (int i = j.index * CHUNK_SIZE;
           i < min((j.index + 1) * CHUNK_SIZE, nov); i++) {
        int r = cycles_3(i);
        results_cpu[i] = r;
      }
    } else if (k == 4) {
#pragma omp parallel for schedule(dynamic) num_threads(8)
      for (int i = j.index * CHUNK_SIZE;
           i < min((j.index + 1) * CHUNK_SIZE, nov); i++) {
        int r = cycles_4(i);
        results_cpu[i] = r;
      }
    } else {
#pragma omp parallel for schedule(dynamic) num_threads(8)
      for (int i = j.index * CHUNK_SIZE;
           i < min((j.index + 1) * CHUNK_SIZE, nov); i++) {
        int r = cycles_5(i);
        results_cpu[i] = r;
      }
    }
    // printf("CPU Job Finished\n");
  }
}
void cycles_cpu(int k) {
  cout << "count cycles CPU" << endl;

  double start = omp_get_wtime();
  if (k == 3) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nov; i++) {
      int r = cycles_3(i);
      results[i] = r;
    }
  } else if (k == 4) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nov; i++) {
      int r = cycles_4(i);
      results[i] = r;
    }
  } else if (k == 5) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nov; i++) {
      int r = cycles_5(i);
      results[i] = r;
    }
  } else {
    cout << "ERROR: Invalid k" << endl;
  }
  double duration = omp_get_wtime() - start;
  cout << "TIME: " << duration << endl;
  cout << "count cycles finished" << endl;
}
void cycles_hybrid(int k) {
  cout << "count cycles CPU+GPU" << endl;

  cudaMalloc((void **)&d_adj, nnz * sizeof(int));
  cudaMalloc((void **)&d_xadj, (nov + 1) * sizeof(int));
  cudaMemcpy(d_adj, adj, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xadj, xadj, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_results, (nov) * sizeof(int));
  cudaMemcpy(d_results, results, (nov) * sizeof(int), cudaMemcpyHostToDevice);

  double start = omp_get_wtime();

  thread gpu_thread(gpu_manager, k);
  thread cpu_thread(cpu_manager, k);
  gpu_thread.join();
  cpu_thread.join();

  gpuErrchk(cudaDeviceSynchronize());
  double duration = omp_get_wtime() - start;
  cout << "TIME: " << duration << endl;
  cout << "count cycles finished" << endl;

  cudaMemcpy(results, d_results, (nov) * sizeof(int), cudaMemcpyDeviceToHost);
}

void cycles_gpu(int k) {
  cout << "count cycles GPU" << endl;
  wrapper(adj, xadj, results, nnz, nov, k);
}

int main(int argc, char *argv[]) {
  if (argc != 3 && argc != 4) {
    printf("invalid number of arguments\n");
    return 1;
  }
  int nt = omp_get_max_threads();
  if (argc == 4) {
    nt = atoi(argv[3]);
  }

  char *filePath = argv[1];
  char *kStr = argv[2];
  k = atoi(kStr);

  readFile(filePath, k);

  if (nt == -1) {
    cout << "Using Hybrid\n";
    omp_set_num_threads(32);
    cycles_hybrid(k);
  } else if (nt == 0) {
    cout << "Using GPU\n";
    cycles_gpu(k);

  } else {
    omp_set_num_threads(nt);
    cout << "Using " << nt << " threads.\n";
    cycles_cpu(k);
  }
  save_results();

  return 0;
}