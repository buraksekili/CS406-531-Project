#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

int* adj, *xadj, *results;
int k, nnz, nov;

void wrapper(int* adj, int* xadj, int* results, int nnz, int nov, int k);

int* get_neighbors(int v) {
  int n = xadj[v+1] - xadj[v];
  int* out = new int[n];
  std::copy(adj + xadj[v],adj + xadj[v+1], out);
  return out;
}

int cycles_3(int v1) {
  int count = 0;
  int n1 = xadj[v1+1] - xadj[v1];
  for (int i1 = 0; i1 < n1; i1++) {
    int v2 = adj[xadj[v1]+i1];
    int n2 = xadj[v2+1] - xadj[v2];
    for (int i2 = 0; i2 < n2; i2++) {
      int v3 = adj[xadj[v2]+i2];
      if (v3 == v1)
        continue;
      int n3 = xadj[v3+1] - xadj[v3];
      for (int i3 = 0; i3 < n3; i3++) {
        int v4 = adj[xadj[v3]+i3];
        if (v4 == v1) {
          count++;
          //cout << "Path " << v1 << " " << v2 << " " << v3 << " " << v4 << endl;
        }
      }
    }
  }
  return count;
}
int cycles_4(int v1) {
  int count = 0;
  int n1 = xadj[v1+1] - xadj[v1];
  for (int i1 = 0; i1 < n1; i1++) {
    int v2 = adj[xadj[v1]+i1];
    int n2 = xadj[v2+1] - xadj[v2];
    for (int i2 = 0; i2 < n2; i2++) {
      int v3 = adj[xadj[v2]+i2];
      if (v3 == v1)
        continue;
      int n3 = xadj[v3+1] - xadj[v3];
      for (int i3 = 0; i3 < n3; i3++) {
        int v4 = adj[xadj[v3]+i3];
        if (v4 == v3 || v4 == v2 || v4 == v1)
          continue;
        int n4 = xadj[v4+1] - xadj[v4];
        for (int i4 = 0; i4 < n4; i4++) {
          int v5 = adj[xadj[v4]+i4];
          if (v5 == v1) {
            count++;
          }
        }
      }
    }
  }
  return count;
}
int cycles_5(int v1) {
  int count = 0;
  int n1 = xadj[v1+1] - xadj[v1];
  for (int i1 = 0; i1 < n1; i1++) {
    int v2 = adj[xadj[v1]+i1];
    int n2 = xadj[v2+1] - xadj[v2];
    for (int i2 = 0; i2 < n2; i2++) {
      int v3 = adj[xadj[v2]+i2];
      if (v3 == v1)
        continue;
      int n3 = xadj[v3+1] - xadj[v3];
      for (int i3 = 0; i3 < n3; i3++) {
        int v4 = adj[xadj[v3]+i3];
        if (v4 == v3 || v4 == v2 || v4 == v1)
          continue;
        int n4 = xadj[v4+1] - xadj[v4];
        for (int i4 = 0; i4 < n4; i4++) {
          int v5 = adj[xadj[v4]+i4];
          if (v5 == v3 || v5 == v4 || v5 == v2 || v5 == v1)
            continue;
          int n5 = xadj[v5+1] - xadj[v5];
          for (int i5 = 0; i5 < n5; i5++) {
            int v6 = adj[xadj[v5]+i5];
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
  nnz = adjSize*2;
  nov = currMax+1;

  xadj = new int[nov+1];
  results = new int[nov-1];
  adj = new int[adjSize * 2];
  
  infile.clear();
  infile.seekg(0);
  
  xadj[0] = 0;
  int adj_i = 0;
  for (int i = 0; i < nov; i++) {
    xadj[i+1] = xadj[i] + mat[i].size();
    for (int j = 0; j < mat[i].size(); j++) {
      adj[adj_i] = mat[i][j];
      adj_i++;
    }
  }
  infile.close();
}

void save_results() {
  ofstream outfile;
  outfile.open ("results.txt");
  for (int i = 0; i < nov; i++) {
    outfile << i << " " << results[i] << endl;
  }
  outfile.close();
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
  cout <<"TIME: " << duration << endl;
  cout << "count cycles finished" << endl;
}
void cycles_gpu(int k) {
  cout << "count cycles GPU" << endl;
  wrapper(adj, xadj, results, nnz, nov, k);
}

int main(int argc, char* argv[]) {
  if (argc != 3 && argc != 4) {
    printf("invalid number of arguments\n");
    return 1;
  }
  int nt = omp_get_max_threads();
  if (argc == 4) {
    nt = atoi(argv[3]);
  }
  
  char* filePath = argv[1];
  char* kStr = argv[2];
  k = atoi(kStr);

  readFile(filePath, k);
  
  if (nt == 0) {
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