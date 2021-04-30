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
int k, V;

int* get_neighbors(int v) {
  int n = xadj[v+1] - xadj[v];
  if (n < 0)
    cout << n << "\t" << v << "\t" << xadj[v+1] << "\t" << xadj[v] << "\n" ;
  int* out = new int[n];
  std::copy(adj + xadj[v],adj + xadj[v+1], out);
  return out;
}
int cycles_3(int v1) {
  int count = 0;
  int* neighbors_v1 = get_neighbors(v1);
  int n1 = xadj[v1+1] - xadj[v1];
  for (int i1 = 0; i1 < n1; i1++) {
    int v2 = neighbors_v1[i1];
    int* neighbors_v2 = get_neighbors(v2);
    int n2 = xadj[v2+1] - xadj[v2];
    for (int i2 = 0; i2 < n2; i2++) {
      int v3 = neighbors_v2[i2];
      if (v3 == v1)
        continue;
      int* neighbors_v3 = get_neighbors(v3);
      int n3 = xadj[v3+1] - xadj[v3];
      for (int i3 = 0; i3 < n3; i3++) {
        int v4 = neighbors_v3[i3];
        
        if (v4 == v1) {
          count++;
          //cout << "Path " << v1 << " " << v2 << " " << v3 << " " << v4 << endl;
        }
      }
      delete neighbors_v3;
    }
    delete neighbors_v2;
  }
  delete neighbors_v1;
  return count;
}
int cycles_4(int v1) {
  int count = 0;
  int* neighbors_v1 = get_neighbors(v1);
  int n1 = xadj[v1+1] - xadj[v1];
  for (int i1 = 0; i1 < n1; i1++) {
    int v2 = neighbors_v1[i1];
    int* neighbors_v2 = get_neighbors(v2);
    int n2 = xadj[v2+1] - xadj[v2];
    for (int i2 = 0; i2 < n2; i2++) {
      int v3 = neighbors_v2[i2];
      if (v3 == v1)
        continue;
      int* neighbors_v3 = get_neighbors(v3);
      int n3 = xadj[v3+1] - xadj[v3];
      for (int i3 = 0; i3 < n3; i3++) {
        int v4 = neighbors_v3[i3];
        if (v4 == v3 || v4 == v2 || v4 == v1)
          continue;
        int* neighbors_v4 = get_neighbors(v4);
        int n4 = xadj[v4+1] - xadj[v4];
        for (int i4 = 0; i4 < n4; i4++) {
          int v5 = neighbors_v4[i4];
          if (v5 == v1) {
            count++;
          }
        }
        delete neighbors_v4;
      }
      delete neighbors_v3;
    }
    delete neighbors_v2;
  }
  delete neighbors_v1;
  return count;
}
int cycles_5(int v1) {
  int count = 0;
  int* neighbors_v1 = get_neighbors(v1);
  int n1 = xadj[v1+1] - xadj[v1];
  for (int i1 = 0; i1 < n1; i1++) {
    int v2 = neighbors_v1[i1];
    int* neighbors_v2 = get_neighbors(v2);
    int n2 = xadj[v2+1] - xadj[v2];
    for (int i2 = 0; i2 < n2; i2++) {
      int v3 = neighbors_v2[i2];
      if (v3 == v1)
        continue;
      int* neighbors_v3 = get_neighbors(v3);
      int n3 = xadj[v3+1] - xadj[v3];
      for (int i3 = 0; i3 < n3; i3++) {
        int v4 = neighbors_v3[i3];
        if (v4 == v3 || v4 == v2 || v4 == v1)
          continue;
        int* neighbors_v4 = get_neighbors(v4);
        int n4 = xadj[v4+1] - xadj[v4];
        for (int i4 = 0; i4 < n4; i4++) {
          int v5 = neighbors_v4[i4];
          if (v5 == v3 || v5 == v2 || v5 == v1 || v5 == v4)
            continue;
          int* neighbors_v5 = get_neighbors(v5);
          int n5 = xadj[v5+1] - xadj[v5];
          for (int i5 = 0; i5 < n5; i5++) {
            int v6 = neighbors_v5[i5];
            if (v6 == v1) {
              count++;
            }
          }
          delete neighbors_v5;
        }
        delete neighbors_v4;
      }
      delete neighbors_v3;
    }
    delete neighbors_v2;
  }
  delete neighbors_v1;
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
	V = currMax;

  xadj = new int[V + 1];
  results = new int[V];
  adj = new int[adjSize * 2];

  infile.clear();
  infile.seekg(0);
  
  xadj[0] = 0;
  int adj_i = 0;
  for (int i = 0; i < V+1; i++) {
    xadj[i+1] = xadj[i] + mat[i].size();
    for (int j = 0; j < mat[i].size(); j++) {
      adj[adj_i] = mat[i][j];
      adj_i++;
    }
  }
  infile.close();
	cout << "count cycles " << endl;

  double start = omp_get_wtime();
  if (k == 3) {
    #pragma omp parallel for num_threads(24) schedule(dynamic)
    for (int i = 0; i <= V; i++) {
      int r = cycles_3(i);
      results[i] = r;
      if (i%2000 == 0) {
      //cout << "Result:" << i << "\t" << r << endl;
      }
    }
  } else if (k == 4) {
    #pragma omp parallel for num_threads(24) schedule(dynamic)
    for (int i = 0; i <= V; i++) {
      int r = cycles_4(i);
      results[i] = r;
      if (i%2000 == 0) {
      //cout << "Result:" << i << "\t" << r << endl;
      }
    }
  } else if (k == 5) {
    #pragma omp parallel for num_threads(24) schedule(dynamic)
    for (int i = 0; i <= V; i++) {
      int r = cycles_5(i);
      results[i] = r;
      if (i%2000 == 0) {
      //cout << "Result:" << i << "\t" << r << endl;
      }
    }
  } else {
    cout << "ERROR: Invalid k" << endl;
  }
  double duration = omp_get_wtime() - start;
  cout <<"TIME: " << duration << endl;
  ofstream outfile;
  outfile.open ("results.txt");
  for (int i = 0; i <= V; i++) {
    outfile << i << " " << results[i] << endl;
  }
  outfile.close();

  
	//cout << "Result: " << cycles_3(0) << endl;
  cout << "count cycles finished" << endl;

}


int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("invalid number of arguments\n");
    return 1;
  }

  char* filePath = argv[1];
  char* kStr = argv[2];
  k = atoi(kStr);
  

  readFile(filePath, k);
  return 0;
}
