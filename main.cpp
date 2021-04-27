#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// const int V = 5;
// auto adjMatrix = new int[V][V];
/*
int dfs(bool *marked, int k, int v, int start, int count) {
  marked[v] = true;

  // v => baktigimiz node // row
  // start => baktigimiz node'un baslangici
  // xadj => her bir rowda

  if (k == 0) {
    marked[v] = false;
    if(adjMatrix[v][start] == 1) {
      count++;
    }
    return count;
  }

  for (int i = 0; i < V; i++) {
    if (marked[i] == false && adjMatrix[v][i] == 1) {
      count = dfs(marked, k-1, i, start,count);
    }
  }
  marked[v] = false;
  return count;
}

// seqCountCycles counts the cycles of length k
// for each vertices in a sequential manner.
void seqCountCycles(int k, int* nov) {

  printf("nov - 1 %d\n", (*nov) - 1 );

  int lV = (*nov) - 1;

  bool* markedx = new bool[lV];
  auto freq = new int[lV];
  for (int i = 0; i < lV; i++) {
    int count = 0;
    count = dfs(markedx, k-1, i, i, count);
    freq[i] = count;
  }

  // Print the number of cycles of length k
  for (int i = 0; i < lV; i++) {
    cout << "i: " << i << "\tfreq[i] " << freq[i]/2 << endl;
  }
}
*/

void readFile(string filePath) {
  ifstream infile(filePath);
  if (!infile.is_open()) {
    cout << "cannot open file: " << filePath << endl;
    return;
  }

  // lastLine keeps previous read line. If the last line of the given file
  // is empty (e.g, '\n'), we can use lastLine to get the last line.
  string line, lastLine;
  int adjSize = 0;
  while (getline(infile, line)) {
    if (line.length() != 0) {
      lastLine = line;
      adjSize++;
    }
  }

  // File contains m number of vertices. Get m from the lastLine.
  stringstream ss(lastLine);
  int V1 = 0, V2 = 0;
  ss >> V1 >> V2;
  int V = max(V1, V2);

  int* xadj = new int[V + 1];
  int* adj = new int[adjSize * 2];

  infile.clear();
  infile.seekg(0);

  int u = 0, v = 0;
  unordered_map<int, int> map;
  unordered_map<int, vector<int>> order;
  while (getline(infile, line)) {
    if (line.length() != 0) {
      stringstream ss(line);
      ss >> u >> v;

      if (map.find(u) == map.end()) {
        map[u] = 1;
        map[v] = 1;

        vector<int> tmp;
        tmp.push_back(v);
        order[u] = tmp;

        vector<int> tmp2;
        tmp2.push_back(u);
        order[v] = tmp2;
      } else {
        map[u]++;
        map[v]++;

        order[u].push_back(v);
        order[v].push_back(u);
      }
    }
  }
  xadj[0] = 0;
  int prev = 0;
  for (int i = 1; i < V + 2; i++) {
    xadj[i] = prev + map[i - 1];
    prev = xadj[i];
  }

  int k = 0;
  for (int i = 0; i < adjSize; i++) {
    vector<int> x = order[i];
    for (int j = 0; j < x.size(); j++) {
      adj[k] = x[j];
      k++;
    }
  }

  cout << "V: " << V << endl;
  for (int i = 0; i < adjSize * 2; i++) {
    cout << "i " << i << '\t' << adj[i] << endl;
  }

  for (int i = 0; i < V + 2; i++) {
    cout << "i " << i << '\t' << xadj[i] << endl;
  }
  infile.close();
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("invalid number of arguments\n");
    return 1;
  }

  char* filePath = argv[1];
  char* kStr = argv[2];
  int k = atoi(kStr);

  readFile(filePath);
  return 0;
}
