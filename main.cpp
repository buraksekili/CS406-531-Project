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

int* adj, *xadj;
int k, V;

// getElement returns the element of r'th row and c'th column.
int getElement(int r, int c) {
	int startIdx = xadj[r], endIdx = xadj[r+1];	
  for (int i = startIdx; i < endIdx; i++) {
		int curr = adj[i];
		if (curr == c) {
			return 1;
		}
	}	
	return -1;
}

int dfs(bool *marked, int k, int v, int start, int count) {
  marked[v] = true;
  if (k == 0) {
 
    marked[v] = false;
    // adjMatrix[v][start] == 1
    if(getElement(v, start) == 1) {
      count++;
    }
    return count;
  }

  for (int i = 0; i < V+1; i++) {
    if (marked[i] == false && getElement(v, i) == 1) {
      count = dfs(marked, k-1, i, start,count);
    }
  }
  marked[v] = false;
  return count;
}


// seqCountCycles counts the cycles of length k
// for each vertices in a sequential manner.
void countCycles(int V) {
  int lV = V + 1;
  // int lV = V - 1;

 // bool* markedx = new bool[lV];
  auto freq = new int[lV];
  int count = 0;
  cout << "starting! " << endl;
  double start = omp_get_wtime();
  #pragma omp parallel for firstprivate(count) 
  for (int i = 0; i < lV; i++) {   
  
    bool* markedx = new bool[lV];
    count = dfs(markedx, k-1, i, i, count);
   
    freq[i] = count;
    count = 0;
    delete[] markedx;
  }
    double end = omp_get_wtime();
    cout << "finished in " << end - start << endl;
  // Print the number of cycles of length k
  /*
  for (int i = 0; i < lV; i++) {
    cout << "i: " << i << "\tfreq[i] " << ceil((float)(freq[i] / 2)) << endl;
  } 
  */
}

void readFile(string filePath) {
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
    }
  }

  // File contains m number of vertices. Get m from the lastLine.
	V = currMax;

  xadj = new int[V + 1];
  adj = new int[adjSize * 2];

  infile.clear();
  infile.seekg(0);

  
  unordered_map<int, int> map;
  unordered_map<int, vector<int>> order;
  while (getline(infile, line)) {
    if (line.length() != 0) {
      stringstream ss(line);
      ss >> u >> v;
    
			if (u == v) {
				continue;
			}

			// if u is not initialized.
      if (map.find(u) == map.end()) {
        map[u] = 1;
				if (map.find(v) == map.end()) {
					map[v] = 1;
				} else {
					map[v]++;
				}

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

  infile.close();
	cout << "count cycles " << endl;
	countCycles(V);
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

  readFile(filePath);
  return 0;
}