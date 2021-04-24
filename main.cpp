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

const int V = 5;
auto adjMatrix = new int[V][V];

int dfs(bool* marked, int k, int v, int start, int count) {
    marked[v] = true;

    // v => baktigimiz node // row
    // start => baktigimiz node'un baslangici
    // xadj => her bir rowda

    if (k == 0) {
        marked[v] = false;
        if (adjMatrix[v][start] == 1) {
            count++;
        }
        return count;
    }

    for (int i = 0; i < V; i++) {
        if (marked[i] == false && adjMatrix[v][i] == 1) {
            count = dfs(marked, k - 1, i, start, count);
        }
    }
    marked[v] = false;
    return count;
}

// seqCountCycles counts the cycles of length k
// for each vertices in a sequential manner.
void seqCountCycles(int k, int* nov) {
    printf("nov - 1 %d\n", (*nov) - 1);

    int lV = (*nov) - 1;

    bool* markedx = new bool[lV];
    auto freq = new int[lV];
    for (int i = 0; i < lV; i++) {
        int count = 0;
        count = dfs(markedx, k - 1, i, i, count);
        freq[i] = count;
    }

    // Print the number of cycles of length k
    for (int i = 0; i < lV; i++) {
        cout << "i: " << i << "\tfreq[i] " << freq[i] / 2 << endl;
    }
}

int readFile(string filePath) {
    ifstream infile(filePath);

    string line;

    int nov = 0;
    getline(infile, line);
    istringstream iss(line);
    iss >> nov;

    int V = nov;

    printf("nov: %d\n", nov);

    int noOne = 0;
    while (getline(infile, line)) {
        istringstream iss(line);
        int u = 0, v = 0;
        iss >> u >> v;
        noOne++;
        //	printf("u: %d\tv: %d\n", u, v);
    }
    printf("no of one: %d\n", noOne * 2);

    // xadj olusturcam size i V+1
    // adj olusturcam nov

    int* xadj = new int[V + 1];
    int* adj = new int[noOne];

    infile.clear();
    infile.seekg(0);
    getline(infile, line);

    int uPrev = -1, counter = 0;
    int u = 0, v = 0;

    unordered_map<int, int> map;
    unordered_map<int, vector<int>> order;

    while (getline(infile, line)) {
        istringstream iss(line);
        iss >> u >> v;

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

        printf("u: %d\tv: %d\n", u, v);
    }

    xadj[0] = 0;
    int prev = 0;
    for (int i = 0; i < V; i++) {
        xadj[i + 1] = prev + map[i];
        prev = xadj[i + 1];
    }

    int k = 0;
    for (int i = 0; i < noOne; i++) {
        vector<int> x = order[i];
        for (int j = 0; j < x.size(); j++) {
            adj[k] = x[j];
            k++;
        }
    }

    printf("\n");
    for (int i = 0; i < V + 1; i++) {
        cout << "i: " << i << " xadj[i]: " << xadj[i] << endl;
    }

    printf("\nV = %d\n", V);
    for (int i = 0; i < V; i++) {
        vector<int> tmp = order[i];
        cout << "i: " << i << "\n";
        for (int x = 0; x < tmp.size(); x++) {
            cout << " order[" << x << "]: " << tmp[x];
        }
        cout << "finished" << endl;
        // cout << "i: " << i << " adj[i]: " << adj[i] << endl;
    }
    infile.close();
    return 1;
}

int main(int argc, char* argv[]) {
    char* filePath = argv[1];
    char* kStr = argv[2];
    int k = atoi(kStr);

    readFile(filePath);
    return 0;
}
