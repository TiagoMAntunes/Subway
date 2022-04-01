#ifndef GRAPH_CUH
#define GRAPH_CUH


#include "globals.hpp"

template <class E>
class Graph
{
private:

public:
	string graphFilePath;
	bool isWeighted;
	bool isLarge;
	size_t num_nodes;
	size_t num_edges;
	size_t *nodePointer;
	E *edgeList;
	uint *outDegree;
	bool *label1;
	bool *label2;
	uint *value;
	uint *d_outDegree;
	uint *d_value;
	bool *d_label1;
	bool *d_label2;
	string graphFormat;
    Graph(string graphFilePath, bool isWeighted);
    string GetFileExtension(string fileName);
    void AssignW8(uint w8, size_t index);
    void ReadGraph();
};

template <class E>
class GraphPR
{
private:

public:
	string graphFilePath;
	bool isWeighted;
	bool isLarge;
	size_t num_nodes;
	size_t num_edges;
	size_t *nodePointer;
	E *edgeList;
	uint *outDegree;
	float *value;
	float *delta;
	float *sum;
	uint *d_outDegree;
	float *d_value;
	float *d_delta;
	float *d_sum;
	string graphFormat;
    GraphPR(string graphFilePath, bool isWeighted);
    string GetFileExtension(string fileName);
    void AssignW8(uint w8, size_t index);
    void ReadGraph();
};

#endif	//	GRAPH_CUH



