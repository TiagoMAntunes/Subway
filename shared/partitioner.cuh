#ifndef PARTITIONER_CUH
#define PARTITIONER_CUH


#include "globals.hpp"
#include "subgraph.cuh"

template <class E>
class Partitioner
{
private:

public:
	uint numPartitions;
	vector<size_t> fromNode;
	vector<size_t> fromEdge;
	vector<size_t> partitionNodeSize;
	vector<size_t> partitionEdgeSize;
	Partitioner();
    void partition(Subgraph<E> &subgraph, uint numActiveNodes);
    void reset();
};

#endif	//	PARTITIONER_CUH



