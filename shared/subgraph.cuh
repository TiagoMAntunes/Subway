#ifndef SUBGRAPH_HPP
#define SUBGRAPH_HPP


#include "globals.hpp"


template <class E>
class Subgraph
{
private:

public:
	size_t num_nodes;
	size_t num_edges;
	size_t numActiveNodes;
	
	size_t *activeNodes;
	size_t *activeNodesPointer;
	E *activeEdgeList;
	
	size_t *d_activeNodes;
	size_t *d_activeNodesPointer;
	E *d_activeEdgeList;
	
	ull max_partition_size;
	
	Subgraph(size_t num_nodes, size_t num_edges);
};

#endif	//	SUBGRAPH_HPP



