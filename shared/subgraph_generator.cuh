#ifndef SUBGRAPH_GENERATOR_HPP
#define SUBGRAPH_GENERATOR_HPP


#include "globals.hpp"
#include "graph.cuh"
#include "subgraph.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thread>

template <class E>
class SubgraphGenerator
{
private:

public:
	unsigned int *activeNodesLabeling;
	size_t *activeNodesDegree;
	size_t *prefixLabeling;
	size_t *prefixSumDegrees;
	unsigned int *d_activeNodesLabeling;
	size_t *d_activeNodesDegree;
	size_t *d_prefixLabeling;
	size_t *d_prefixSumDegrees;
	SubgraphGenerator(Graph<E> &graph);
	SubgraphGenerator(GraphPR<E> &graph);
	void generate(Graph<E> &graph, Subgraph<E> &subgraph);
	void generate(GraphPR<E> &graph, Subgraph<E> &subgraph, float acc);
};

#endif	//	SUBGRAPH_GENERATOR_HPP



