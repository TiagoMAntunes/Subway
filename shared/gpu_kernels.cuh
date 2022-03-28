
#include "globals.hpp"
#include "graph.cuh"
#include "subgraph.cuh"


__global__ void bfs_kernel(size_t numNodes,
							size_t from,
							size_t numPartitionedEdges,
							size_t *activeNodes,
							size_t *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *value,
							//bool *finished,
							bool *label1,
							bool *label2);

__global__ void cc_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2);

__global__ void sssp_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2);
							
__global__ void sswp_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2);
							
__global__ void pr_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							float *dist,
							float *delta,
							//bool *finished,
							float acc);						

__global__ void bfs_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2);	
							
__global__ void sssp_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2);
							
__global__ void sswp_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2);
							
__global__ void cc_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2);		
							
__global__ void pr_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							float *dist,
							float *delta,
							bool *finished,
							float acc);	

__global__ void clearLabel(size_t * activeNodes, bool *label, size_t size, size_t from);

__global__ void mixLabels(size_t * activeNodes, bool *label1, bool *label2, size_t size, size_t from);

__global__ void moveUpLabels(size_t * activeNodes, bool *label1, bool *label2, size_t size, size_t from);


