
#include "gpu_kernels.cuh"
#include "globals.hpp"
#include "gpu_error_check.cuh"
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
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int sourceWeight = value[id];

		size_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		size_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(size_t i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			finalDist = sourceWeight + 1;
			if(finalDist < value[edgeList[i].end])
			{
				atomicMin(&value[edgeList[i].end] , finalDist);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void cc_kernel(size_t numNodes,
							size_t from,
							size_t numPartitionedEdges,
							size_t *activeNodes,
							size_t *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		size_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		size_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		//unsigned int finalDist;
		
		for(size_t i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			if(sourceWeight < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , sourceWeight);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}


__global__ void sssp_kernel(size_t numNodes,
							size_t from,
							size_t numPartitionedEdges,
							size_t *activeNodes,
							size_t *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int sourceWeight = dist[id];

		size_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		size_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(size_t i=thisFrom; i<thisTo; i++)
		{	
			finalDist = sourceWeight + edgeList[i].w8;
			if(finalDist < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , finalDist);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void sswp_kernel(size_t numNodes,
							size_t from,
							size_t numPartitionedEdges,
							size_t *activeNodes,
							size_t *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		size_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		size_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(size_t i=thisFrom; i<thisTo; i++)
		{	
			finalDist = min(sourceWeight, edgeList[i].w8);
			if(finalDist > dist[edgeList[i].end])
			{
				atomicMax(&dist[edgeList[i].end] , finalDist);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void pr_kernel(size_t numNodes,
							size_t from,
							size_t numPartitionedEdges,
							size_t *activeNodes,
							size_t *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							float *dist,
							float *delta,
							//bool *finished,
							float acc)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		unsigned int degree = outDegree[id];
		float thisDelta = delta[id];

		if(thisDelta > acc)
		{
			dist[id] += thisDelta;
			
			if(degree != 0)
			{
				//*finished = false;
				
				float sourcePR = ((float) thisDelta / degree) * 0.85;

				size_t thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;
				size_t thisto = thisfrom + degree;
				
				for(size_t i=thisfrom; i<thisto; i++)
				{
					atomicAdd(&delta[edgeList[i].end], sourcePR);
				}				
			}
			
			atomicAdd(&delta[id], -thisDelta);
		}
		
	}
}


__global__ void bfs_async(size_t numNodes,
							size_t from,
							size_t numPartitionedEdges,
							size_t *activeNodes,
							size_t *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		size_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		size_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(size_t i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			finalDist = sourceWeight + 1;
			if(finalDist < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , finalDist);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void sssp_async(size_t numNodes,
							size_t from,
							size_t numPartitionedEdges,
							size_t *activeNodes,
							size_t *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		size_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		size_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(size_t i=thisFrom; i<thisTo; i++)
		{	
			finalDist = sourceWeight + edgeList[i].w8;
			if(finalDist < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , finalDist);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void sswp_async(size_t numNodes,
							size_t from,
							size_t numPartitionedEdges,
							size_t *activeNodes,
							size_t *activeNodesPointer,
							OutEdgeWeighted *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		size_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		size_t thisTo = thisFrom + degree;
		
		
		unsigned int finalDist;
		
		for(size_t i=thisFrom; i<thisTo; i++)
		{	
			finalDist = min(sourceWeight, edgeList[i].w8);
			if(finalDist > dist[edgeList[i].end])
			{
				atomicMax(&dist[edgeList[i].end] , finalDist);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}


__global__ void cc_async(size_t numNodes,
							size_t from,
							size_t numPartitionedEdges,
							size_t *activeNodes,
							size_t *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int sourceWeight = dist[id];

		size_t thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		size_t thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		//unsigned int finalDist;
		
		for(size_t i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			if(sourceWeight < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , sourceWeight);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}


__global__ void pr_async(size_t numNodes,
							size_t from,
							size_t numPartitionedEdges,
							size_t *activeNodes,
							size_t *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							float *dist,
							float *delta,
							bool *finished,
							float acc)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		unsigned int degree = outDegree[id];
		float thisDelta = delta[id];

		if(thisDelta > acc)
		{
			dist[id] += thisDelta;
			
			if(degree != 0)
			{
				*finished = false;
				
				float sourcePR = ((float) thisDelta / degree) * 0.85;

				size_t thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;
				size_t thisto = thisfrom + degree;
				
				for(size_t i=thisfrom; i<thisto; i++)
				{
					atomicAdd(&delta[edgeList[i].end], sourcePR);
				}				
			}
			
			atomicAdd(&delta[id], -thisDelta);
		}
		
	}
}



__global__ void clearLabel(size_t * activeNodes, bool *label, size_t size, size_t from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
	{
		label[activeNodes[id+from]] = false;
	}
}

__global__ void mixLabels(size_t * activeNodes, bool *label1, bool *label2, size_t size, size_t from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		int nID = activeNodes[id+from];
		label1[nID] = label1[nID] || label2[nID];
		label2[nID] = false;	
	}
}

__global__ void moveUpLabels(size_t * activeNodes, bool *label1, bool *label2, size_t size, size_t from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int nID;
	if(id < size){
		nID = activeNodes[id+from];
		label1[nID] = label2[nID];
		label2[nID] = false;	
	}
}

