#include "../shared/globals.hpp"
#include "../shared/timer.hpp"
#include "../shared/argument_parsing.cuh"
#include "../shared/graph.cuh"
#include "../shared/subgraph.cuh"
#include "../shared/partitioner.cuh"
#include "../shared/subgraph_generator.cuh"
#include "../shared/gpu_error_check.cuh"
#include "../shared/gpu_kernels.cuh"
#include "../shared/subway_utilities.hpp"
#include "../shared/test.cuh"
#include "../shared/test.cu"


int main(int argc, char** argv)
{
	
	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);
	
	Timer timer;
	timer.Start();
	
	GraphPR<OutEdge> graph(arguments.input, true);
	graph.ReadGraph();
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
	
	//for(unsigned int i=0; i<100; i++)
	//	cout << graph.edgeList[i].end << " " << graph.edgeList[i].w8;
	
	float initPR = 0.15;
	float acc = 0.01;
	float alpha = 0.85;
	
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.delta[i] = initPR;
		graph.value[i] = 0;
	}
	//graph.value[arguments.sourceNode] = 0;
	//graph.label[arguments.sourceNode] = true;


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_delta, graph.delta, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	
	Subgraph<OutEdge> subgraph(graph.num_nodes, graph.num_edges);
	
	SubgraphGenerator<OutEdge> subgen(graph);
	
	subgen.generate(graph, subgraph, acc, alpha);	

	Partitioner<OutEdge> partitioner;
	
	timer.Start();
	
	uint gItr = 0;
	
	bool finished;
	bool *d_finished;
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
		
	while (subgraph.numActiveNodes>0)
	{
		gItr++;
		
		partitioner.partition(subgraph, subgraph.numActiveNodes);
		// a super iteration
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();

			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			//mixLabels<<<partitioner.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			
			uint itr = 0;
			do
			{
				itr++;
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				pr_async<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
													partitioner.fromNode[i],
													partitioner.fromEdge[i],
													subgraph.d_activeNodes,
													subgraph.d_activeNodesPointer,
													subgraph.d_activeEdgeList,
													graph.d_outDegree,
													graph.d_value,
													graph.d_delta,
													d_finished,
													acc);		
																						

				cudaDeviceSynchronize();
				gpuErrorcheck( cudaPeekAtLastError() );	
				
				gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
			}while(!(finished));
			
			cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;			
		}
		
		subgen.generate(graph, subgraph, acc, alpha);
			
	}	
	
	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";
	
	gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, graph.num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
	
	utilities::PrintResults(graph.value, min(30, (uint)graph.num_nodes));
	
		
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
}

