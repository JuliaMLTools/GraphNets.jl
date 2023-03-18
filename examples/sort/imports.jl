using GraphNets
using Functors
using Flux
using SparseArrays
using BenchmarkTools
using CUDA
using Random
using ProgressMeter

#Graph Visualization
using Graphs
using GraphPlot
using Graphs: smallgraph
import Cairo, Fontconfig
#using LightGraphs