Program 7: Dijkstra’s Algorithm
(A) Data Structures Defined
- an integer 2d array graph[v][v] is defined to represent the weighted graph using adjacency matrix.
- dist[v] is an integer array that stores shortest distances from the source node to each vertex.
- visited[v] is an integer array that tracks whether a vertex has been processed.
- v is the number of vertices (here set to 6).

(B) Functions
1: min_distance(int dist[], int visited[])
- finds the unvisited vertex with the smallest distance value.
- returns the index of that vertex.
2: dijkstra(int graph[v][v], int src)
- initializes all distances to INT_MAX and visited array to 0.
- sets the source distance to 0.
- repeats v-1 times:
- selects the vertex with minimum distance using min_distance.
- marks it visited.
- updates distances of its neighbors if a shorter path is found.
- prints the final shortest distances from the source to all vertices.

(C) The main() function
- defines the weighted graph as an adjacency matrix.
- calls dijkstra(graph, 0) to compute shortest paths from source vertex 0
- prints the results.

(D) Output
vertex   distance from source 0
0        0
1        4
2        12
3        19
4        21
5        16




