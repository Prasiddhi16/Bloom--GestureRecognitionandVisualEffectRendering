#include<stdio.h>
#include<stdlib.h>
#define MAX 10
int mat[MAX][MAX];
int visit[MAX];
void add(int v, int w )
{
    mat[v][w]=1;
    mat[w][v]=1;
}
void bfs(int start, int vertices)
{
    int queue[MAX],front=0,rear=0;
    for(int i=0;i<vertices;i++)
    {
        visit[i]=0;
    }
    visit[start]=1;
    queue[rear++]=start;
    printf("Breadth First Search \n");
    while(front<rear)
    {
        int node=queue[front++];
        printf("%d ",node);
        for(int i=0;i<vertices;i++)
        {
            if(mat[node][i]==1 && !visit[i])
            {
                queue[rear++]=i;
                visit[i]=1;
            }
        }

    }
    printf("\n");
}
void dfsu(int node, int vertices) {
    visit[node] = 1;
    printf("%d ", node);

    for (int i = 0; i < vertices; i++) {
        if (mat[node][i] == 1 && !visit[i]) {
            dfsu(i, vertices);
        }
    }
}

void dfs(int start, int vertices) {
    for (int i = 0; i < vertices; i++) visit[i] = 0;
    printf("DFS Traversal: ");
    dfsu(start, vertices);
    printf("\n");
}

int main()
{
    int vertices=5;
    for (int i=0;i<vertices;i++)
    {
        for(int j=0;j<vertices;j++)
        {
            mat[i][j]=0;
        }
    }
    add(0, 1);
    add(0, 2);
    add(1, 3);
    add(2, 4);
    bfs(0,vertices);
    dfs(0,vertices);
    return 0;
}
