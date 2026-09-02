#include<stdio.h>
#include<stdlib.h>
#define MAX 5
typedef struct {
    int arr[MAX];
    int front;
    int rear;
    
}queue;
void init(queue *q)
{
    q->front =-1;
    q->rear=-1;
}
int isempty(queue *q)
{
    return q->front ==-1;//same
}
int isfull(queue *q)
{
    return q->rear==MAX-1;
    //q->front==(q->rear+1)%MAX
}
void enqueue(queue *q,int value)
{
    if(isfull(q))
    {
        printf("The queue id full\n");
        return;
    }
    if(q->front==-1)
    {
        q->rear=0;
        q->front=0;
    }
    else
    {
        q->rear++;//q->rear=(q->rear+1)%MAX
    }
      q->arr[q->rear]=value;
      printf("Enqueued %d\n",value);
      return;
}
int dequeue(queue *q)
{
    if(isempty(q))
    {
        printf("The queue is empty\n");
        return-1;
    }
    int value=q->arr[q->front];
    if(q->front==q->rear)
    {
        q->front=-1;
        q->rear=-1;
    }
    else{
        q->front++;//q->front=(q->front +1)%MAX
    }
    return value;

}
void display(queue *q)
{
    if(isempty(q))
    {
        printf("The queue is empty\n");
        return ;
    }/*int i=q->front;
    while(1)
    {
    printf("%d\n",q->arr[i]);
    if(i==q->rear) break;
    i=(i+1)%MAX;
    }*/
    
    for(int i=q->front;i<=q->rear;i++)
    {
        printf("%d\n",q->arr[i]);
    }
    printf("\n");
}
int main()
{
    queue q;
    init(&q);
     enqueue(&q, 10);
    enqueue(&q, 20);
    enqueue(&q, 30);
    enqueue(&q, 40);
    display(&q);

    printf("Dequeued: %d\n", dequeue(&q));
    display(&q);

    enqueue(&q, 50);
    enqueue(&q, 60);  // This will show "Queue is full"
    display(&q);

    return 0;


}
