#include<stdio.h>
#include<stdlib.h>
#define MAX 10
typedef struct 
{
    char data;
    int priority;
    
}Element;
typedef struct {
    Element elements[MAX];
    int size;
    
}queue;
void init(queue *q)
{
    q->size=0;
}
int isEmpty(queue*q)
{
    return q->size==0;
}
int isFull(queue *q)
{
     return q->size==MAX;
}
void enqueue(queue *q,char data,int priority)
{
    int i;
    if(isFull(q))
    {
        printf("Full\n");
        return;
    }
    for(i=q->size-1;i>=0;i--)
    {
        if(q->elements[i].priority>priority)
        {
        q->elements[i+1]=q->elements[i];
        }
    else{
        break;
    }
}
    q->elements[i+1].data=data;
    q->elements[i+1].priority=priority;
    q->size++;
    return;

    }
char dequeue(queue *q)
{
    if(isEmpty(q))
    {
        printf("Empty\n");
        return 0 ;
    }
    char value=q->elements[0].data;
    int i;
    for (i=1;i<q->size;i++)
    {
        q->elements[i-1]=q->elements[i];

    }
    q->size --;
    return value;

}
void display(queue *q)
{
    for (int i=0;i<q->size;i++)
    {
        printf("%c  - %d\t",q->elements[i].data,q->elements[i].priority);

    }
    printf("\n");
    return;
    
}

int main() {
    queue q;
    init(&q);

    // Insert the given data
    enqueue(&q, 'A', 2);
    enqueue(&q, 'B', 5);
    enqueue(&q, 'X', 8);
    enqueue(&q, 'W', 6);
    enqueue(&q, 'C', 1);
    enqueue(&q, 'D', 4);
    enqueue(&q, 'E', 7);
    enqueue(&q, 'F', 3);

    display(&q);

    // Dequeue all to show ascending priority order
    printf("\nDequeuing all elements:\n");
    while (!isEmpty(&q)) {
        dequeue(&q);
        display(&q);
        printf("\n");
    }

    return 0;
}
