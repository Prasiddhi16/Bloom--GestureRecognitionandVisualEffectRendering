#include<stdio.h>
#include<stdlib.h>
struct Node 
{
    int data;
    struct Node *next;
};
struct Stack
{
    struct Stack *top;
};
void initstack(struct Stack *stack)
{
    stack->top=NULL;
}
int isEmpty(struct Stack *stack)
{
    return stack->top==NULL;
}
void push(struct Stack *stack, int data)
{
    struct Node *newnode=(struct Node *)malloc(sizeof(struct Node*));
    newnode->data=data;
    newnode->next=stack->top;
    stack->top=newnode;
    
}
void pop(struct Stack *stack)
{
    if (isEmpty(stack))
    {
        printf("The stack is empty\n");

    }
    struct Node *temp=stack->top;
    int popped=temp->data;
    stack->top=(temp)->next;
    free(temp);
    printf("Popped %d",popped);
}
int peek(struct Stack *stack)
{
    struct Node *temp=stack->top;
    int peek=temp->data;
    return peek;
}
void display(struct Stack *stack)
{
    if(isEmpty(stack))
    {
        printf("The stack is empty\n");
    }
    struct Node *temp=stack->top;
    while(temp!=NULL)
    {
        printf("%d\n",temp->data);
        temp=temp->next;
    }
    return;
}
int main()
{
    struct Stack stack;
    initstack( &stack);
    push(&stack,10);
     push(&stack,20);
      push(&stack,30);
       push(&stack,40);
        push(&stack,50);
        display(&stack);
        pop(&stack);
         pop(&stack);
          pop(&stack);
          peek(&stack);
          display(&stack);
          return 0;


}